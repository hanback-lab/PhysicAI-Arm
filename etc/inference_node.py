import os
import time
import threading
from pathlib import Path
 
import cv2
import numpy as np
import rclpy
import torch
import yaml
from cv_bridge import CvBridge
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
 
 
TOP_CALIB_PATH = "/home/soda/top_camera_calib.yaml"
FRONT_CALIB_PATH = "/home/soda/front_camera_calib.yaml"
 
 
ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
 
 
TOP_H, TOP_W = 168, 224
FRONT_H, FRONT_W = 168, 224
 
 
class Undistorter:
    def __init__(self, yaml_path):
        with Path(yaml_path).open("r") as file:
            data = yaml.safe_load(file)
 
        self.calib_size = (int(data["image_width"]), int(data["image_height"]))
        self.model = data["distortion_model"]
        self.K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
        self.D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
        self.P = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)[:, :3]
        self.maps = {}
 
        if self.model not in ("equidistant", "plumb_bob"):
            raise ValueError(f"Unsupported distortion_model: {self.model}")
 
    @staticmethod
    def _scale(matrix, sx, sy):
        scaled = matrix.copy()
        scaled[0, 0] *= sx
        scaled[0, 2] *= sx
        scaled[1, 1] *= sy
        scaled[1, 2] *= sy
        return scaled
 
    def _get_maps(self, size):
        if size in self.maps:
            return self.maps[size]
 
        sx = size[0] / self.calib_size[0]
        sy = size[1] / self.calib_size[1]
        if abs(sx - sy) > 1e-3:
            raise ValueError(
                f"Aspect ratio mismatch: image {size}, calibration {self.calib_size}"
            )
 
        K = self._scale(self.K, sx, sy)
        new_K = self._scale(self.P, sx, sy)
        R = np.eye(3)
 
        if self.model == "equidistant":
            maps = cv2.fisheye.initUndistortRectifyMap(
                K, self.D.reshape(4, 1), R, new_K, size, cv2.CV_16SC2
            )
        else:
            maps = cv2.initUndistortRectifyMap(
                K, self.D, R, new_K, size, cv2.CV_16SC2
            )
 
        self.maps[size] = maps
        return maps
 
    def __call__(self, image):
        height, width = image.shape[:2]
        map1, map2 = self._get_maps((width, height))
        return cv2.remap(
            image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
 
 
class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
 
        self.declare_parameter("command_deadband_rad", 0.000)
 
        self.command_deadband_rad = float(
            self.get_parameter("command_deadband_rad").value
        )
 
        self.last_command = None
        self._debug_command = None
        self._debug_time = None
        self._debug_failed = False
        self._debug_roi = {}
        self._debug_drag = None
 
        self.declare_parameter(
            "policy_path",
            "<id>/<policy_name>",
        )
 
        self.declare_parameter("device", "cuda")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("max_age_s", 0.5)
        self.declare_parameter("top_calib_path", TOP_CALIB_PATH)
        self.declare_parameter("front_calib_path", FRONT_CALIB_PATH)
        self.declare_parameter("show_images", True)
        self.declare_parameter("n_action_steps", 50)
 
        self.show_images = bool(
            self.get_parameter("show_images").value
        )
 
        self.policy_path = str(
            self.get_parameter("policy_path").value
        )
 
        self.device = str(
            self.get_parameter("device").value
        )
 
        self.fps = float(
            self.get_parameter("fps").value
        )
 
        self.max_age_s = float(
            self.get_parameter("max_age_s").value
        )
 
        self._calib = {
            "top": self._load_calibration(
                "top", str(self.get_parameter("top_calib_path").value)
            ),
            "front": self._load_calibration(
                "front", str(self.get_parameter("front_calib_path").value)
            ),
        }
 
        if (
            self.device == "cuda"
            and not torch.cuda.is_available()
        ):
            self.get_logger().warn(
                "CUDA is unavailable; using CPU."
            )
            self.device = "cpu"
 
        elif (
            self.device == "mps"
            and not torch.backends.mps.is_available()
        ):
            self.get_logger().warn(
                "MPS is unavailable; using CPU."
            )
            self.device = "cpu"
 
        self.bridge = CvBridge()
 
        self.top_image = None
        self.front_image = None
        self.current_joint = None
 
        self._rx_top = None
        self._rx_front = None
        self._rx_joints = None
 
        self._joint_idx = None
        self._joint_idx_ready = False
 
        self._busy = False
        self._step_count = 0
 
        self._data_lock = threading.Lock()
        self._policy_lock = threading.Lock()
 
        self.get_logger().info(
            f"Loading policy from: {self.policy_path}"
        )
 
        self.policy = ACTPolicy.from_pretrained(
            pretrained_name_or_path=self.policy_path,
            local_files_only=False,
        )
 
        self.policy.to(self.device)
        self.policy.eval()

        n_steps = int(self.get_parameter("n_action_steps").value)
        cfg = self.policy.config

        if cfg.temporal_ensemble_coeff is not None:
            self.get_logger().warn(
                "temporal_ensemble_coeff is set; n_action_steps must stay 1. Ignoring override."
            )
        elif not 1 <= n_steps <= cfg.chunk_size:
            raise ValueError(
                f"n_action_steps={n_steps} must be in [1, chunk_size={cfg.chunk_size}]"
            )
        else:
            cfg.n_action_steps = n_steps

        self.get_logger().info(
            f"chunk_size={cfg.chunk_size}, n_action_steps={cfg.n_action_steps}"
        )
 
        self.preprocessor, self.postprocessor = (
            
            make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=self.policy_path,
                preprocessor_overrides={
                    "device_processor": {
                        "device": self.device
                    },
                },
            )
        )
 
        self._reset_policy_state()
 
        self.create_subscription(
            Image,
            "/arm/top_cam",
            self._on_top_image,
            qos_profile_sensor_data,
        )
 
        self.create_subscription(
            Image,
            "/arm/front_cam",
            self._on_front_image,
            qos_profile_sensor_data,
        )
 
        self.create_subscription(
            JointState,
            "/follower/joint_states",
            self._on_joints,
            qos_profile_sensor_data,
        )
 
        self.target_pub = self.create_publisher(
            JointState,
            "/follower/joint_targets",
            10,
        )
 
        self.create_service(
            Trigger,
            "/inference/reset",
            self._on_reset,
        )
 
        self.create_timer(
            1.0 / self.fps,
            self.control_loop,
        )
 
        self.debug_pub = self.create_publisher(Image, "/inference/debug", 1)
        self.create_timer(0.2, self._show_debug)

        self.get_logger().info(
            f"Started: fps={self.fps}, "
            f"device={self.device}"
        )
 
    def _debug_mouse(self, event, x, y, flags, userdata):
        # Drag an ROI on either enlarged camera view to exclude baskets.
        if event == cv2.EVENT_RBUTTONDOWN:
            self._debug_roi.clear()
        if event == cv2.EVENT_LBUTTONDOWN and 40 <= y < 376 and 0 <= x < 896:
            self._debug_drag = (x // 448, x % 448, y - 40)
        if event == cv2.EVENT_LBUTTONUP and self._debug_drag is not None:
            cam, x0, y0 = self._debug_drag
            x1 = max(0, min(447, x - cam * 448))
            y1 = max(0, min(335, y - 40))
            image = self.top_image if cam == 0 else self.front_image
            if image is None:
                self._debug_drag = None
                return
            h,w = image.shape[:2]
            scale = min(448/w,336/h)
            dw,dh = round(w*scale),round(h*scale)
            x0,x1 = min(x0,dw),min(x1,dw)
            y0,y1 = min(y0,dh),min(y1,dh)
            if abs(x1-x0) > 10 and abs(y1-y0) > 10:
                self._debug_roi[cam] = (min(x0,x1)/dw, min(y0,y1)/dh,
                                        max(x0,x1)/dw, max(y0,y1)/dh)
            self._debug_drag = None

    def _debug_camera(self, rgb, cam):
        bgr = cv2.cvtColor((np.clip(rgb, 0, 1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 100, 65), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 65), (179, 255, 255))
        green = cv2.inRange(hsv, (35, 75, 50), (90, 255, 255))
        roi = self._debug_roi.get(cam)
        allowed = np.ones(bgr.shape[:2], dtype=np.uint8)*255
        if roi:
            allowed[:] = 0
            h, w = bgr.shape[:2]
            x0,y0,x1,y1 = (int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h))
            allowed[y0:y1, x0:x1] = 255
            cv2.rectangle(bgr, (x0,y0), (x1,y1), (255,255,0), 1)
        for mask, name, color in ((red, "R", (0,0,255)), (green, "G", (0,230,0))):
            mask = cv2.bitwise_and(mask, allowed)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 12:
                    continue
                x,y,w,h = cv2.boundingRect(contour)
                m = cv2.moments(contour)
                if m['m00'] <= 0:
                    continue
                cx,cy = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
                cv2.rectangle(bgr, (x,y), (x+w,y+h), color, 1)
                cv2.drawMarker(bgr, (cx,cy), color, cv2.MARKER_CROSS, 9, 1)
                cv2.putText(bgr, f"{name} {cx},{cy}", (x,max(10,y-3)), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        h, w = bgr.shape[:2]
        # Letterbox for display only; never change model input geometry.
        scale = min(448/w, 336/h)
        resized = cv2.resize(bgr, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((336,448,3), dtype=np.uint8)
        panel[:resized.shape[0], :resized.shape[1]] = resized
        return panel


    def _show_debug(self):
        with self._data_lock:
            if self.top_image is None or self.front_image is None:
                return

            top, front = self.top_image.copy(), self.front_image.copy()
            measured = None if self.current_joint is None else self.current_joint.copy()
            command = None if self._debug_command is None else self._debug_command.copy()
            command_time = self._debug_time
            stamps = (self._rx_top, self._rx_front, self._rx_joints)
        canvas = np.full((700,896,3), 25, dtype=np.uint8)
        def text(label, x, y, color=(225,225,225), scale=.55):
            cv2.putText(canvas, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        text("TOP - model input", 12, 26)
        text("FRONT - model input", 460, 26)
        canvas[40:376,:448] = self._debug_camera(top, 0)
        canvas[40:376,448:] = self._debug_camera(front, 1)
        text("Boxes = color candidates, NOT ACT-selected targets", 12, 399, (0,210,255))
        text("Drag ROI to exclude baskets; right-click clears ROI", 12, 423)
        text("Joint                 measured       sent target       target-current", 12, 457)
        for i,name in enumerate(ARM_JOINTS):
            if measured is None or command is None:
                line = f"{name:18s}  waiting for joint / command data"
            else:
                line = f"{name:18s}  {measured[i]:+9.4f}      {command[i]:+9.4f}        {command[i]-measured[i]:+9.4f}"
            text(line, 12, 485+i*26, scale=.51)
        age = None if command_time is None else time.monotonic()-command_time
        now = self.get_clock().now()
        stale = any(t is None or (now-t).nanoseconds*1e-9 > self.max_age_s for t in stamps)
        status = "INPUT STALE / MISSING" if stale else "INPUT FRESH"
        text(status + (" | no command yet" if age is None else f" | command age {age:.2f}s"), 12, 652,
             (0,100,255) if stale else (100,230,120))
        text("Values use JointState units. No Cartesian target available without FK.", 12, 680, scale=.5)
        msg = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        msg.header.stamp = now.to_msg()
        self.debug_pub.publish(msg)
        if not self.show_images or self._debug_failed:
            return
        if os.name != "nt" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self._debug_failed = True
            self.get_logger().warn("No display; view /inference/debug in an image viewer.")
            return
        try:
            cv2.imshow("ACT debug", canvas)
            cv2.setMouseCallback("ACT debug", self._debug_mouse)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._debug_failed = True
            self.get_logger().warn(f"Debug window unavailable: {exc}; use /inference/debug")

    def _reset_policy_state(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
 
    def _on_reset(self, request, response):
        del request
 
        with self._policy_lock:
            self._reset_policy_state()
 
        self._step_count = 0
 
        response.success = True
        response.message = (
            "ACT action queue reset. "
            "Ready for a new attempt."
        )
 
        self.get_logger().info(
            response.message
        )
 
        return response
 
    def _load_calibration(self, cam, path):
        calib_path = Path(path)
 
        if not calib_path.exists():
            self.get_logger().warn(
                f"[{cam}] No calibration file at "
                f"{calib_path}; "
                f"running without undistortion."
            )
 
            return None
 
        try:
            undistorter = Undistorter(calib_path)
 
            self.get_logger().info(
                f"[{cam}] Loaded calibration: "
                f"{calib_path} "
                f"(model={undistorter.model}, "
                f"size={undistorter.calib_size})"
            )
 
            return undistorter
 
        except Exception as exc:
            self.get_logger().warn(
                f"[{cam}] Failed to read calibration "
                f"({exc}); "
                f"running without undistortion."
            )
 
            return None
 
    def _undistort_image(self, img, cam):
        undistorter = self._calib.get(cam)
 
        if undistorter is None:
            return img
 
        return undistorter(img)
 
    def _preprocess_image(
        self,
        img_bgr,
        out_h,
        out_w,
        cam,
    ):
        img_bgr = self._undistort_image(
            img_bgr,
            cam,
        )
 
        img_bgr = cv2.resize(
            img_bgr,
            (out_w, out_h),
        )
 
        img_rgb = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB,
        )
 
        return np.ascontiguousarray(
            img_rgb.astype(
                np.float32
            )
            / 255.0
        )
 
    def _on_top_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            image = self._preprocess_image(
                img,
                TOP_H,
                TOP_W,
                "top",
            )

            # 전처리된 영상의 리더암 영역을 회색으로 마스킹
            image[0:95, 0:70, :] = 0.5

            with self._data_lock:
                self.top_image = image
                self._rx_top = self.get_clock().now()

        except Exception as exc:
            self.get_logger().error(f"Top image error: {exc}")
 
    def _on_front_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
 
            image = self._preprocess_image(
                img,
                FRONT_H,
                FRONT_W,
                "front",
            )
 
            with self._data_lock:
                self.front_image = image
                self._rx_front = (
                    self.get_clock().now()
                )
 
        except Exception as exc:
            self.get_logger().error(
                f"Front image error: {exc}"
            )
 
    def _on_joints(self, msg):
        if (
            not self._joint_idx_ready
            and not self._init_joint_idx(msg)
        ):
            return
 
        try:
            joint = np.array(
                [
                    msg.position[i]
                    for i in self._joint_idx
                ],
                dtype=np.float32,
            )
 
        except IndexError:
            self.get_logger().error(
                "JointState position is "
                "shorter than name."
            )
 
            return
 
        with self._data_lock:
            self.current_joint = joint
 
            self._rx_joints = (
                self.get_clock().now()
            )
 
    def _init_joint_idx(self, msg):
        name_to_idx = {
            name: i
            for i, name in enumerate(
                msg.name
            )
        }
 
        missing = [
            joint
            for joint in ARM_JOINTS
            if joint not in name_to_idx
        ]
 
        if missing:
            self.get_logger().error(
                f"Missing joints: {missing}. "
                f"Available: {list(msg.name)}"
            )
 
            return False
 
        self._joint_idx = [
            name_to_idx[joint]
            for joint in ARM_JOINTS
        ]
 
        self._joint_idx_ready = True
 
        self.get_logger().info(
            f"Joint mapping: "
            f"{ARM_JOINTS} "
            f"-> {self._joint_idx}"
        )
 
        return True
 
    def _snapshot(self):
        with self._data_lock:
            if (
                self.top_image is None
                or self.front_image is None
                or self.current_joint is None
            ):
                return None
 
            now = self.get_clock().now()
 
            stamps = (
                self._rx_top,
                self._rx_front,
                self._rx_joints,
            )
 
            if any(
                stamp is None
                for stamp in stamps
            ):
                return None
 
            if any(
                (
                    now - stamp
                ).nanoseconds
                * 1e-9
                > self.max_age_s
                for stamp in stamps
            ):
                return "stale"
 
            return (
                self.top_image.copy(),
                self.front_image.copy(),
                self.current_joint.copy(),
            )
 
    def _make_batch(self, snapshot):
        (
            top_image,
            front_image,
            joint_state,
        ) = snapshot
 
        top = (
            torch.from_numpy(
                top_image
            )
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
            .to(self.device)
        )
 
        front = (
            torch.from_numpy(
                front_image
            )
            .permute(
                2,
                0,
                1,
            )
            .unsqueeze(0)
            .to(self.device)
        )
 
        state = (
            torch.from_numpy(
                joint_state
            )
            .unsqueeze(0)
            .to(self.device)
        )
 
        return {
            "observation.images.top": top,
            "observation.images.front": front,
            "observation.state": state,
        }
 
    def _run_inference(
        self,
        batch,
        measured_joint,
    ):
        try:
            with (
                self._policy_lock,
                torch.inference_mode(),
            ):
                batch = self.preprocessor(
                    batch
                )
 
                action = (
                    self.policy
                    .select_action(batch)
                )
 
                action = (
                    self.postprocessor(
                        action
                    )
                )
 
            action = (
                action
                .detach()
                .cpu()
                .numpy()
            )
 
            if action.ndim == 2:
                action = action[0]
 
            action = (
                action.astype(
                    np.float32
                )
                [:len(ARM_JOINTS)]
            )
 
            if (
                action.shape[0]
                != len(ARM_JOINTS)
                or not np.all(
                    np.isfinite(action)
                )
            ):
                self.get_logger().error(
                    f"Invalid action: "
                    f"{action}"
                )
 
                return
 
            if self.last_command is not None:
                small_change = (
                    np.abs(
                        action
                        - self.last_command
                    )
                    < self.command_deadband_rad
                )
 
                action[
                    small_change
                ] = self.last_command[
                    small_change
                ]
 
            self.last_command = (
                action.copy()
            )
 
            msg = JointState()
 
            msg.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )
 
            msg.name = ARM_JOINTS
 
            msg.position = [
                float(value)
                for value in action
            ]
 
            self.target_pub.publish(
                msg
            )
            with self._data_lock:
                self._debug_command = action.copy()
                self._debug_time = time.monotonic()
 
            self._step_count += 1
 
            if (
                self._step_count
                % 10
                == 0
            ):
                self.get_logger().info(
                    f"measured="
                    f"{np.round(measured_joint, 3)} "
                    f"target="
                    f"{np.round(action, 3)}"
                )
 
        except Exception:
            import traceback
 
            self.get_logger().error(
                traceback.format_exc()
            )
 
        finally:
            with self._data_lock:
                self._busy = False
 
    def control_loop(self):
        snapshot = self._snapshot()
 
        if snapshot is None:
            self.get_logger().warn(
                "Waiting for sensor data...",
                throttle_duration_sec=2.0,
            )
 
            return
 
        if snapshot == "stale":
            self.get_logger().warn(
                "Stale sensor data.",
                throttle_duration_sec=1.0,
            )
 
            return
 
        with self._data_lock:
            if self._busy:
                return
 
            self._busy = True
 
        batch = self._make_batch(
            snapshot
        )
 
        thread = threading.Thread(
            target=self._run_inference,
            args=(
                batch,
                snapshot[2],
            ),
            daemon=True,
        )
 
        thread.start()
 
 
def main(args=None):
    rclpy.init(args=args)
 
    node = InferenceNode()
 
    try:
        rclpy.spin(node)
 
    except KeyboardInterrupt:
        pass
 
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        node.destroy_node()
 
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == "__main__":
    main()
 