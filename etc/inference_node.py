import json
import threading
from pathlib import Path

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger


CALIB_PATH = "./<calibration_json_path>"


ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

TOP_H, TOP_W = 128, 128
FRONT_H, FRONT_W = 128, 128


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")
        self.declare_parameter("command_deadband_rad", 0.000)
        self.command_deadband_rad = float(
            self.get_parameter("command_deadband_rad").value
        )
        self.last_command = None
        self.declare_parameter("policy_path", "<user_name>/<policy_path>")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("max_age_s", 0.5)
        self.declare_parameter("calib_path", CALIB_PATH)

        self.policy_path = str(self.get_parameter("policy_path").value)
        self.device = str(self.get_parameter("device").value)
        self.fps = float(self.get_parameter("fps").value)
        self.max_age_s = float(self.get_parameter("max_age_s").value)
        self.calib_path = Path(str(self.get_parameter("calib_path").value))

        self._calib = self._load_calibration()
        self._undistort_maps = {}

        if self.device == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA is unavailable; using CPU.")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            self.get_logger().warn("MPS is unavailable; using CPU.")
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

        self.get_logger().info(f"Loading policy from: {self.policy_path}")
        self.policy = ACTPolicy.from_pretrained(
            pretrained_name_or_path=self.policy_path,
            local_files_only=False,
        )
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.policy_path,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )
        self._reset_policy_state()

        self.create_subscription(
            Image, "/arm/top_cam", self._on_top_image, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, "/arm/front_cam", self._on_front_image, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState,
            "/follower/joint_states",
            self._on_joints,
            qos_profile_sensor_data,
        )
        self.target_pub = self.create_publisher(
            JointState, "/follower/joint_targets", 10
        )
        self.create_service(
            Trigger,
            "/inference/reset",
            self._on_reset,
        )
        self.create_timer(1.0 / self.fps, self.control_loop)
        self.get_logger().info(
            f"Started: fps={self.fps}, device={self.device}"
        )

    def _reset_policy_state(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _on_reset(self, request, response):
        del request
        with self._policy_lock:
            self._reset_policy_state()
        self._step_count = 0
        response.success = True
        response.message = "ACT action queue reset. Ready for a new attempt."
        self.get_logger().info(response.message)
        return response

    def _load_calibration(self):
        """캘리브레이션 JSON을 읽어 반환. 파일이 없거나 잘못되면 None."""
        if not self.calib_path.exists():
            self.get_logger().info(
                f"No calibration file at {self.calib_path}; "
                f"running without undistortion."
            )
            return None
        try:
            with open(self.calib_path, "r") as f:
                calib = json.load(f)
            self.get_logger().info(f"Loaded calibration: {self.calib_path}")
            return calib
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to read calibration ({exc}); running without undistortion."
            )
            return None

    def _calib_for_camera(self, cam):
        """카메라별 섹션({"top": {...}, "front": {...}})이 있으면 그것을,
        없으면 공용 K/D를 사용한다. K/D가 없으면 None."""
        if self._calib is None:
            return None
        section = self._calib.get(cam, self._calib)
        if "K" not in section or "D" not in section:
            return None
        return section

    def _undistort_image(self, img, cam):
        """캘리브레이션이 있으면 fisheye 왜곡을 보정, 없으면 원본을 그대로 반환."""
        section = self._calib_for_camera(cam)
        if section is None:
            return img

        h, w = img.shape[:2]
        key = (cam, w, h)
        maps = self._undistort_maps.get(key)
        if maps is None:
            K = np.array(section["K"], dtype=np.float64).reshape(3, 3)
            D = np.array(section["D"], dtype=np.float64).reshape(-1, 1)
            balance = float(section.get("balance", 0.0))
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, (w, h), np.eye(3), balance=balance
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
            )
            maps = (map1, map2)
            self._undistort_maps[key] = maps

        return cv2.remap(
            img, maps[0], maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def _preprocess_image(self, img_bgr, out_h, out_w, cam):
        img_bgr = self._undistort_image(img_bgr, cam)
        img_bgr = cv2.resize(img_bgr, (out_w, out_h))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img_rgb.astype(np.float32) / 255.0)

    def _on_top_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image = self._preprocess_image(img, TOP_H, TOP_W, "top")
            with self._data_lock:
                self.top_image = image
                self._rx_top = self.get_clock().now()
        except Exception as exc:
            self.get_logger().error(f"top image error: {exc}")

    def _on_front_image(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image = self._preprocess_image(img, FRONT_H, FRONT_W, "front")
            with self._data_lock:
                self.front_image = image
                self._rx_front = self.get_clock().now()
        except Exception as exc:
            self.get_logger().error(f"front image error: {exc}")

    def _on_joints(self, msg):
        if not self._joint_idx_ready and not self._init_joint_idx(msg):
            return
        try:
            joint = np.array(
                [msg.position[i] for i in self._joint_idx], dtype=np.float32
            )
        except IndexError:
            self.get_logger().error("JointState position is shorter than name.")
            return

        with self._data_lock:
            self.current_joint = joint
            self._rx_joints = self.get_clock().now()

    def _init_joint_idx(self, msg):
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        missing = [joint for joint in ARM_JOINTS if joint not in name_to_idx]
        if missing:
            self.get_logger().error(
                f"Missing joints: {missing}. Available: {list(msg.name)}"
            )
            return False
        self._joint_idx = [name_to_idx[joint] for joint in ARM_JOINTS]
        self._joint_idx_ready = True
        self.get_logger().info(
            f"Joint mapping: {ARM_JOINTS} -> {self._joint_idx}"
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
            stamps = (self._rx_top, self._rx_front, self._rx_joints)
            if any(stamp is None for stamp in stamps):
                return None
            if any(
                (now - stamp).nanoseconds * 1e-9 > self.max_age_s
                for stamp in stamps
            ):
                return "stale"
            return (
                self.top_image.copy(),
                self.front_image.copy(),
                self.current_joint.copy(),
            )

    def _make_batch(self, snapshot):
        top_image, front_image, joint_state = snapshot
        top = (
            torch.from_numpy(top_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        front = (
            torch.from_numpy(front_image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        state = torch.from_numpy(joint_state).unsqueeze(0).to(self.device)
        return {
            "observation.images.top": top,
            "observation.images.front": front,
            "observation.state": state,
        }

    def _run_inference(self, batch, measured_joint):
        try:
            with self._policy_lock, torch.inference_mode():
                batch = self.preprocessor(batch)
                action = self.policy.select_action(batch)
                action = self.postprocessor(action)

            action = action.detach().cpu().numpy()
            if action.ndim == 2:
                action = action[0]
            action = action.astype(np.float32)[: len(ARM_JOINTS)]
            if action.shape[0] != len(ARM_JOINTS) or not np.all(np.isfinite(action)):
                self.get_logger().error(f"Invalid action: {action}")
                return

            if self.last_command is not None:
                small_change = np.abs(action - self.last_command) < self.command_deadband_rad
                action[small_change] = self.last_command[small_change]

            self.last_command = action.copy()

            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ARM_JOINTS
            msg.position = [float(value) for value in action]
            self.target_pub.publish(msg)

            self._step_count += 1
            if self._step_count % 10 == 0:
                self.get_logger().info(
                    f"measured={np.round(measured_joint, 3)} "
                    f"target={np.round(action, 3)}"
                )
        except Exception:
            import traceback

            self.get_logger().error(traceback.format_exc())
        finally:
            with self._data_lock:
                self._busy = False

    def control_loop(self):
        snapshot = self._snapshot()
        if snapshot is None:
            self.get_logger().warn(
                "Waiting for sensor data...", throttle_duration_sec=2.0
            )
            return
        if snapshot == "stale":
            self.get_logger().warn(
                "Stale sensor data.", throttle_duration_sec=1.0
            )
            return

        with self._data_lock:
            if self._busy:
                return
            self._busy = True

        batch = self._make_batch(snapshot)
        thread = threading.Thread(
            target=self._run_inference,
            args=(batch, snapshot[2]),
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
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()