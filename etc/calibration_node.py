import glob
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


class CalibNode(Node):
    def __init__(self):
        super().__init__("calib_node")

        self.declare_parameter("topic", "/arm/top_cam")
        self.declare_parameter("save_dir", "top_calib_images")
        self.declare_parameter("out", "top_camera_calib.yaml")
        self.declare_parameter("camera_name", "top_cam")
        self.declare_parameter("cols", 8)
        self.declare_parameter("rows", 6)
        self.declare_parameter("square", 0.025)
        self.declare_parameter("model", "fisheye")
        self.declare_parameter("balance", 0.0)
        self.declare_parameter("max_error", 1.0)
        self.declare_parameter("min_images", 15)
        self.declare_parameter("min_diff", 0.05)
        self.declare_parameter("auto_interval", 1.0)

        gp = lambda n: self.get_parameter(n).value
        self.topic = str(gp("topic"))
        self.save_dir = str(gp("save_dir"))
        self.out = str(gp("out"))
        self.camera_name = str(gp("camera_name"))
        self.pattern = (int(gp("cols")), int(gp("rows")))
        self.square = float(gp("square"))
        self.model = str(gp("model"))
        self.balance = float(gp("balance"))
        self.max_error = float(gp("max_error"))
        self.min_images = int(gp("min_images"))
        self.min_diff = float(gp("min_diff"))
        self.auto_interval = float(gp("auto_interval"))

        if self.model not in ("fisheye", "pinhole"):
            raise ValueError("model must be 'fisheye' or 'pinhole'")

        os.makedirs(self.save_dir, exist_ok=True)

        n = self.pattern[0] * self.pattern[1]
        self.objp = np.zeros((n, 1, 3), np.float64)
        self.objp[:, 0, :2] = (
            np.mgrid[0:self.pattern[0], 0:self.pattern[1]].T.reshape(-1, 2) * self.square
        )

        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_size = None
        self.samples = []
        self.auto_mode = False
        self.show_undistort = False
        self.maps = None
        self.last_save_time = 0.0
        self.status_msg = ""
        self.quit = False

        self.load_existing()

        self.create_subscription(Image, self.topic, self.on_image, 10)
        self.create_timer(1.0 / 30.0, self.loop)

        self.get_logger().info(f"topic={self.topic} save_dir={self.save_dir} model={self.model}")
        self.get_logger().info("keys: s=save  a=auto  c=calibrate  u=undistort view  q=quit")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern, DETECT_FLAGS)
        if not found:
            return None
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
        return corners.reshape(-1, 1, 2).astype(np.float64)

    def load_existing(self):
        paths = sorted(glob.glob(os.path.join(self.save_dir, "*.png")))
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            size = (img.shape[1], img.shape[0])
            if self.image_size is None:
                self.image_size = size
            elif size != self.image_size:
                continue
            corners = self.detect(img)
            if corners is not None:
                self.samples.append((path, corners))
        if paths:
            self.get_logger().info(f"loaded {len(self.samples)}/{len(paths)} existing images")

    def is_new_view(self, corners):
        diag = np.hypot(*self.image_size)
        for _, prev in self.samples:
            if np.mean(np.linalg.norm(corners - prev, axis=2)) / diag < self.min_diff:
                return False
        return True

    def save_sample(self, frame, corners):
        index = 0
        while os.path.exists(os.path.join(self.save_dir, f"{index:04d}.png")):
            index += 1
        path = os.path.join(self.save_dir, f"{index:04d}.png")
        cv2.imwrite(path, frame)
        self.samples.append((path, corners))
        self.last_save_time = time.monotonic()
        self.get_logger().info(f"saved {path} ({len(self.samples)} total)")

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"image convert error: {exc}")
            return
        size = (frame.shape[1], frame.shape[0])
        if self.image_size is None:
            self.image_size = size
        elif size != self.image_size:
            self.get_logger().warn(
                f"frame size {size} != {self.image_size}, ignored", throttle_duration_sec=5.0
            )
            return
        self.latest_frame = frame

    def calibrate_once(self, obj_pts, img_pts):
        if self.model == "fisheye":
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            return cv2.fisheye.calibrate(
                obj_pts, img_pts, self.image_size, np.zeros((3, 3)), np.zeros((4, 1)),
                flags=flags, criteria=criteria,
            )
        return cv2.calibrateCamera(
            [o.astype(np.float32) for o in obj_pts],
            [i.astype(np.float32) for i in img_pts],
            self.image_size, None, None,
        )

    def view_errors(self, obj_pts, img_pts, rvecs, tvecs, K, D):
        errs = []
        for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
            if self.model == "fisheye":
                proj, _ = cv2.fisheye.projectPoints(o, r, t, K, D)
            else:
                proj, _ = cv2.projectPoints(o, r, t, K, D)
            errs.append(np.sqrt(np.mean(np.sum((i - proj) ** 2, axis=2))))
        return np.array(errs)

    def run_calibration(self):
        if len(self.samples) < self.min_images:
            self.status_msg = f"need {self.min_images} images (have {len(self.samples)})"
            self.get_logger().warn(self.status_msg)
            return

        self.get_logger().info(f"calibrating with {len(self.samples)} images...")
        paths = [p for p, _ in self.samples]
        obj_pts = [self.objp.copy() for _ in self.samples]
        img_pts = [c for _, c in self.samples]

        try:
            rms, K, D, rvecs, tvecs = self.calibrate_once(obj_pts, img_pts)
            errs = self.view_errors(obj_pts, img_pts, rvecs, tvecs, K, D)

            bad = errs > self.max_error
            if bad.any() and (~bad).sum() >= 10:
                for p, e in zip(paths, errs):
                    if e > self.max_error:
                        self.get_logger().warn(f"drop {os.path.basename(p)} err={e:.3f}")
                keep = np.where(~bad)[0]
                obj_pts = [obj_pts[i] for i in keep]
                img_pts = [img_pts[i] for i in keep]
                rms, K, D, rvecs, tvecs = self.calibrate_once(obj_pts, img_pts)
                errs = self.view_errors(obj_pts, img_pts, rvecs, tvecs, K, D)
        except cv2.error as exc:
            self.status_msg = "calibration failed (see log)"
            self.get_logger().error(f"calibration failed: {exc}")
            return

        eye = np.eye(3)
        if self.model == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, self.image_size, eye, balance=self.balance
            )
            self.maps = cv2.fisheye.initUndistortRectifyMap(
                K, D, eye, new_K, self.image_size, cv2.CV_16SC2
            )
            dist_model = "equidistant"
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, self.image_size, self.balance, self.image_size)
            self.maps = cv2.initUndistortRectifyMap(K, D, eye, new_K, self.image_size, cv2.CV_16SC2)
            dist_model = "plumb_bob"

        P = np.hstack([new_K, np.zeros((3, 1))])
        self.write_yaml(K, D, dist_model, eye, P)

        self.status_msg = f"RMS {rms:.3f}px  mean {errs.mean():.3f}  max {errs.max():.3f}  ({len(obj_pts)} imgs)"
        self.get_logger().info(self.status_msg)
        self.get_logger().info(f"K=\n{K}\nD={D.ravel()}")
        self.get_logger().info(f"saved {self.out}  (press 'u' to check undistortion)")

    def write_yaml(self, K, D, dist_model, R, P):
        fmt = lambda v: "[" + ", ".join(f"{x:.8f}" for x in np.ravel(v)) + "]"
        D = np.ravel(D)
        w, h = self.image_size
        with open(self.out, "w") as f:
            f.write(f"image_width: {w}\n")
            f.write(f"image_height: {h}\n")
            f.write(f"camera_name: {self.camera_name}\n")
            f.write(f"camera_matrix:\n  rows: 3\n  cols: 3\n  data: {fmt(K)}\n")
            f.write(f"distortion_model: {dist_model}\n")
            f.write(f"distortion_coefficients:\n  rows: 1\n  cols: {D.size}\n  data: {fmt(D)}\n")
            f.write(f"rectification_matrix:\n  rows: 3\n  cols: 3\n  data: {fmt(R)}\n")
            f.write(f"projection_matrix:\n  rows: 3\n  cols: 4\n  data: {fmt(P)}\n")

    def draw_coverage(self, display):
        overlay = display.copy()
        for _, corners in self.samples:
            hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
            cv2.fillConvexPoly(overlay, hull, (255, 160, 0))
        cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

    def put(self, img, text, y, color):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def loop(self):
        if self.latest_frame is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.quit = True
            return

        frame = self.latest_frame.copy()

        if self.show_undistort and self.maps is not None:
            display = cv2.remap(frame, self.maps[0], self.maps[1], cv2.INTER_LINEAR)
            self.put(display, "UNDISTORTED  (u: back)", 25, (0, 255, 255))
            self.put(display, self.status_msg, 50, (255, 255, 255))
            cv2.imshow(f"calib [{self.topic}]", display)
            self.handle_key(cv2.waitKey(1) & 0xFF, frame, None)
            return

        corners = self.detect(frame)
        display = frame.copy()
        self.draw_coverage(display)

        if corners is not None:
            cv2.drawChessboardCorners(display, self.pattern, corners.astype(np.float32), True)
            new_view = self.is_new_view(corners)
            text, color = ("FOUND - new view", (0, 255, 0)) if new_view else ("FOUND - too similar", (0, 200, 255))
        else:
            new_view = False
            text, color = "not found", (0, 0, 255)

        self.put(display, text, 25, color)
        self.put(display, f"saved: {len(self.samples)}/{self.min_images}  auto: {'ON' if self.auto_mode else 'off'}", 50, (255, 255, 0))
        if self.status_msg:
            self.put(display, self.status_msg, 75, (255, 255, 255))
        cv2.imshow(f"calib [{self.topic}]", display)

        now = time.monotonic()
        if self.auto_mode and new_view and now - self.last_save_time > self.auto_interval:
            self.save_sample(frame, corners)

        self.handle_key(cv2.waitKey(1) & 0xFF, frame, corners if new_view else None)

    def handle_key(self, key, frame, corners):
        if key == ord("s"):
            if corners is not None and time.monotonic() - self.last_save_time > 0.5:
                self.save_sample(frame, corners)
        elif key == ord("a"):
            self.auto_mode = not self.auto_mode
            self.get_logger().info(f"auto mode: {self.auto_mode}")
        elif key == ord("c"):
            self.auto_mode = False
            self.run_calibration()
        elif key == ord("u"):
            if self.maps is None:
                self.status_msg = "calibrate first (c)"
            else:
                self.show_undistort = not self.show_undistort
        elif key == ord("q"):
            self.quit = True


def main(args=None):
    rclpy.init(args=args)
    node = CalibNode()
    try:
        while rclpy.ok() and not node.quit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()