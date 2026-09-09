import os
import json
import csv
import threading
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2

EPISODE_TIMEOUT_SEC = 90.0


class LoggerNode(Node):
    def __init__(self):
        super().__init__("logger_node")

        self.declare_parameter("save_dir", "<dir_name>")
        self.declare_parameter("episode_id", 0)
        self.declare_parameter("task_name", "<task_name>")

        self.save_dir  = self.get_parameter("save_dir").value
        self.task_name = self.get_parameter("task_name").value
        param_episode_id = int(self.get_parameter("episode_id").value)

        self.episode_id = self.get_next_episode_id() if param_episode_id <= 0 else param_episode_id

        self.bridge             = CvBridge()
        self.front_frame_count  = 0
        self.top_frame_count    = 0
        self.state_row_count    = 0
        self.action_row_count   = 0
        self.start_time         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.running            = True
        self.recording          = True
        self.episode_finished   = False
        self.episode_start_time = None
        self.latest_ee          = None

        self.setup_episode_directory()
        self.init_files()

        self.create_subscription(Image,      "/arm/front_cam",          self.front_cb,  10)
        self.create_subscription(Image,      "/arm/top_cam",            self.top_cb,    10)
        self.create_subscription(JointState, "/follower/joint_states",  self.joint_cb,  10)
        self.create_subscription(JointState, "/follower/joint_targets", self.action_cb, 10)
        self.create_subscription(PoseStamped,"/ee_pose",                self.ee_cb,     10)

        self.create_timer(1.0 / 60.0, self.watchdog_cb)

        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("Logger Node started (per-frame image + timestamp mode).")
        self.get_logger().info(f"Saving to: {self.episode_path}")
        self.get_logger().info("s=success  f=fail  start=next episode  q=quit")

    def now(self):
        t = self.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9

    def stamp_time(self, msg):
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            return self.now()
        return stamp.sec + stamp.nanosec * 1e-9

    def watchdog_cb(self):
        if not self.recording or self.episode_start_time is None:
            return
        elapsed = self.now() - self.episode_start_time
        if elapsed >= EPISODE_TIMEOUT_SEC:
            self.get_logger().warn(
                f"Episode {self.episode_name} timed out after {EPISODE_TIMEOUT_SEC}s. "
                f"Auto-finishing as fail."
            )
            self.finish_episode(success=False, fail_reason="timeout")

    def front_cb(self, msg):
        if not self.recording:
            return
        if self.episode_start_time is None:
            self.episode_start_time = self.now()
        t = self.stamp_time(msg)
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"{self.front_frame_count:06d}_{t:.6f}.png"
            save_path = os.path.join(self.front_img_path, filename)
            if not cv2.imwrite(save_path, image):
                self.get_logger().error(f"Failed to write: {save_path}")
        except Exception as e:
            self.get_logger().error(f"front image write error: {e}")
        self.front_frame_count += 1

    def top_cb(self, msg):
        if not self.recording:
            return
        if self.episode_start_time is None:
            self.episode_start_time = self.now()
        t = self.stamp_time(msg)
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            filename = f"{self.top_frame_count:06d}_{t:.6f}.png"
            save_path = os.path.join(self.top_img_path, filename)
            if not cv2.imwrite(save_path, image):
                self.get_logger().error(f"Failed to write: {save_path}")
        except Exception as e:
            self.get_logger().error(f"top image write error: {e}")
        self.top_frame_count += 1

    def joint_cb(self, msg):
        if not self.recording:
            return
        t = self.stamp_time(msg)
        positions = list(msg.position)
        while len(positions) < 6:
            positions.append(0.0)
        if self.latest_ee is not None:
            p = self.latest_ee.pose
            ee = [p.position.x, p.position.y, p.position.z,
                  p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        else:
            ee = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        with open(self.states_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{t:.6f}", *positions[:6], *ee])
        self.state_row_count += 1

    def action_cb(self, msg):
        if not self.recording:
            return
        t = self.stamp_time(msg)
        positions = list(msg.position)
        while len(positions) < 6:
            positions.append(0.0)
        with open(self.actions_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{t:.6f}", *positions[:6]])
        self.action_row_count += 1

    def ee_cb(self, msg):
        self.latest_ee = msg

    def get_next_episode_id(self):
        if not os.path.exists(self.save_dir):
            return 1
        ids = []
        for name in os.listdir(self.save_dir):
            if name.startswith("episode_"):
                try:
                    ids.append(int(name.replace("episode_", "")))
                except ValueError:
                    pass
        return max(ids) + 1 if ids else 1

    def setup_episode_directory(self):
        self.episode_name   = f"episode_{self.episode_id:06d}"
        self.episode_path   = os.path.join(self.save_dir, self.episode_name)
        self.front_img_path = os.path.join(self.episode_path, "images", "front")
        self.top_img_path   = os.path.join(self.episode_path, "images", "top")
        os.makedirs(self.front_img_path, exist_ok=True)
        os.makedirs(self.top_img_path, exist_ok=True)
        self.states_csv_path  = os.path.join(self.episode_path, "states.csv")
        self.actions_csv_path = os.path.join(self.episode_path, "actions.csv")
        self.metadata_path    = os.path.join(self.episode_path, "metadata.json")

    def init_files(self):
        with open(self.states_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp",
                "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
                "ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw",
            ])
        with open(self.actions_csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp",
                "target_joint_0", "target_joint_1", "target_joint_2",
                "target_joint_3", "target_joint_4", "target_joint_5",
            ])
        self.front_frame_count = 0
        self.top_frame_count = 0
        self.state_row_count = 0
        self.action_row_count = 0
        self.episode_start_time = None

    def keyboard_listener(self):
        while self.running:
            try:
                cmd = input().strip()
            except EOFError:
                break
            if cmd == "start":
                if self.episode_finished:
                    self.start_next_episode()
                else:
                    self.get_logger().info("Still recording. Press s or f first.")
            elif cmd == "s":
                if self.recording:
                    self.finish_episode(success=True)
            elif cmd == "f":
                if self.recording:
                    self.finish_episode(success=False, fail_reason="manual_fail")
            elif cmd == "q":
                self.running = False
                if self.recording:
                    self.finish_episode(success=None, fail_reason="quit")
                rclpy.shutdown()
                break

    def finish_episode(self, success=None, fail_reason=None):
        self.recording = False
        self.save_metadata(success=success, fail_reason=fail_reason)
        self.episode_finished = True
        self.get_logger().info(
            f"Episode {self.episode_name} finished "
            f"(front={self.front_frame_count}, top={self.top_frame_count}, "
            f"states={self.state_row_count}, actions={self.action_row_count}). "
            f"Press start or q."
        )

    def start_next_episode(self):
        self.episode_id += 1
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.setup_episode_directory()
        self.init_files()
        self.recording = True
        self.episode_finished = False
        self.get_logger().info(f"Started {self.episode_name}")

    def save_metadata(self, success=None, fail_reason=None):
        metadata = {
            "episode_id": self.episode_id,
            "episode_name": self.episode_name,
            "task_name": self.task_name,
            "start_time": self.start_time,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": success,
            "fail_reason": fail_reason,
            "front_frame_count": self.front_frame_count,
            "top_frame_count": self.top_frame_count,
            "state_row_count": self.state_row_count,
            "action_row_count": self.action_row_count,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def destroy_node(self):
        self.running = False
        self.save_metadata(success=None, fail_reason=None)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()