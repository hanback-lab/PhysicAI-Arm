import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RAW_ROOT = Path("./<raw_dir>")
OUT_REPO_ID = "<id>/<dataset_name>"

FPS = 20
IMAGE_SIZE = (224, 168)
TOP_SIZE = (224, 168)

MAX_TIME_DIFF = 0.03
TASK_DEFAULT = "<task_name>"

TOP_CALIB_PATH = Path("./top_camera_calib.yaml")
FRONT_CALIB_PATH = Path("./front_camera_calib.yaml")
PREVIEW_PATH = Path("./top_preview.png")

TOP_MASK = (slice(0, 95), slice(0, 70))


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


def load_undistorter(yaml_path):
    if yaml_path is None:
        return None
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Calibration file not found: {yaml_path}")
    undistorter = Undistorter(yaml_path)
    print(
        f"Calibration loaded: {yaml_path} "
        f"(model={undistorter.model}, size={undistorter.calib_size})"
    )
    return undistorter


def image_time_from_path(path):
    try:
        timestamp = float(path.stem.split("_")[-1])
        return timestamp if np.isfinite(timestamp) and timestamp > 0 else None
    except (TypeError, ValueError):
        return None


def build_timed_images(image_paths):
    timed = []

    for path in image_paths:
        timestamp = image_time_from_path(path)
        if timestamp is not None:
            timed.append((timestamp, path))

    timed.sort(key=lambda item: item[0])
    return timed


def read_csv_dict(path):
    with path.open("r", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_row_time(row, fallback_index=None):
    for key in ["timestamp", "time", "t", "stamp"]:
        if key in row:
            return safe_float(row, key)

    if "sec" in row and "nanosec" in row:
        return safe_float(row, "sec") + safe_float(row, "nanosec") * 1e-9

    if "secs" in row and "nsecs" in row:
        return safe_float(row, "secs") + safe_float(row, "nsecs") * 1e-9

    if fallback_index is not None:
        return fallback_index / FPS

    return None


def row_to_state(row):
    keys = [f"joint_{index}" for index in range(6)]
    return np.array(
        [safe_float(row, key) for key in keys],
        dtype=np.float32,
    )


def row_to_action_target(row):
    keys = [f"target_joint_{index}" for index in range(6)]
    return np.array(
        [safe_float(row, key) for key in keys],
        dtype=np.float32,
    )


def load_image(path, size, undistorter=None):
    image = cv2.imread(str(path))
    if image is None:
        return None

    if undistorter is not None:
        image = undistorter(image)

    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_top_image(path, undistorter=None):
    image = load_image(path, TOP_SIZE, undistorter)
    if image is None:
        return None

    image[TOP_MASK[0], TOP_MASK[1], :] = 128
    return image


def load_front_image(path, undistorter=None):
    return load_image(path, IMAGE_SIZE, undistorter)


def save_top_preview(path, undistorter):
    raw = cv2.imread(str(path))
    if raw is None:
        return

    before = cv2.cvtColor(load_top_image(path, None), cv2.COLOR_RGB2BGR)
    after = cv2.cvtColor(load_top_image(path, undistorter), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(PREVIEW_PATH), np.hstack([before, after]))
    print(f"Top preview (before | after): {PREVIEW_PATH.resolve()}")


def get_metadata(episode_path):
    metadata_path = episode_path / "metadata.json"
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r") as file:
            metadata = json.load(file)
        return metadata if isinstance(metadata, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_task_name(metadata):
    task_name = metadata.get("task_name", TASK_DEFAULT)

    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or task_name.strip() == "classify"
    ):
        return TASK_DEFAULT

    return task_name.strip()


def get_output_root():
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / OUT_REPO_ID
    )


def build_timed_rows(rows):
    timed = []

    for index, row in enumerate(rows):
        timestamp = get_row_time(row, fallback_index=index)
        if timestamp is not None and np.isfinite(timestamp):
            timed.append((timestamp, row))

    timed.sort(key=lambda item: item[0])
    return timed


def nearest_by_time(timed_data, target_time, start_index=0):
    if not timed_data:
        return None, start_index, None

    index = start_index

    while index + 1 < len(timed_data):
        current_diff = abs(timed_data[index][0] - target_time)
        next_diff = abs(timed_data[index + 1][0] - target_time)

        if next_diff <= current_diff:
            index += 1
        else:
            break

    matched_time, matched_value = timed_data[index]
    return matched_value, index, abs(matched_time - target_time)


def main():
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_ROOT.resolve()}"
        )

    output_root = get_output_root()

    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Choose a different OUT_REPO_ID or move the existing output "
            "before running again."
        )

    episode_paths = sorted(
        path
        for path in RAW_ROOT.glob("episode_*")
        if path.is_dir()
    )

    if not episode_paths:
        raise FileNotFoundError(
            f"No episode folders found in: {RAW_ROOT.resolve()}"
        )

    top_undistorter = load_undistorter(TOP_CALIB_PATH)
    front_undistorter = load_undistorter(FRONT_CALIB_PATH)

    if top_undistorter is not None:
        for episode_path in episode_paths:
            first = build_timed_images((episode_path / "images" / "top").glob("*.png"))
            if first:
                save_top_preview(first[0][1], top_undistorter)
                break

    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (TOP_SIZE[1], TOP_SIZE[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": (IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["action"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=OUT_REPO_ID,
        root=output_root,
        fps=FPS,
        features=features,
    )

    print(f"Found {len(episode_paths)} raw episode folders")
    print("Camera mapping: images/top -> top, images/front -> front")
    print(f"Undistort: top={top_undistorter is not None}, front={front_undistorter is not None}")
    print(f"Top mask: rows {TOP_MASK[0].start}:{TOP_MASK[0].stop}, cols {TOP_MASK[1].start}:{TOP_MASK[1].stop} = 128")
    print(f"Output: {output_root}")

    valid_episode_count = 0
    total_frame_count = 0
    skipped_frame_count = 0

    for episode_path in episode_paths:
        metadata = get_metadata(episode_path)

        if metadata.get("success", True) is False:
            print(f"Skip failed episode: {episode_path.name}")
            continue

        states_csv = episode_path / "states.csv"
        actions_csv = episode_path / "actions.csv"

        top_dir = episode_path / "images" / "top"
        front_dir = episode_path / "images" / "front"

        required_paths = [
            states_csv,
            actions_csv,
            top_dir,
            front_dir,
        ]

        missing_paths = [
            path for path in required_paths if not path.exists()
        ]

        if missing_paths:
            print(
                f"Skip {episode_path.name}: missing "
                + ", ".join(str(path) for path in missing_paths)
            )
            continue

        timed_states = build_timed_rows(read_csv_dict(states_csv))
        timed_actions = build_timed_rows(read_csv_dict(actions_csv))
        timed_top_images = build_timed_images(top_dir.glob("*.png"))
        timed_front_images = build_timed_images(front_dir.glob("*.png"))

        if not all([
            timed_states,
            timed_actions,
            timed_top_images,
            timed_front_images,
        ]):
            print(f"Skip empty or invalid episode: {episode_path.name}")
            continue

        task = get_task_name(metadata)

        print(
            f"Processing {episode_path.name}: "
            f"states={len(timed_states)}, "
            f"actions={len(timed_actions)}, "
            f"top={len(timed_top_images)}, "
            f"front={len(timed_front_images)}"
        )

        action_index = 0
        top_index = 0
        front_index = 0
        added_count = 0
        episode_skipped = 0

        for frame_time, state_row in timed_states:
            top_path, top_index, top_diff = nearest_by_time(
                timed_top_images, frame_time, top_index
            )

            front_path, front_index, front_diff = nearest_by_time(
                timed_front_images, frame_time, front_index
            )

            action_row, action_index, action_diff = nearest_by_time(
                timed_actions, frame_time, action_index
            )

            if any(
                diff is None or diff > MAX_TIME_DIFF
                for diff in [top_diff, front_diff, action_diff]
            ):
                episode_skipped += 1
                continue

            top_image = load_top_image(top_path, top_undistorter)
            front_image = load_front_image(front_path, front_undistorter)

            if top_image is None or front_image is None:
                episode_skipped += 1
                continue

            dataset.add_frame({
                "observation.images.top": top_image,
                "observation.images.front": front_image,
                "observation.state": torch.tensor(
                    row_to_state(state_row),
                    dtype=torch.float32,
                ),
                "action": torch.tensor(
                    row_to_action_target(action_row),
                    dtype=torch.float32,
                ),
                "task": task,
            })

            added_count += 1
            total_frame_count += 1

        skipped_frame_count += episode_skipped

        if added_count == 0:
            print(f"Skip save, no valid frames: {episode_path.name}")
            continue

        dataset.save_episode()
        valid_episode_count += 1

        print(
            f"  Saved: frames={added_count}, "
            f"skipped={episode_skipped}"
        )

    print("\nConversion done")
    print(f"Valid episodes: {valid_episode_count}")
    print(f"Total frames:   {total_frame_count}")
    print(f"Skipped frames: {skipped_frame_count}")
    print(f"Local output:   {output_root}")

    if valid_episode_count == 0:
        print("No valid episodes. Nothing to finalize or upload.")
        return

    print("\nFinalizing dataset...")
    dataset.finalize()

    tags = ["pick_token", "physicai_arm", "lerobot", "leader_masked"]
    if top_undistorter is not None or front_undistorter is not None:
        tags.append("undistorted")

    print("\nUploading dataset...")
    try:
        dataset.push_to_hub(
            tags=tags,
            private=False,
        )

        print(
            "Upload complete: "
            f"https://huggingface.co/datasets/{OUT_REPO_ID}"
        )

    except Exception as exception:
        print(f"Upload failed: {exception}")
        print(f"Finalized local dataset is preserved: {output_root}")


if __name__ == "__main__":
    main()