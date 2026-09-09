import csv
import json
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2
import numpy as np
import torch


RAW_ROOT = Path("./<raw_data_dir_name>")
OUT_REPO_ID = "<hf_username>/<dataset_repo_name>"

FPS = 20
IMAGE_SIZE = (128, 128)
TOP_SIZE = (128, 128)

MAX_TIME_DIFF = 0.03

TASK_DEFAULT = "<task_name>"

# Camera Calibration 결과가 저장된 JSON 파일 경로.
# 파일이 있으면 fisheye 왜곡 보정을 적용하고, 없으면 보정 없이 그대로 파싱한다.
CALIB_PATH = Path("./<calibration_json_path>")

_CALIB = None
_UNDISTORT_MAPS = {}


def load_calibration():
    """캘리브레이션 JSON을 읽어 반환. 파일이 없거나 잘못되면 None."""
    if not CALIB_PATH.exists():
        print(f"No calibration file at {CALIB_PATH}; parsing without undistortion.")
        return None
    try:
        with open(CALIB_PATH, "r") as f:
            calib = json.load(f)
        print(f"Loaded calibration: {CALIB_PATH}")
        return calib
    except Exception as exc:
        print(f"Failed to read calibration ({exc}); parsing without undistortion.")
        return None


def _calib_for_camera(cam):
    """카메라별 섹션({"top": {...}, "front": {...}})이 있으면 그것을,
    없으면 공용 K/D를 사용한다. K/D가 없으면 None."""
    if _CALIB is None:
        return None
    section = _CALIB.get(cam, _CALIB)
    if "K" not in section or "D" not in section:
        return None
    return section


def undistort_image(img, cam):
    """캘리브레이션이 있으면 fisheye 왜곡을 보정, 없으면 원본을 그대로 반환."""
    section = _calib_for_camera(cam)
    if section is None:
        return img

    h, w = img.shape[:2]
    key = (cam, w, h)
    maps = _UNDISTORT_MAPS.get(key)
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
        _UNDISTORT_MAPS[key] = maps

    return cv2.remap(
        img, maps[0], maps[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def image_time_from_path(path):
    try:
        return float(path.stem.split("_")[-1])
    except Exception:
        return None


def build_timed_images(image_paths):
    timed = []
    for path in image_paths:
        t = image_time_from_path(path)
        if t is None or t <= 0.0:
            continue
        timed.append((t, path))
    timed.sort(key=lambda x: x[0])
    return timed


def read_csv_dict(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
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
    keys = [
        "joint_0", "joint_1", "joint_2",
        "joint_3", "joint_4", "joint_5",
    ]
    return np.array([safe_float(row, k) for k in keys], dtype=np.float32)


def row_to_action_target(row):
    keys = [
        "target_joint_0", "target_joint_1", "target_joint_2",
        "target_joint_3", "target_joint_4", "target_joint_5",
    ]
    return np.array([safe_float(row, k) for k in keys], dtype=np.float32)


def load_top_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = undistort_image(img, "top")
    img = cv2.resize(img, TOP_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_front_image(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = undistort_image(img, "front")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_task_name(episode_path):
    metadata_path = episode_path / "metadata.json"
    if not metadata_path.exists():
        return TASK_DEFAULT
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        task_name = metadata.get("task_name", TASK_DEFAULT)
        if task_name == "classify":
            return TASK_DEFAULT
        return task_name
    except Exception:
        return TASK_DEFAULT


def get_success(episode_path):
    metadata_path = episode_path / "metadata.json"
    if not metadata_path.exists():
        return True
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        success = metadata.get("success", True)
        return success is not False
    except Exception:
        return True


def remove_old_output():
    cache_path = Path.home() / ".cache/huggingface/lerobot" / OUT_REPO_ID
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"Removed old output: {cache_path}")


def build_timed_rows(rows):
    timed = []
    for i, row in enumerate(rows):
        t = get_row_time(row, fallback_index=i)
        if t is None:
            continue
        timed.append((t, row))
    timed.sort(key=lambda x: x[0])
    return timed


def nearest_by_time(timed_data, target_time, start_index=0):
    if not timed_data:
        return None, start_index, None

    idx = start_index

    while idx + 1 < len(timed_data):
        now_diff = abs(timed_data[idx][0] - target_time)
        next_diff = abs(timed_data[idx + 1][0] - target_time)

        if next_diff <= now_diff:
            idx += 1
        else:
            break

    matched_time, matched_value = timed_data[idx]
    diff = abs(matched_time - target_time)

    return matched_value, idx, diff


def main():
    global _CALIB
    _CALIB = load_calibration()

    remove_old_output()

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
        fps=FPS,
        features=features,
    )

    episode_paths = sorted(RAW_ROOT.glob("episode_*"))

    print(f"Found {len(episode_paths)} raw episodes")
    print(f"top image size: {TOP_SIZE[0]}x{TOP_SIZE[1]} (w x h), no crop")
    print(f"front image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} (w x h)")
    print(f"MAX_TIME_DIFF: {MAX_TIME_DIFF}s")

    valid_episode_count = 0
    total_frame_count = 0
    skipped_frame_count = 0

    for episode_path in episode_paths:
        if not episode_path.is_dir():
            continue

        if not get_success(episode_path):
            print(f"Skip failed episode: {episode_path}")
            continue

        states_csv = episode_path / "states.csv"
        actions_csv = episode_path / "actions.csv"
        top_dir = episode_path / "images" / "top"
        front_dir = episode_path / "images" / "front"

        if not states_csv.exists():
            print(f"Skip: missing states.csv - {episode_path}")
            continue
        if not actions_csv.exists():
            print(f"Skip: missing actions.csv - {episode_path}")
            continue
        if not top_dir.exists():
            print(f"Skip: missing top image dir - {episode_path}")
            continue
        if not front_dir.exists():
            print(f"Skip: missing front image dir - {episode_path}")
            continue

        states = read_csv_dict(states_csv)
        actions = read_csv_dict(actions_csv)
        top_images = sorted(top_dir.glob("*.png"))
        front_images = sorted(front_dir.glob("*.png"))

        timed_top_images = build_timed_images(top_images)
        timed_front_images = build_timed_images(front_images)
        timed_states = build_timed_rows(states)
        timed_actions = build_timed_rows(actions)

        if not timed_states or not timed_actions or not timed_top_images or not timed_front_images:
            print(f"Skip empty episode: {episode_path}")
            continue

        task = get_task_name(episode_path)

        print(
            f"Processing {episode_path.name}: "
            f"states={len(timed_states)}, actions={len(timed_actions)}, "
            f"top={len(top_images)}, front={len(front_images)}"
        )

        state_idx = 0
        action_idx = 0
        top_idx = 0
        front_idx = 0
        added_count = 0
        episode_skipped = 0

        for frame_time, state_row in timed_states:
            top_path,   top_idx,   top_diff   = nearest_by_time(timed_top_images,   frame_time, top_idx)
            front_path, front_idx, front_diff = nearest_by_time(timed_front_images, frame_time, front_idx)
            action_row, action_idx, act_diff  = nearest_by_time(timed_actions,      frame_time, action_idx)

            if any(d is None or d > MAX_TIME_DIFF for d in [top_diff, front_diff, act_diff]):
                episode_skipped += 1
                continue

            if state_row is None or action_row is None:
                episode_skipped += 1
                continue

            current_joint = row_to_state(state_row)
            target_joint  = row_to_action_target(action_row)

            top_img   = load_top_image(top_path)
            front_img = load_front_image(front_path)

            if top_img is None or front_img is None:
                episode_skipped += 1
                continue

            dataset.add_frame({
                "observation.images.top":   top_img,
                "observation.images.front": front_img,
                "observation.state":  torch.tensor(current_joint, dtype=torch.float32),
                "action":             torch.tensor(target_joint,  dtype=torch.float32),
                "task": task,
            })

            total_frame_count += 1
            added_count       += 1

        skipped_frame_count += episode_skipped

        if added_count == 0:
            print(f"Skip save: no valid frames - {episode_path}")
            continue

        dataset.save_episode()
        valid_episode_count += 1

        print(f"  Saved: {episode_path.name}, frames={added_count}, skipped={episode_skipped}")

    print("\nConversion done!")
    print(f"Valid episodes: {valid_episode_count}")
    print(f"Total frames:   {total_frame_count}")
    print(f"Skipped frames: {skipped_frame_count}")
    print(f"Saved repo_id:  {OUT_REPO_ID}")

    print("\nPushing dataset to Hugging Face Hub...")
    try:
        dataset.push_to_hub(
            tags=["<task_name>", "physicai_arm", "lerobot"],
            private=False,
        )
        print(f"Upload complete: https://huggingface.co/datasets/{OUT_REPO_ID}")
    except Exception as exc:
        print(f"Push to hub failed: {exc}")
        print("huggingface-cli login 상태와 write 권한을 확인해주세요.")


if __name__ == "__main__":
    main()