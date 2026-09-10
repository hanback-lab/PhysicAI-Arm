import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


RAW_ROOT = Path("<raw_data_path>")

USER_NAME = "<user_name>"
DATASET_NAME = "<dataset_name>"
OUT_REPO_ID = f"{USER_NAME}/{DATASET_NAME}"

FPS = 20
IMAGE_SIZE = (224, 168)
TOP_SIZE = (224, 168)

MAX_TIME_DIFF = 0.03

TASK_DEFAULT = "<task_name>"


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
        return (
            safe_float(row, "sec")
            + safe_float(row, "nanosec") * 1e-9
        )

    if "secs" in row and "nsecs" in row:
        return (
            safe_float(row, "secs")
            + safe_float(row, "nsecs") * 1e-9
        )

    if fallback_index is not None:
        return fallback_index / FPS

    return None


def row_to_state(row):
    keys = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    ]

    return np.array(
        [safe_float(row, k) for k in keys],
        dtype=np.float32,
    )


def row_to_action_target(row):
    keys = [
        "target_joint_0",
        "target_joint_1",
        "target_joint_2",
        "target_joint_3",
        "target_joint_4",
        "target_joint_5",
    ]

    return np.array(
        [safe_float(row, k) for k in keys],
        dtype=np.float32,
    )


def load_top_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

    img = cv2.resize(img, TOP_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_front_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

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

        task_name = metadata.get(
            "task_name",
            TASK_DEFAULT,
        )

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

        success = metadata.get(
            "success",
            True,
        )

        return success is not False

    except Exception:
        return True


def remove_old_output():
    cache_path = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / OUT_REPO_ID
    )

    if cache_path.exists():
        shutil.rmtree(cache_path)

        print(
            f"Removed old output: "
            f"{cache_path}"
        )


def build_timed_rows(rows):
    timed = []

    for i, row in enumerate(rows):
        t = get_row_time(
            row,
            fallback_index=i,
        )

        if t is None:
            continue

        timed.append((t, row))

    timed.sort(key=lambda x: x[0])

    return timed


def nearest_by_time(
    timed_data,
    target_time,
    start_index=0,
):
    if not timed_data:
        return None, start_index, None

    idx = start_index

    while idx + 1 < len(timed_data):
        now_diff = abs(
            timed_data[idx][0]
            - target_time
        )

        next_diff = abs(
            timed_data[idx + 1][0]
            - target_time
        )

        if next_diff <= now_diff:
            idx += 1
        else:
            break

    matched_time, matched_value = (
        timed_data[idx]
    )

    diff = abs(
        matched_time
        - target_time
    )

    return matched_value, idx, diff


def main():
    remove_old_output()

    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (
                TOP_SIZE[1],
                TOP_SIZE[0],
                3,
            ),
            "names": [
                "height",
                "width",
                "channel",
            ],
        },

        "observation.images.front": {
            "dtype": "image",
            "shape": (
                IMAGE_SIZE[1],
                IMAGE_SIZE[0],
                3,
            ),
            "names": [
                "height",
                "width",
                "channel",
            ],
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

    episode_paths = sorted(
        RAW_ROOT.glob("episode_*")
    )

    print(
        f"Found {len(episode_paths)} "
        f"raw episodes"
    )

    print(
        f"Top image size: "
        f"{TOP_SIZE[0]}x{TOP_SIZE[1]} "
        f"(W x H), no crop"
    )

    print(
        f"Front image size: "
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} "
        f"(W x H)"
    )

    print(
        f"MAX_TIME_DIFF: "
        f"{MAX_TIME_DIFF}s"
    )

    valid_episode_count = 0
    total_frame_count = 0
    skipped_frame_count = 0

    for episode_path in episode_paths:

        if not episode_path.is_dir():
            continue

        if not get_success(episode_path):
            print(
                f"Skipping failed episode: "
                f"{episode_path}"
            )
            continue

        states_csv = (
            episode_path
            / "states.csv"
        )

        actions_csv = (
            episode_path
            / "actions.csv"
        )

        top_dir = (
            episode_path
            / "images"
            / "top"
        )

        front_dir = (
            episode_path
            / "images"
            / "front"
        )

        if not states_csv.exists():
            print(
                f"Skipping episode: "
                f"missing states.csv - "
                f"{episode_path}"
            )
            continue

        if not actions_csv.exists():
            print(
                f"Skipping episode: "
                f"missing actions.csv - "
                f"{episode_path}"
            )
            continue

        if not top_dir.exists():
            print(
                f"Skipping episode: "
                f"missing top image directory - "
                f"{episode_path}"
            )
            continue

        if not front_dir.exists():
            print(
                f"Skipping episode: "
                f"missing front image directory - "
                f"{episode_path}"
            )
            continue

        states = read_csv_dict(
            states_csv
        )

        actions = read_csv_dict(
            actions_csv
        )

        top_images = sorted(
            top_dir.glob("*.png")
        )

        front_images = sorted(
            front_dir.glob("*.png")
        )

        timed_top_images = (
            build_timed_images(
                top_images
            )
        )

        timed_front_images = (
            build_timed_images(
                front_images
            )
        )

        timed_states = (
            build_timed_rows(
                states
            )
        )

        timed_actions = (
            build_timed_rows(
                actions
            )
        )

        if (
            not timed_states
            or not timed_actions
            or not timed_top_images
            or not timed_front_images
        ):
            print(
                f"Skipping empty episode: "
                f"{episode_path}"
            )
            continue

        task = get_task_name(
            episode_path
        )

        print(
            f"Processing "
            f"{episode_path.name}: "
            f"states={len(timed_states)}, "
            f"actions={len(timed_actions)}, "
            f"top={len(top_images)}, "
            f"front={len(front_images)}"
        )

        action_idx = 0
        top_idx = 0
        front_idx = 0

        added_count = 0
        episode_skipped = 0

        for frame_time, state_row in timed_states:

            (
                top_path,
                top_idx,
                top_diff,
            ) = nearest_by_time(
                timed_top_images,
                frame_time,
                top_idx,
            )

            (
                front_path,
                front_idx,
                front_diff,
            ) = nearest_by_time(
                timed_front_images,
                frame_time,
                front_idx,
            )

            (
                action_row,
                action_idx,
                act_diff,
            ) = nearest_by_time(
                timed_actions,
                frame_time,
                action_idx,
            )

            if any(
                d is None
                or d > MAX_TIME_DIFF
                for d in [
                    top_diff,
                    front_diff,
                    act_diff,
                ]
            ):
                episode_skipped += 1
                continue

            if (
                state_row is None
                or action_row is None
            ):
                episode_skipped += 1
                continue

            current_joint = (
                row_to_state(
                    state_row
                )
            )

            target_joint = (
                row_to_action_target(
                    action_row
                )
            )

            top_img = (
                load_top_image(
                    top_path
                )
            )

            front_img = (
                load_front_image(
                    front_path
                )
            )

            if (
                top_img is None
                or front_img is None
            ):
                episode_skipped += 1
                continue

            dataset.add_frame({
                "observation.images.top":
                    top_img,

                "observation.images.front":
                    front_img,

                "observation.state":
                    torch.tensor(
                        current_joint,
                        dtype=torch.float32,
                    ),

                "action":
                    torch.tensor(
                        target_joint,
                        dtype=torch.float32,
                    ),

                "task":
                    task,
            })

            total_frame_count += 1
            added_count += 1

        skipped_frame_count += (
            episode_skipped
        )

        if added_count == 0:
            print(
                f"Skipping save: "
                f"no valid frames - "
                f"{episode_path}"
            )
            continue

        dataset.save_episode()

        valid_episode_count += 1

        print(
            f"Saved: "
            f"{episode_path.name}, "
            f"frames={added_count}, "
            f"skipped={episode_skipped}"
        )

    print("\nConversion complete!")

    print(
        f"Valid episodes: "
        f"{valid_episode_count}"
    )

    print(
        f"Total frames:   "
        f"{total_frame_count}"
    )

    print(
        f"Skipped frames: "
        f"{skipped_frame_count}"
    )

    print(
        f"Saved repo ID:  "
        f"{OUT_REPO_ID}"
    )

    if valid_episode_count == 0:
        print(
            "\nNo valid episodes. "
            "Nothing to finalize or upload."
        )
        return

    print(
        "\nFinalizing dataset..."
    )

    dataset.finalize()

    print(
        "Dataset finalized."
    )

    print(
        "\nPushing complete dataset "
        "to Hugging Face Hub..."
    )

    try:
        dataset.push_to_hub(
            tags=[
                "<task_tag>",
                "<project_tag>",
                "lerobot",
            ],
            private=False,
        )

        print(
            f"Upload complete: "
            f"https://huggingface.co/"
            f"datasets/{OUT_REPO_ID}"
        )

    except Exception as exc:
        print(
            f"Push to Hub failed: "
            f"{exc}"
        )

        print(
            "Local dataset is already "
            "finalized and saved."
        )


if __name__ == "__main__":
    main()