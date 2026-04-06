import csv
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_config import get_dataset_repo_id


REPO = get_dataset_repo_id()
seconds_before = 0.3
threshold = 0.1
output_path = REPO_ROOT / "output" / "trim_timestamps.csv"

ds = LeRobotDataset(REPO, force_cache_sync=True)
fps = ds.fps
frames_before = round(seconds_before * fps)
joint_names = ["left_joint_2.pos", "right_joint_2.pos"]
joint_idxs = [ds.features["observation.state"]["names"].index(name) for name in joint_names]
ds._ensure_hf_dataset_loaded()


def confirm_overwrite(path: Path) -> None:
    if not path.exists():
        return

    reply = input(f"{path} already exists. Delete it and regenerate it? [y/N]: ").strip().lower()
    if reply not in {"y", "yes"}:
        raise SystemExit("Aborted; existing trim CSV was left untouched.")

    path.unlink()
    print(f"deleted {path}", flush=True)


def format_ts(frame_idx: int) -> str:
    seconds = frame_idx / fps
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{whole}.{millis:03d}s"

fieldnames = [
    "episode_number",
    "episode_index",
    "dataset_start",
    "dataset_end",
    "local_start",
    "local_end",
    "local_new_start",
    "trim_timestamp_s",
]

print(f"num_episodes={ds.num_episodes}", flush=True)
print(f"joints={', '.join(joint_names)}", flush=True)

output_path.parent.mkdir(parents=True, exist_ok=True)
confirm_overwrite(output_path)

with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for ep in range(ds.num_episodes):
        ep_start = ds.meta.episodes["dataset_from_index"][ep]
        ep_end = ds.meta.episodes["dataset_to_index"][ep]
        states = np.asarray(ds.hf_dataset[ep_start:ep_end]["observation.state"])

        hit = np.flatnonzero(np.any(states[:, joint_idxs] > threshold, axis=1))

        new_start = ep_start if len(hit) == 0 else max(ep_start, ep_start + int(hit[0]) - frames_before)
        local_start = 0
        local_end = ep_end - ep_start
        local_new_start = new_start - ep_start
        trim_timestamp = format_ts(local_new_start)

        writer.writerow({
            "episode_number": ep,
            "episode_index": ep,
            "dataset_start": ep_start,
            "dataset_end": ep_end,
            "local_start": local_start,
            "local_end": local_end,
            "local_new_start": local_new_start,
            "trim_timestamp_s": trim_timestamp,
        })
        f.flush()
        print(
            f"ep {ep}/{ds.num_episodes - 1}: {ep} {local_start} {local_end} -> "
            f"{local_new_start} {local_end} ({trim_timestamp})",
            flush=True,
        )

print("done", flush=True)
