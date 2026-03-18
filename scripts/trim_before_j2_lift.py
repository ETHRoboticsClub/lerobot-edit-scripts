import csv
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "ETHRC/towelspring26-cleaned"
seconds_before = 0.3
threshold = 0.1
output_path = Path("output/trim_timestamps.csv")

ds = LeRobotDataset(repo_id)
fps = ds.fps
frames_before = round(seconds_before * fps)
joint_names = ["left_joint_2.pos", "right_joint_2.pos"]
joint_idxs = [ds.features["observation.state"]["names"].index(name) for name in joint_names]
ds._ensure_hf_dataset_loaded()


def load_processed_episodes(path: Path) -> set[int]:
    processed = set()
    if not path.exists():
        return processed

    with path.open(newline="") as f:
        try:
            reader = csv.DictReader(f)
        except csv.Error:
            return processed

        for row in reader:
            try:
                processed.add(int(row["episode_index"]))
            except (KeyError, TypeError, ValueError):
                continue

    return processed


def format_ts(frame_idx: int) -> str:
    seconds = frame_idx / fps
    whole = int(seconds)
    millis = round((seconds - whole) * 1000)
    if millis == 1000:
        whole += 1
        millis = 0
    return f"{whole}.{millis:03d}s"

processed_episodes = load_processed_episodes(output_path)
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

with output_path.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if f.tell() == 0:
        writer.writeheader()

    for ep in range(ds.num_episodes):
        if ep in processed_episodes:
            continue

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
