import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


DEFAULT_REPO_ID = "ETHRC/towelspring26-cleaned"
DEFAULT_TRIM_CSV = Path("output/trim_timestamps.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply per-episode trim offsets from a CSV by rewriting a new LeRobot dataset. "
            "The new episodes start at each row's local_new_start frame."
        )
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Source dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Optional source dataset root.")
    parser.add_argument(
        "--new-repo-id",
        default=f"{DEFAULT_REPO_ID}-trimmed",
        help="Destination dataset repo id.",
    )
    parser.add_argument("--new-root", type=Path, default=None, help="Optional destination dataset root.")
    parser.add_argument(
        "--trim-csv",
        type=Path,
        default=DEFAULT_TRIM_CSV,
        help="CSV produced by scripts/trim_before_j2_lift.py.",
    )
    parser.add_argument(
        "--trim-column",
        default="local_new_start",
        help="CSV column containing the per-episode trim start frame.",
    )
    parser.add_argument(
        "--episode-column",
        default="episode_index",
        help="CSV column containing the source episode index.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the rewritten dataset to the Hugging Face Hub after writing it locally.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional Hub branch name to push to, for example 'trimmed'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination root first if it already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing destination dataset and skip already written episodes.",
    )
    parser.add_argument(
        "--vcodec",
        default="auto",
        help=(
            "Video codec for the rewritten dataset. "
            "Use 'auto' to prefer hardware encoders like h264_videotoolbox on macOS."
        ),
    )
    parser.add_argument(
        "--streaming-encoding",
        action="store_true",
        help="Encode video frames while writing instead of first spilling per-frame PNGs.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=None,
        help="Optional encoder thread count hint passed to the LeRobot video encoder.",
    )
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Skip rewriting and only push an existing local destination dataset to the Hub.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use the Hugging Face large-folder upload path, which is more resilient for big datasets.",
    )
    return parser.parse_args()


def load_trim_starts(csv_path: Path, episode_column: str, trim_column: str) -> dict[int, int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Trim CSV not found: {csv_path}")

    trim_starts: dict[int, int] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            try:
                episode_index = int(row[episode_column])
                trim_start = int(row[trim_column])
            except KeyError as exc:
                raise KeyError(f"Missing CSV column: {exc.args[0]}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CSV value on line {row_num}: {row}") from exc

            if episode_index in trim_starts:
                raise ValueError(f"Duplicate episode_index in CSV: {episode_index}")
            if trim_start < 0:
                raise ValueError(f"Negative trim start for episode {episode_index}: {trim_start}")

            trim_starts[episode_index] = trim_start

    if not trim_starts:
        raise ValueError(f"No trim rows found in {csv_path}")

    return trim_starts


def build_frame(item: dict, feature_names: list[str]) -> dict:
    frame = {}
    for name in feature_names:
        value = item[name]
        if hasattr(value, "numpy"):
            value = value.numpy()

        if isinstance(value, np.ndarray) and value.ndim == 3:
            if value.shape[0] in (1, 3) and value.shape[-1] not in (1, 3):
                value = np.transpose(value, (1, 2, 0))

        frame[name] = value

    frame["task"] = item["task"]
    return frame


def main() -> None:
    args = parse_args()

    if args.overwrite and args.resume:
        raise ValueError("Use only one of --overwrite or --resume")

    if args.push_only and not args.push_to_hub:
        raise ValueError("--push-only requires --push-to-hub")

    if args.push_only:
        if args.new_root is None or not args.new_root.exists():
            raise FileNotFoundError(
                "--push-only requires an existing --new-root destination dataset on disk"
            )
        dst = LeRobotDataset(repo_id=args.new_repo_id, root=args.new_root)
        print(
            f"pushing existing local dataset {args.new_repo_id}"
            + (f" to branch {args.branch}" if args.branch else ""),
            flush=True,
        )
        dst.push_to_hub(
            branch=args.branch,
            private=args.private,
            upload_large_folder=args.upload_large_folder,
        )
        print(f"done: pushed {dst.root}", flush=True)
        return

    trim_starts = load_trim_starts(args.trim_csv, args.episode_column, args.trim_column)

    src = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    if args.new_root is not None and args.new_root.exists():
        if not args.overwrite:
            if not args.resume:
                raise FileExistsError(
                    f"Destination already exists: {args.new_root}. "
                    "Pass --overwrite to delete it before rewriting or --resume to continue."
                )
        else:
            shutil.rmtree(args.new_root)

    expected_episodes = set(range(src.num_episodes))
    provided_episodes = set(trim_starts)
    missing = sorted(expected_episodes - provided_episodes)
    extra = sorted(provided_episodes - expected_episodes)
    if missing:
        raise ValueError(f"Trim CSV is missing episodes: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        raise ValueError(f"Trim CSV has unknown episodes: {extra[:10]}{'...' if len(extra) > 10 else ''}")

    resuming = bool(args.resume and args.new_root is not None and args.new_root.exists())
    if resuming:
        dst = LeRobotDataset(
            repo_id=args.new_repo_id,
            root=args.new_root,
            video_backend=src.video_backend,
        )
        if dst.num_episodes > src.num_episodes:
            raise ValueError(
                f"Destination has more episodes than source: {dst.num_episodes} > {src.num_episodes}"
            )
        if dst.features != src.features:
            raise ValueError("Destination dataset features do not match source dataset")
        if dst.fps != src.fps:
            raise ValueError("Destination dataset fps does not match source dataset")
    else:
        dst = LeRobotDataset.create(
            repo_id=args.new_repo_id,
            root=args.new_root,
            fps=src.fps,
            features=src.features,
            robot_type=src.meta.robot_type,
            use_videos=len(src.meta.video_keys) > 0,
            video_backend=src.video_backend,
            vcodec=args.vcodec,
            streaming_encoding=args.streaming_encoding,
            encoder_threads=args.encoder_threads,
        )
        dst.meta.update_chunk_settings(
            chunks_size=src.meta.chunks_size,
            data_files_size_in_mb=src.meta.data_files_size_in_mb,
            video_files_size_in_mb=src.meta.video_files_size_in_mb,
        )

    frame_feature_names = [name for name in src.features if name not in DEFAULT_FEATURES]
    start_episode = dst.num_episodes if resuming else 0
    if start_episode:
        print(f"resuming from episode {start_episode}", flush=True)

    try:
        for episode_index in range(start_episode, src.num_episodes):
            ep = src.meta.episodes[episode_index]
            ep_start = int(ep["dataset_from_index"])
            ep_end = int(ep["dataset_to_index"])
            ep_length = ep_end - ep_start

            trim_start = trim_starts[episode_index]
            if trim_start >= ep_length:
                raise ValueError(
                    f"Trim start {trim_start} would empty episode {episode_index} (length={ep_length})"
                )

            abs_start = ep_start + trim_start
            new_length = ep_end - abs_start

            print(
                f"episode {episode_index}: trimming first {trim_start} frames, "
                f"keeping {new_length}/{ep_length}",
                flush=True,
            )

            for frame_index in range(abs_start, ep_end):
                item = src[frame_index]
                frame = build_frame(item, frame_feature_names)
                dst.add_frame(frame)

            dst.save_episode()

        dst.finalize()
    except Exception:
        dst.finalize()
        raise

    if args.push_to_hub:
        print(
            f"pushing dataset {args.new_repo_id}"
            + (f" to branch {args.branch}" if args.branch else ""),
            flush=True,
        )
        dst.push_to_hub(
            branch=args.branch,
            private=args.private,
            upload_large_folder=args.upload_large_folder,
        )

    print(f"done: wrote {dst.num_episodes} episodes to {dst.root}", flush=True)


if __name__ == "__main__":
    main()
