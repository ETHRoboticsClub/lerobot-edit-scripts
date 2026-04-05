#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path

import av

from lerobot.datasets import aggregate as aggregate_module
from lerobot.datasets import video_utils as video_utils_module
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def safe_concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path | str,
    overwrite: bool = True,
) -> None:
    """Concatenate videos by decoding and re-encoding to avoid broken seek tables."""

    if not input_video_paths:
        raise FileNotFoundError("No input video paths provided.")

    output_path = Path(output_video_path)
    if output_path.exists() and not overwrite:
        logging.warning("Video file already exists: %s. Skipping concatenation.", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(path) for path in input_video_paths]
    logging.info(
        "Re-encoding concatenation into %s from %d input file(s)",
        output_path,
        len(input_paths),
    )

    with av.open(str(input_paths[0])) as first_container:
        first_stream = first_container.streams.video[0]
        width = first_stream.width
        height = first_stream.height
        rate = first_stream.average_rate or Fraction(30, 1)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_path = Path(tmp_named_file.name)

    output_container = av.open(str(tmp_output_path), mode="w", options={"movflags": "faststart"})
    output_stream = output_container.add_stream(
        "libx264",
        rate=rate,
        options={
            "crf": "18",
            "preset": "fast",
            "tune": "zerolatency",
            "bf": "0",
        },
    )
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = "yuv420p"

    try:
        for input_path in input_paths:
            logging.info("Reading video input %s", input_path)
            frames_processed = 0
            with av.open(str(input_path)) as input_container:
                input_stream = input_container.streams.video[0]
                for frame in input_container.decode(input_stream):
                    rgb = frame.to_ndarray(format="rgb24")
                    clean_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    clean_frame = clean_frame.reformat(width=width, height=height, format="yuv420p")
                    for packet in output_stream.encode(clean_frame):
                        output_container.mux(packet)
                    frames_processed += 1
                    if frames_processed % 300 == 0:
                        logging.info("Processed %d frames from %s", frames_processed, input_path)
            logging.info("Finished %s with %d frames", input_path, frames_processed)

        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        output_container.close()

    shutil.move(str(tmp_output_path), str(output_path))
    logging.info("Finished concatenated video %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge LeRobot datasets with safe video concatenation.")
    parser.add_argument("source_repo_id_1")
    parser.add_argument("source_repo_id_2")
    parser.add_argument("new_repo_id")
    parser.add_argument("source_root_1", nargs="?")
    parser.add_argument("source_root_2", nargs="?")
    parser.add_argument("--push-to-hub", default="true")
    parser.add_argument("--new-root", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if bool(args.source_root_1) != bool(args.source_root_2):
        raise SystemExit("Provide both source_root_1 and source_root_2, or neither.")

    push_to_hub = parse_bool(args.push_to_hub)
    repo_ids = [args.source_repo_id_1, args.source_repo_id_2]
    roots = [args.source_root_1, args.source_root_2] if args.source_root_1 else None

    init_logging()

    if roots:
        logging.info("Loading %d datasets to merge from explicit roots", len(repo_ids))
        datasets = [
            LeRobotDataset(repo_id=repo_id, root=root)
            for repo_id, root in zip(repo_ids, roots, strict=True)
        ]
    else:
        logging.info("Loading %d datasets to merge", len(repo_ids))
        datasets = [LeRobotDataset(repo_id) for repo_id in repo_ids]

    output_dir = Path(args.new_root) if args.new_root else HF_LEROBOT_HOME / args.new_repo_id

    aggregate_module.concatenate_video_files = safe_concatenate_video_files
    video_utils_module.concatenate_video_files = safe_concatenate_video_files

    logging.info("Merging datasets into %s", args.new_repo_id)
    logging.info("Using safe video concatenation with re-encoding to preserve seekability")
    merged_dataset = merge_datasets(
        datasets,
        output_repo_id=args.new_repo_id,
        output_dir=output_dir,
    )

    logging.info("Merged dataset saved to %s", output_dir)
    logging.info(
        "Episodes: %s, Frames: %s",
        merged_dataset.meta.total_episodes,
        merged_dataset.meta.total_frames,
    )

    if push_to_hub:
        logging.info("Pushing to hub as %s", args.new_repo_id)
        LeRobotDataset(merged_dataset.repo_id, root=output_dir).push_to_hub()


if __name__ == "__main__":
    main()
