#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.utils import init_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--web-port", type=int, default=9090)
    parser.add_argument("--ws-port", type=int)
    parser.add_argument("--grpc-port", type=int, default=9876)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--display-compressed-images", action="store_true")
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "video_reader", "torchcodec"],
        help="Decoder backend for LeRobotDataset. Defaulting to pyav avoids mixing PyAV with TorchCodec/Homebrew FFmpeg.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.ws_port is not None:
        logging.warning(
            "--ws-port is deprecated and will be removed in future versions. Please use --grpc-port instead."
        )
        logging.warning("Setting grpc_port to ws_port value.")
        args.grpc_port = args.ws_port

    init_logging()
    logging.info("Loading dataset")
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        root=args.root,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
    )

    visualize_dataset(
        dataset,
        episode_index=args.episode_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.mode,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
        save=bool(args.save),
        output_dir=args.output_dir,
        display_compressed_images=args.display_compressed_images,
    )


if __name__ == "__main__":
    main()
