#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./}"
REPO_ID="${REPO_ID:-$(UV_CACHE_DIR=.uv-cache uv run python project_config.py dataset_repo_id)}"
DEFAULT_EPISODE="${DEFAULT_EPISODE:-}"
REQUESTED_EPISODE="${1:-${DEFAULT_EPISODE}}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

AVAILABLE_EPISODES="$("${PYTHON_BIN}" - "${ROOT_DIR}" "${REPO_ID}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq
from lerobot.utils.constants import HF_LEROBOT_HOME

root = Path(sys.argv[1])
repo_id = sys.argv[2]
default_hf_lerobot_home = Path.home() / ".cache" / "huggingface" / "lerobot"

candidate_roots: list[Path] = []
if root != Path("./"):
    candidate_roots.append(root)
elif root.exists():
    candidate_roots.append(root)

candidate_roots.append(HF_LEROBOT_HOME / repo_id)
candidate_roots.append(default_hf_lerobot_home / repo_id)

unique_candidate_roots: list[Path] = []
seen = set()
for candidate_root in candidate_roots:
    normalized = str(candidate_root.expanduser().resolve(strict=False))
    if normalized in seen:
        continue
    seen.add(normalized)
    unique_candidate_roots.append(candidate_root)

paths: list[Path] = []
resolved_root: Path | None = None
for candidate_root in unique_candidate_roots:
    data_paths = sorted((candidate_root / "data").glob("*/*.parquet"))
    if data_paths:
        paths = data_paths
        resolved_root = candidate_root
        break

if not paths:
    searched = ", ".join(str(path) for path in unique_candidate_roots)
    sys.exit(f"No parquet files found under any of: {searched}")

episode_values = set()
for path in paths:
    table = pq.read_table(path, columns=["episode_index"])
    episode_values.update(table.column("episode_index").to_pylist())

print(f"{resolved_root}\t{' '.join(str(ep) for ep in sorted(episode_values))}")
PY
)"

if [[ -z "${AVAILABLE_EPISODES}" ]]; then
  echo "No episodes found for ${REPO_ID}" >&2
  exit 1
fi

RESOLVED_ROOT_DIR="${AVAILABLE_EPISODES%%$'\t'*}"
AVAILABLE_EPISODES="${AVAILABLE_EPISODES#*$'\t'}"

if [[ -z "${REQUESTED_EPISODE}" ]]; then
  REQUESTED_EPISODE="all"
fi

if [[ "${REQUESTED_EPISODE}" != "all" ]] && ! grep -Eq "(^| )${REQUESTED_EPISODE}( |$)" <<<"${AVAILABLE_EPISODES}"; then
  echo "Episode ${REQUESTED_EPISODE} is not available under ${RESOLVED_ROOT_DIR}." >&2
  echo "Available episodes: ${AVAILABLE_EPISODES}" >&2
  exit 1
fi

if [[ "${REQUESTED_EPISODE}" == "all" ]]; then
  exec "${PYTHON_BIN}" - "${REPO_ID}" "${RESOLVED_ROOT_DIR}" <<'PY'
import logging
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.utils import init_logging

repo_id = sys.argv[1]
root = Path(sys.argv[2])

init_logging()
logging.info("Loading dataset")
dataset = LeRobotDataset(repo_id, root=root)
visualize_dataset(
    dataset,
    episode_index=0,
    display_compressed_images=True,
    mode="local",
)
PY
fi

uv run lerobot-dataset-viz \
  --repo-id "${REPO_ID}" \
  --episode-index "${REQUESTED_EPISODE}" \
  --display-compressed-images \
  --root "${RESOLVED_ROOT_DIR}" \
  --mode local
