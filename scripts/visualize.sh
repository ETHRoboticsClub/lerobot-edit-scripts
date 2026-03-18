#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-./}"
REPO_ID="${REPO_ID:-ETHRC/towelspring26-cleaned}"
DEFAULT_EPISODE="${DEFAULT_EPISODE:-}"
REQUESTED_EPISODE="${1:-${DEFAULT_EPISODE}}"
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
HF_HOME_DIR="${HF_HOME_DIR:-./.cache/huggingface}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing project interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${HF_HOME_DIR}"
export HF_HOME="${HF_HOME:-${HF_HOME_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

AVAILABLE_EPISODES="$("${PYTHON_BIN}" - "${ROOT_DIR}" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
data_paths = sorted((root / "data").glob("*/*.parquet"))
paths = data_paths

if not paths:
    sys.exit("No parquet files found under data")

episode_values = set()
for path in paths:
    table = pq.read_table(path, columns=["episode_index"])
    episode_values.update(table.column("episode_index").to_pylist())

print(" ".join(str(ep) for ep in sorted(episode_values)))
PY
)"

if [[ -z "${AVAILABLE_EPISODES}" ]]; then
  echo "No episodes found in ${ROOT_DIR}" >&2
  exit 1
fi

if [[ -z "${REQUESTED_EPISODE}" ]]; then
  REQUESTED_EPISODE="$(awk '{print $NF}' <<<"${AVAILABLE_EPISODES}")"
fi

if ! grep -Eq "(^| )${REQUESTED_EPISODE}( |$)" <<<"${AVAILABLE_EPISODES}"; then
  echo "Episode ${REQUESTED_EPISODE} is not available under ${ROOT_DIR}." >&2
  echo "Available episodes: ${AVAILABLE_EPISODES}" >&2
  exit 1
fi

uv run lerobot-dataset-viz \
  --repo-id "${REPO_ID}" \
  --episode-index "${REQUESTED_EPISODE}" \
  --display-compressed-images \
  --root "${ROOT_DIR}" \
  --mode local
