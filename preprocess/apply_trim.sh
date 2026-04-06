#!/usr/bin/env bash

set -euo pipefail

REPO="$(UV_CACHE_DIR=.uv-cache uv run python project_config.py dataset_repo_id)"
NEW_REPO="$(UV_CACHE_DIR=.uv-cache uv run python project_config.py new_repo_id)"

UV_CACHE_DIR=.uv-cache uv run python preprocess/apply_trim.py \
  --repo-id "$REPO" \
  --new-repo-id "$NEW_REPO" \
  --resume \
  --vcodec auto \
  --streaming-encoding \
  --push-to-hub \
  --branch main
  # --new-root /tmp/towelspring26-cleaned-trimmed \
