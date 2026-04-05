#!/usr/bin/env bash

set -euo pipefail

REPO_ID="$(UV_CACHE_DIR=.uv-cache uv run python project_config.py dataset_repo_id)"

uv run lerobot-edit-dataset \
    --repo_id "$REPO_ID" \
    --operation.type delete_episodes \
    --operation.episode_indices "[12]" \
    --push_to_hub true
    # --new_repo_id ETHRC/towelspring26-cleaned \
