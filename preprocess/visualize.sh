REPO_ID="$(UV_CACHE_DIR=.uv-cache uv run python project_config.py dataset_repo_id)"

uv run lerobot-dataset-viz \
    --repo-id $REPO_ID \
    --episode-index 85