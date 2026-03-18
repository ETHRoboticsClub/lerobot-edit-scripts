UV_CACHE_DIR=.uv-cache uv run python scripts/apply_trim.py \
  --repo-id ETHRC/towelspring26-cleaned \
  --new-repo-id ETHRC/towelspring26-cleaned \
  --new-root /tmp/towelspring26-cleaned-trimmed \
  --resume \
  --vcodec auto \
  --streaming-encoding \
  --push-to-hub \
  --branch trimmed
