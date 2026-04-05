UV_CACHE_DIR=.uv-cache uv run python scripts/apply_trim.py \
  --repo-id ETHRC/towelspring26_2 \
  --new-repo-id ETHRC/towelspring26_2 \
  --resume \
  --vcodec auto \
  --streaming-encoding \
  --push-to-hub \
  --branch main
  # --new-root /tmp/towelspring26-cleaned-trimmed \
