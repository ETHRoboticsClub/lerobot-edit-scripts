
SIZE="${1:-135}"

for episode in $(seq 0 10 "$SIZE"); do
    uv run lerobot-dataset-viz --mode local \
        --repo-id ETHRC/towelspring26-cleaned \
        --episode-index "$episode"
done
