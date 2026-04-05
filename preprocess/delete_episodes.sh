uv run lerobot-edit-dataset \
    --repo_id ETHRC/towelspring26_2 \
    --operation.type delete_episodes \
    --operation.episode_indices "[2, 10, 12, 29, 51]" \
    --push_to_hub true
    # --new_repo_id ETHRC/towelspring26-cleaned \