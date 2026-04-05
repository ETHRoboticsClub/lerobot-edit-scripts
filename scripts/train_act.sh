export WANDB_MODE=online

uv run --active --no-sync lerobot-train \
  --dataset.repo_id="ETHRC/towelspring26_2" \
  --dataset.revision="main" \
  --policy.type="act" \
  --policy.repo_id="ETHRC/act-towelspring26_2" \
  --output_dir="$HOME/Desktop/training/checkpoints/act/run2" \
  --batch_size="29" \
  --num_workers="0" \
  --dataset.video_backend="torchcodec" \
  --save_freq="900" \
  --log_freq="20" \
  --policy.push_to_hub="true" \
  --job_name="act_training_1" \
  --wandb.project="act" \
  --wandb.enable="true" \
  --policy.device="cuda"
