export WANDB_MODE=online

OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Desktop/training/checkpoints/act/run2}"
RESUME="${RESUME:-true}"
CONFIG_PATH="${CONFIG_PATH:-$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json}"

if [ -d "$OUTPUT_DIR" ] && [ "$RESUME" != "true" ]; then
  printf 'Output directory already exists: %s\n' "$OUTPUT_DIR"
  printf 'Resume is disabled. Remove it and continue? [y/N] '
  read -r confirm

  case "$confirm" in
    y|Y)
      rm -rf "$OUTPUT_DIR"
      ;;
    *)
      printf 'Aborting.\n'
      exit 1
      ;;
  esac
fi

UV_CACHE_DIR=/tmp/uv-cache uv run accelerate launch \
  --num_machines="1" \
  --num_processes="${NUM_PROCESSES:-1}" \
  --mixed_precision="${MIXED_PRECISION:-bf16}" \
  --dynamo_backend="no" \
  "../.venv/bin/lerobot-train" \
  ${CONFIG_PATH:+--config_path=$CONFIG_PATH} \
  --dataset.repo_id="ETHRC/towelspring26_2" \
  --dataset.revision="main" \
  --policy.type="act" \
  --policy.repo_id="ETHRC/act-towelspring26_2" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=20 \
  --steps=40000 \
  --optimizer.lr=1.5e-5 \
  --num_workers="12" \
  --dataset.video_backend="torchcodec" \
  --save_freq="900" \
  --log_freq="20" \
  --policy.push_to_hub="true" \
  --job_name="act_training_1" \
  --wandb.project="act" \
  --wandb.enable="true" \
  --policy.device="cuda" \
  --resume="$RESUME"
