export WANDB_MODE=online

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_CONFIG="${DEVICE_CONFIG:-$SCRIPT_DIR/configs/aws_gpul.yaml}"

DATASET_REPO_ID="$(yq -r '.dataset_repo_id' "$DEVICE_CONFIG")"
POLICY_REPO_ID="$(yq -r '.policy_repo_id' "$DEVICE_CONFIG")"
OUTPUT_DIR="${OUTPUT_DIR:-$(yq -r '.output_dir' "$DEVICE_CONFIG")}"
JOB_NAME="$(yq -r '.job_name' "$DEVICE_CONFIG")"
RESUME="${RESUME:-true}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last}"
CONFIG_PATH="${CONFIG_PATH:-$OUTPUT_DIR/checkpoints/$CHECKPOINT_NAME/pretrained_model/train_config.json}"

NUM_MACHINES="$(yq -r '.num_machines' "$DEVICE_CONFIG")"
MULTI_GPU="$(yq -r '.multi_gpu' "$DEVICE_CONFIG")"
NUM_PROCESSES="$(yq -r '.num_processes' "$DEVICE_CONFIG")"
MIXED_PRECISION="$(yq -r '.mixed_precision' "$DEVICE_CONFIG")"
DYNAMO_BACKEND="$(yq -r '.dynamo_backend' "$DEVICE_CONFIG")"
BATCH_SIZE="$(yq -r '.batch_size' "$DEVICE_CONFIG")"
STEPS="$(yq -r '.steps' "$DEVICE_CONFIG")"
OPTIMIZER_LR="$(yq -r '.optimizer_lr' "$DEVICE_CONFIG")"
NUM_WORKERS="$(yq -r '.num_workers' "$DEVICE_CONFIG")"
VIDEO_BACKEND="$(yq -r '.video_backend' "$DEVICE_CONFIG")"
POLICY_DEVICE="$(yq -r '.policy_device' "$DEVICE_CONFIG")"

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

accelerate_args=()
if [ "$MULTI_GPU" = "true" ]; then
  accelerate_args+=(--multi_gpu)
fi

UV_CACHE_DIR=/tmp/uv-cache uv run accelerate launch \
  --num_machines="$NUM_MACHINES" \
  "${accelerate_args[@]}" \
  --num_processes="$NUM_PROCESSES" \
  --mixed_precision="$MIXED_PRECISION" \
  --dynamo_backend="$DYNAMO_BACKEND" \
  "$SCRIPT_DIR/../.venv/bin/lerobot-train" \
  ${CONFIG_PATH:+--config_path=$CONFIG_PATH} \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision="main" \
  --policy.type="act" \
  --policy.repo_id="$POLICY_REPO_ID" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --optimizer.lr="$OPTIMIZER_LR" \
  --num_workers="$NUM_WORKERS" \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --save_freq="900" \
  --log_freq="20" \
  --policy.push_to_hub="true" \
  --job_name="$JOB_NAME" \
  --wandb.project="act" \
  --wandb.enable="true" \
  --policy.device="$POLICY_DEVICE" \
  --resume="$RESUME"
