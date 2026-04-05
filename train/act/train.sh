export WANDB_MODE=online

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_CONFIG="${DEVICE_CONFIG:-$SCRIPT_DIR/configs/aws_gpul.yaml}"

eval "$(
  UV_CACHE_DIR=/tmp/uv-cache uv run python - <<'PY' "$DEVICE_CONFIG"
import shlex
import sys

import yaml

with open(sys.argv[1]) as f:
    config = yaml.safe_load(f)

for key, value in config.items():
    shell_key = key.upper()
    if isinstance(value, bool):
        shell_value = "true" if value else "false"
    else:
        shell_value = str(value)
    print(f'{shell_key}={shlex.quote(shell_value)}')
PY
)"

DATASET_REPO_ID="${DATASET_REPO_ID_OVERRIDE:-$DATASET_REPO_ID}"
POLICY_REPO_ID="${POLICY_REPO_ID_OVERRIDE:-$POLICY_REPO_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_DIR}"
JOB_NAME="${JOB_NAME_OVERRIDE:-$JOB_NAME}"
RESUME="${RESUME:-true}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last}"
CONFIG_PATH="${CONFIG_PATH:-$OUTPUT_DIR/checkpoints/$CHECKPOINT_NAME/pretrained_model/train_config.json}"

NUM_MACHINES="${NUM_MACHINES_OVERRIDE:-$NUM_MACHINES}"
MULTI_GPU="${MULTI_GPU_OVERRIDE:-$MULTI_GPU}"
NUM_PROCESSES="${NUM_PROCESSES_OVERRIDE:-$NUM_PROCESSES}"
MIXED_PRECISION="${MIXED_PRECISION_OVERRIDE:-$MIXED_PRECISION}"
DYNAMO_BACKEND="${DYNAMO_BACKEND_OVERRIDE:-$DYNAMO_BACKEND}"
BATCH_SIZE="${BATCH_SIZE_OVERRIDE:-$BATCH_SIZE}"
STEPS="${STEPS_OVERRIDE:-$STEPS}"
OPTIMIZER_LR="${OPTIMIZER_LR_OVERRIDE:-$OPTIMIZER_LR}"
NUM_WORKERS="${NUM_WORKERS_OVERRIDE:-$NUM_WORKERS}"
VIDEO_BACKEND="${VIDEO_BACKEND_OVERRIDE:-$VIDEO_BACKEND}"
POLICY_DEVICE="${POLICY_DEVICE_OVERRIDE:-$POLICY_DEVICE}"

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
