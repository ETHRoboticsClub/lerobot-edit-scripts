# if [ -z "${WANDB_API_KEY:-}" ]; then
#   printf 'Missing required environment variable: WANDB_API_KEY\n'
#   printf 'Run `wandb login` or export WANDB_API_KEY before starting training.\n'
#   exit 1
# fi

export WANDB_MODE=online
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS="${TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
JOB_CONFIG="${JOB_CONFIG:-$SCRIPT_DIR/configs/job.yaml}"

detect_device_config() {
  local gpu_count gpu_name gpu_name_lower gpu_names l40s_count fallback_config
  fallback_config="$CONFIG_DIR/aws_gpul.yaml"
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
  gpu_name="$(printf '%s\n' "$gpu_names" | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  gpu_name_lower="$(printf '%s' "$gpu_name" | tr '[:upper:]' '[:lower:]')"
  gpu_count="$(printf '%s\n' "$gpu_names" | awk 'NF { count++ } END { print count + 0 }')"
  l40s_count="$(printf '%s\n' "$gpu_names" | awk 'tolower($0) ~ /l40s/ { count++ } END { print count + 0 }')"

  if [ "$gpu_count" -eq 4 ] && [ "$l40s_count" -eq 4 ]; then
    printf '%s\n' "$CONFIG_DIR/aws_gpul.yaml"
    return
  fi

  case "$gpu_name_lower" in
    *gb10*)
      printf '%s\n' "$CONFIG_DIR/spark.yaml"
      ;;
    *5080*)
      printf '%s\n' "$CONFIG_DIR/5080_workstation.yaml"
      ;;
    *a100*)
      printf '%s\n' "$CONFIG_DIR/aws_a100.yaml"
      ;;
    *)
      printf '%s\n' "$fallback_config"
      ;;
  esac
}

DEVICE_CONFIG="${DEVICE_CONFIG:-$(detect_device_config)}"

printf 'Using device config: %s\n' "$DEVICE_CONFIG"
printf 'Using job config: %s\n' "$JOB_CONFIG"

OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR:-}"
CONFIG_PATH_OVERRIDE="${CONFIG_PATH:-}"
RESUME_OVERRIDE="${RESUME:-}"

eval "$(
  UV_CACHE_DIR=/tmp/uv-cache uv run python - "$DEVICE_CONFIG" "$JOB_CONFIG" <<'PY'
import shlex
import sys

import yaml

config = {}
for path in sys.argv[1:]:
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    config.update(loaded)

for key, value in config.items():
    if isinstance(value, (dict, list)):
        continue
    shell_key = key.upper()
    if isinstance(value, bool):
        shell_value = "true" if value else "false"
    else:
        shell_value = str(value)
    print(f'{shell_key}={shlex.quote(shell_value)}')
PY
)"

policy_args=()
while IFS= read -r policy_arg; do
  policy_args+=("$policy_arg")
done < <(
  UV_CACHE_DIR=/tmp/uv-cache uv run python - "$DEVICE_CONFIG" "$JOB_CONFIG" <<'PY'
import json
import sys

import yaml

config = {}
for path in sys.argv[1:]:
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    config.update(loaded)

policy = config.get("policy") or {}
for key, value in policy.items():
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(",", ":"))
    elif isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = "null"
    else:
        value = str(value)
    print(f"--policy.{key}={value}")
PY
)

dataset_args=()
while IFS= read -r dataset_arg; do
  dataset_args+=("$dataset_arg")
done < <(
  UV_CACHE_DIR=/tmp/uv-cache uv run python - "$DEVICE_CONFIG" "$JOB_CONFIG" <<'PY'
import json
import sys

import yaml

config = {}
for path in sys.argv[1:]:
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    config.update(loaded)


def emit_args(prefix, value):
    if prefix.endswith(".tfs") and isinstance(value, dict):
        print(f"--{prefix}={json.dumps(value, separators=(',', ':'))}")
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            emit_args(f"{prefix}.{key}", nested_value)
    elif isinstance(value, list):
        print(f"--{prefix}={json.dumps(value, separators=(',', ':'))}")
    elif isinstance(value, bool):
        print(f"--{prefix}={'true' if value else 'false'}")
    elif value is None:
        print(f"--{prefix}=null")
    else:
        print(f"--{prefix}={value}")


dataset = config.get("dataset") or {}
for key, value in dataset.items():
    emit_args(f"dataset.{key}", value)
PY
)

DATASET_REPO_ID="${DATASET_REPO_ID:-$(UV_CACHE_DIR=/tmp/uv-cache uv run python "$REPO_ROOT/project_config.py" dataset_repo_id)}"

DATASET_REPO_ID="${DATASET_REPO_ID_OVERRIDE:-$DATASET_REPO_ID}"
POLICY_REPO_ID="${POLICY_REPO_ID_OVERRIDE:-$POLICY_REPO_ID}"
JOB_NAME="${JOB_NAME_OVERRIDE:-$JOB_NAME}"
ARCHITECTURE="${ARCHITECTURE_OVERRIDE:-$ARCHITECTURE}"
RUN_NAME="${RUN_NAME_OVERRIDE:-$RUN_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR_OVERRIDE:-$CHECKPOINT_DIR}"
OUTPUT_DIR_DEFAULT="$CHECKPOINT_DIR/$ARCHITECTURE/$RUN_NAME"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-$OUTPUT_DIR_DEFAULT}"
RESUME="${RESUME_OVERRIDE:-${RESUME:-true}}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-last}"
if [ -n "$CONFIG_PATH_OVERRIDE" ]; then
  CONFIG_PATH="$CONFIG_PATH_OVERRIDE"
elif [ "$RESUME" = "true" ]; then
  CONFIG_PATH="$OUTPUT_DIR/checkpoints/$CHECKPOINT_NAME/pretrained_model/train_config.json"
else
  CONFIG_PATH=""
fi

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
CHECKPOINT_STEPS="${CHECKPOINT_STEPS_OVERRIDE:-${CHECKPOINT_STEPS:-900}}"

required_vars=(
  DATASET_REPO_ID
  POLICY_REPO_ID
  OUTPUT_DIR
  JOB_NAME
  ARCHITECTURE
  RUN_NAME
  CHECKPOINT_DIR
  NUM_MACHINES
  MULTI_GPU
  NUM_PROCESSES
  MIXED_PRECISION
  DYNAMO_BACKEND
  BATCH_SIZE
  STEPS
  OPTIMIZER_LR
  NUM_WORKERS
  VIDEO_BACKEND
  POLICY_DEVICE
)

for var_name in "${required_vars[@]}"; do
  if [ -z "${!var_name}" ]; then
    printf 'Missing config value: %s\n' "$var_name"
    printf 'Device config file: %s\n' "$DEVICE_CONFIG"
    printf 'Job config file: %s\n' "$JOB_CONFIG"
    exit 1
  fi
done

if [ "$RESUME" = "true" ] && [ ! -f "$CONFIG_PATH" ]; then
  printf 'Resume config file not found: %s\n' "$CONFIG_PATH"
  printf 'Set CONFIG_PATH explicitly or choose an OUTPUT_DIR/CHECKPOINT_NAME with a valid checkpoint.\n'
  exit 1
fi

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
  "$REPO_ROOT/.venv/bin/lerobot-train" \
  ${CONFIG_PATH:+--config_path=$CONFIG_PATH} \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.revision="main" \
  "${dataset_args[@]}" \
  --policy.type="act" \
  --policy.repo_id="$POLICY_REPO_ID" \
  "${policy_args[@]}" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --optimizer.lr="$OPTIMIZER_LR" \
  --policy.optimizer_lr="$OPTIMIZER_LR" \
  --num_workers="$NUM_WORKERS" \
  --dataset.video_backend="$VIDEO_BACKEND" \
  --save_freq="$CHECKPOINT_STEPS" \
  --log_freq="20" \
  --policy.push_to_hub="true" \
  --job_name="$JOB_NAME" \
  --wandb.project="act" \
  --wandb.entity="eth-robotics-club" \
  --wandb.enable="true" \
  --policy.device="$POLICY_DEVICE" \
  --resume="$RESUME"
