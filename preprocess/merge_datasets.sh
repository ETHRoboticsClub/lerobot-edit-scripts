#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  preprocess/merge_datasets.sh <source_repo_id_1> <source_repo_id_2> <new_repo_id> [source_root_1] [source_root_2]

Examples:
  preprocess/merge_datasets.sh ETHRC/towelspring26_2 ETHRC/towelspring26_3 ETHRC/towelspring26_merged
  preprocess/merge_datasets.sh ETHRC/a ETHRC/b ETHRC/merged /tmp/a /tmp/b

Environment overrides:
  PUSH_TO_HUB=true|false   Whether to upload the merged dataset. Default: true
  NEW_ROOT=/path           Optional output root for the merged dataset
  UV_CACHE_DIR=/path       Optional uv cache directory. Default: .uv-cache
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage >&2
  exit 1
fi

SOURCE_REPO_ID_1="${1}"
SOURCE_REPO_ID_2="${2}"
NEW_REPO_ID="${3}"
SOURCE_ROOT_1="${4:-}"
SOURCE_ROOT_2="${5:-}"

if [[ -n "${SOURCE_ROOT_1}" && -z "${SOURCE_ROOT_2}" ]] || [[ -z "${SOURCE_ROOT_1}" && -n "${SOURCE_ROOT_2}" ]]; then
  echo "Provide both source_root_1 and source_root_2, or neither." >&2
  exit 1
fi

PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
NEW_ROOT="${NEW_ROOT:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

REPO_IDS="[\"${SOURCE_REPO_ID_1}\",\"${SOURCE_REPO_ID_2}\"]"

cmd=(
  uv
  run
  lerobot-edit-dataset
  --operation.type
  merge
  --operation.repo_ids
  "${REPO_IDS}"
  --new_repo_id
  "${NEW_REPO_ID}"
  --push_to_hub
  "${PUSH_TO_HUB}"
)

if [[ -n "${SOURCE_ROOT_1}" ]]; then
  ROOTS="[\"${SOURCE_ROOT_1}\",\"${SOURCE_ROOT_2}\"]"
  cmd+=(
    --operation.roots
    "${ROOTS}"
  )
fi

if [[ -n "${NEW_ROOT}" ]]; then
  cmd+=(
    --new_root
    "${NEW_ROOT}"
  )
fi

UV_CACHE_DIR="${UV_CACHE_DIR}" "${cmd[@]}"
