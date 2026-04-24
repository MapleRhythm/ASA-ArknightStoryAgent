#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/data/processed/sft_data/teacher_v2}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/processed/llama_factory/teacher_v2}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config/llama_factory_config.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
MASTER_PORT="${MASTER_PORT:-29501}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-teacher_v2_qwen35_4b_train}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-teacher_v2}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/outputs/wandb}"

cd "$ROOT_DIR"

mkdir -p "$WANDB_DIR"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_RUN_GROUP
export WANDB_DIR
LLAMAFACTORY_PATH="$(command -v "$LLAMAFACTORY_BIN" || true)"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
fi

VISIBLE_GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
import os
devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(len(devices))
PY
)"

"$PYTHON_BIN" scripts/llama_factory/prepare_sft_dataset.py \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$DATASET_DIR"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training processes=$VISIBLE_GPU_COUNT"

if [[ "$VISIBLE_GPU_COUNT" -gt 1 ]]; then
  if [[ -z "$LLAMAFACTORY_PATH" ]]; then
    echo "Cannot find executable: $LLAMAFACTORY_BIN" >&2
    exit 1
  fi
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$VISIBLE_GPU_COUNT" \
    --master_port="$MASTER_PORT" \
    "$LLAMAFACTORY_PATH" train "$CONFIG_PATH"
fi

exec "$LLAMAFACTORY_BIN" train "$CONFIG_PATH"
