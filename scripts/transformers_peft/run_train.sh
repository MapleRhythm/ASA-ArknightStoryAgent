#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config/transformers_peft_train.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-teacher_v2_plus_prompt_supplement_v2_qwen35_4b_train}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-teacher_v2_plus_prompt_supplement_v2}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/outputs/wandb}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/.cache}"
HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"

cd "$ROOT_DIR"

mkdir -p "$WANDB_DIR" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_RUN_GROUP
export WANDB_DIR
export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export MPLCONFIGDIR
export TOKENIZERS_PARALLELISM=false

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
fi

VISIBLE_GPU_COUNT="$("$PYTHON_BIN" - <<'PY'
import os
devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
print(len(devices))
PY
)"

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training processes=$VISIBLE_GPU_COUNT"

if [[ "$VISIBLE_GPU_COUNT" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$VISIBLE_GPU_COUNT" \
    scripts/transformers_peft/train_sft.py \
    --config "$CONFIG_PATH"
fi

exec "$PYTHON_BIN" scripts/transformers_peft/train_sft.py --config "$CONFIG_PATH"
