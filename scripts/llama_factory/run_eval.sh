#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/data/processed/sft_data/teacher_v2}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/processed/llama_factory/teacher_v2}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$ROOT_DIR/src/config/llama_factory_eval.yaml}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_qwen35_4b/generated_predictions.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_qwen35_4b/custom_metrics.json}"
REFERENCE_FILE="${REFERENCE_FILE:-$DATASET_DIR/test.json}"
EVAL_GPUS="${EVAL_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29511}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-teacher_v2_qwen35_4b_eval}"
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
  export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"
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
echo "Evaluation processes=$VISIBLE_GPU_COUNT"

if [[ "$VISIBLE_GPU_COUNT" -gt 1 ]]; then
  if [[ -z "$LLAMAFACTORY_PATH" ]]; then
    echo "Cannot find executable: $LLAMAFACTORY_BIN" >&2
    exit 1
  fi
  "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node="$VISIBLE_GPU_COUNT" \
    --master_port="$MASTER_PORT" \
    "$LLAMAFACTORY_PATH" train "$EVAL_CONFIG_PATH"
else
  "$LLAMAFACTORY_BIN" train "$EVAL_CONFIG_PATH"
fi

exec "$PYTHON_BIN" scripts/llama_factory/summarize_predictions.py \
  --reference-file "$REFERENCE_FILE" \
  --predictions-file "$PREDICTIONS_FILE" \
  --output-file "$SUMMARY_FILE"
