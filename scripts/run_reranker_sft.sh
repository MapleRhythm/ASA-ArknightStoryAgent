#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/data/processed/evidence_chain_reranker/batch_v1_strict/reranker_pairwise.jsonl}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/reranker/bge-reranker-v2-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/model/reranker/bge-reranker-v2-m3-evidence-chain-answerability}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/.cache}"
HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
FILTERED_OVERLAY_DIR="${FILTERED_OVERLAY_DIR:-$CACHE_DIR/python_overlay_filtered_reranker}"

MAX_LENGTH="${MAX_LENGTH:-1024}"
EVAL_RATIO="${EVAL_RATIO:-0.05}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
REPORT_TO="${REPORT_TO:-none}"
SEED="${SEED:-20260509}"
BF16="${BF16:-auto}"
FP16="${FP16:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
LOSS_TYPE="${LOSS_TYPE:-softplus}"
DPO_BETA="${DPO_BETA:-0.1}"
DRY_RUN="${DRY_RUN:-false}"

cd "$ROOT_DIR"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "TRAIN_FILE does not exist: $TRAIN_FILE" >&2
  echo "Generate it first with scripts/run_evidence_chain_batch.py." >&2
  exit 1
fi

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "BASE_MODEL does not exist: $BASE_MODEL" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR" "$PYTHON_OVERLAY_DIR"
export PYTHON_OVERLAY_DIR
export FILTERED_OVERLAY_DIR

if [[ -d "$PYTHON_OVERLAY_DIR/transformers" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
from pathlib import Path

src = Path(os.environ["PYTHON_OVERLAY_DIR"]).resolve()
dst = Path(os.environ["FILTERED_OVERLAY_DIR"]).resolve()
allow_prefixes = (
    "accelerate",
    "accelerate-",
    "filelock",
    "filelock-",
    "fsspec",
    "fsspec-",
    "huggingface_hub",
    "huggingface_hub-",
    "numpy",
    "numpy-",
    "packaging",
    "packaging-",
    "peft",
    "peft-",
    "regex",
    "regex-",
    "requests",
    "requests-",
    "safetensors",
    "safetensors-",
    "sentencepiece",
    "sentencepiece-",
    "tokenizers",
    "tokenizers-",
    "tqdm",
    "tqdm-",
    "transformers",
    "transformers-",
    "yaml",
    "PyYAML-",
)

dst.mkdir(parents=True, exist_ok=True)
for child in list(dst.iterdir()):
    if child.is_symlink() or child.is_file():
        child.unlink()
    elif child.is_dir():
        shutil.rmtree(child)

for child in src.iterdir():
    if not child.name.startswith(allow_prefixes):
        continue
    target = dst / child.name
    target.symlink_to(child)
PY
  export PYTHONPATH="$FILTERED_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
  echo "Using python overlay: $FILTERED_OVERLAY_DIR"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import transformers  # noqa: F401
import accelerate  # noqa: F401
PY
then
  echo "PYTHON_BIN lacks required packages even after overlay: torch, transformers, accelerate" >&2
  echo "Install them in the selected environment or check PYTHON_OVERLAY_DIR=$PYTHON_OVERLAY_DIR." >&2
  exit 1
fi

export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export MPLCONFIGDIR
export PYTHON_OVERLAY_DIR
export FILTERED_OVERLAY_DIR
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

if [[ "$BF16" == "auto" ]]; then
  BF16="$("$PYTHON_BIN" - <<'PY'
try:
    import torch
    print("true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "false")
except Exception:
    print("false")
PY
)"
fi

COMMON_ARGS=(
  scripts/train_evidence_chain_reranker.py
  --train-file "$TRAIN_FILE"
  --model-name-or-path "$BASE_MODEL"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --eval-ratio "$EVAL_RATIO"
  --num-train-epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --save-steps "$SAVE_STEPS"
  --eval-steps "$EVAL_STEPS"
  --logging-steps "$LOGGING_STEPS"
  --report-to "$REPORT_TO"
  --seed "$SEED"
  --loss-type "$LOSS_TYPE"
  --dpo-beta "$DPO_BETA"
)

if [[ "$BF16" == "true" ]]; then
  COMMON_ARGS+=(--bf16)
fi

if [[ "$FP16" == "true" ]]; then
  COMMON_ARGS+=(--fp16)
fi

if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  COMMON_ARGS+=(--gradient-checkpointing)
fi

if [[ "$OVERWRITE_OUTPUT_DIR" == "true" ]]; then
  COMMON_ARGS+=(--overwrite-output-dir)
fi

if [[ "$DRY_RUN" == "true" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training processes=$VISIBLE_GPU_COUNT"
echo "Train file=$TRAIN_FILE"
echo "Base model=$BASE_MODEL"
echo "Output dir=$OUTPUT_DIR"

if [[ "$VISIBLE_GPU_COUNT" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$VISIBLE_GPU_COUNT" \
    "${COMMON_ARGS[@]}"
fi

exec "$PYTHON_BIN" "${COMMON_ARGS[@]}"
