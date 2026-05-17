#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
SOURCE_DIR="${SOURCE_DIR:-}"
DATASET_DIR="${DATASET_DIR:-}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$ROOT_DIR/src/config/llama_factory_eval.yaml}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_plus_prompt_supplement_v4_qwen35_4b/generated_predictions.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_plus_prompt_supplement_v4_qwen35_4b/custom_metrics.json}"
REFERENCE_FILE="${REFERENCE_FILE:-$DATASET_DIR/test.json}"
EVAL_GPUS="${EVAL_GPUS:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-teacher_v2_plus_prompt_supplement_v2_qwen35_4b_eval}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-teacher_v2_plus_prompt_supplement_v2}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/outputs/wandb}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/.cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"
HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
DATASETS_CACHE="${DATASETS_CACHE:-$HF_DATASETS_CACHE}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
FILTERED_OVERLAY_DIR="${FILTERED_OVERLAY_DIR:-$CACHE_DIR/python_overlay_filtered}"

cd "$ROOT_DIR"

if [[ -z "$SOURCE_DIR" ]]; then
  echo "SOURCE_DIR is required. Refusing to fall back to any default dataset." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "SOURCE_DIR does not exist: $SOURCE_DIR" >&2
  exit 1
fi

SOURCE_BASENAME="$(basename "$SOURCE_DIR")"

if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="$ROOT_DIR/data/processed/llama_factory/$SOURCE_BASENAME"
fi

if [[ "$PREDICTIONS_FILE" == "$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_plus_prompt_supplement_v4_qwen35_4b/generated_predictions.jsonl" ]]; then
  PREDICTIONS_FILE="$ROOT_DIR/outputs/llama_factory_eval/${SOURCE_BASENAME}_qwen35_4b/generated_predictions.jsonl"
fi

if [[ "$SUMMARY_FILE" == "$ROOT_DIR/outputs/llama_factory_eval/teacher_v2_plus_prompt_supplement_v4_qwen35_4b/custom_metrics.json" ]]; then
  SUMMARY_FILE="$ROOT_DIR/outputs/llama_factory_eval/${SOURCE_BASENAME}_qwen35_4b/custom_metrics.json"
fi

if [[ "$WANDB_RUN_NAME" == "teacher_v2_plus_prompt_supplement_v2_qwen35_4b_eval" ]]; then
  WANDB_RUN_NAME="${SOURCE_BASENAME}_qwen35_4b_eval"
fi

if [[ "$WANDB_RUN_GROUP" == "teacher_v2_plus_prompt_supplement_v2" ]]; then
  WANDB_RUN_GROUP="$SOURCE_BASENAME"
fi

REFERENCE_FILE="${REFERENCE_FILE:-$DATASET_DIR/test.json}"

mkdir -p "$WANDB_DIR"
mkdir -p "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" "$DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_RUN_GROUP
export WANDB_DIR
export XDG_CACHE_HOME
export HF_HOME
export HF_DATASETS_CACHE
export DATASETS_CACHE
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export MPLCONFIGDIR
export PYTHON_OVERLAY_DIR
export FILTERED_OVERLAY_DIR
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
LLAMAFACTORY_PATH="$(command -v "$LLAMAFACTORY_BIN" || true)"

if [[ -d "$PYTHON_OVERLAY_DIR/llamafactory" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["PYTHON_OVERLAY_DIR"]).resolve()
dst = Path(os.environ["FILTERED_OVERLAY_DIR"]).resolve()
allow_prefixes = (
    "llamafactory",
    "llamafactory-",
    "peft",
    "peft-",
    "transformers",
    "transformers-",
    "huggingface_hub",
    "huggingface_hub-",
    "tokenizers",
    "tokenizers-",
    "safetensors",
    "safetensors-",
)

dst.mkdir(parents=True, exist_ok=True)
for child in list(dst.iterdir()):
    if child.is_symlink() or child.is_file():
        child.unlink()
    elif child.is_dir():
        import shutil
        shutil.rmtree(child)

for child in src.iterdir():
    name = child.name
    if not name.startswith(allow_prefixes):
        continue
    target = dst / name
    target.symlink_to(child)
PY
  export PYTHONPATH="$FILTERED_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import transformers  # noqa: F401
PY
then
  FALLBACK_PYTHON="/home/zhb/miniconda3/bin/python"
  if [[ -x "$FALLBACK_PYTHON" ]] && "$FALLBACK_PYTHON" - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
import transformers  # noqa: F401
PY
  then
    PYTHON_BIN="$FALLBACK_PYTHON"
  fi
fi

PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
if [[ -d "$PYTHON_BIN_DIR" ]]; then
  export PATH="$PYTHON_BIN_DIR:$PATH"
fi

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

if [[ -z "$LLAMAFACTORY_PATH" ]]; then
  echo "Cannot find executable: $LLAMAFACTORY_BIN" >&2
  exit 1
fi

if [[ -d "$PYTHON_OVERLAY_DIR/llamafactory" ]]; then
  "$PYTHON_BIN" -m llamafactory.cli train "$EVAL_CONFIG_PATH"
else
  "$LLAMAFACTORY_BIN" train "$EVAL_CONFIG_PATH"
fi

exec "$PYTHON_BIN" scripts/llama_factory/summarize_predictions.py \
  --reference-file "$REFERENCE_FILE" \
  --predictions-file "$PREDICTIONS_FILE" \
  --output-file "$SUMMARY_FILE"
