#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT_DIR/data/processed/sft_data/teacher_v2_plus_prompt_supplement_v4}"
SOURCE_BASENAME="${SOURCE_BASENAME:-$(basename "$SOURCE_DIR")}"
DATASET_DIR="${DATASET_DIR:-$ROOT_DIR/data/processed/llama_factory/$SOURCE_BASENAME}"
CONFIG_TEMPLATE_PATH="${CONFIG_TEMPLATE_PATH:-$ROOT_DIR/src/config/llama_factory_config.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${SOURCE_BASENAME}_qwen35_4b_train}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$SOURCE_BASENAME}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-$ROOT_DIR/model/lora/${SOURCE_BASENAME}_qwen35_4b}"
WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/outputs/wandb}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/.cache}"
HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_PATH:-$CACHE_DIR/llama_factory_${SOURCE_BASENAME}.yaml}"
USE_PYTHON_OVERLAY=0

cd "$ROOT_DIR"

mkdir -p "$WANDB_DIR" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"
mkdir -p "$PYTHON_OVERLAY_DIR"
mkdir -p "$(dirname "$LORA_OUTPUT_DIR")"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_RUN_NAME
export WANDB_NAME="$WANDB_RUN_NAME"
export WANDB_RUN_GROUP
export WANDB_DIR
export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export MPLCONFIGDIR
export SOURCE_BASENAME
export DATASET_DIR
export CONFIG_TEMPLATE_PATH
export GENERATED_CONFIG_PATH
export LORA_OUTPUT_DIR
export TOKENIZERS_PARALLELISM=false

if [[ -d "$PYTHON_OVERLAY_DIR/llamafactory" ]]; then
  USE_PYTHON_OVERLAY=1
  export PYTHONPATH="$PYTHON_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

LLAMAFACTORY_PATH="$(command -v "$LLAMAFACTORY_BIN" || true)"
HAS_LLAMFACTORY_MODULE="$("$PYTHON_BIN" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("llamafactory") else "0")
PY
)"

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

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

template_path = Path(os.environ["CONFIG_TEMPLATE_PATH"])
generated_path = Path(os.environ["GENERATED_CONFIG_PATH"])
dataset_dir = Path(os.environ["DATASET_DIR"])
source_basename = os.environ["SOURCE_BASENAME"]
lora_output_dir = Path(os.environ["LORA_OUTPUT_DIR"])
wandb_run_name = os.environ["WANDB_RUN_NAME"]

replacements = {
    "dataset_dir": dataset_dir.as_posix(),
    "dataset": f"{source_basename}_train",
    "eval_dataset": f"{source_basename}_val",
    "output_dir": lora_output_dir.as_posix(),
    "run_name": wandb_run_name,
}

lines = template_path.read_text(encoding="utf-8").splitlines()
rendered: list[str] = []
for line in lines:
    stripped = line.strip()
    key = stripped.split(":", 1)[0] if ":" in stripped else None
    if key in replacements and not line.startswith((" ", "\t")):
        rendered.append(f"{key}: {replacements[key]}")
    else:
        rendered.append(line)
generated_path.parent.mkdir(parents=True, exist_ok=True)
generated_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
PY

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training processes=$VISIBLE_GPU_COUNT"
echo "Source dataset=$SOURCE_DIR"
echo "LLaMA-Factory dataset_dir=$DATASET_DIR"
echo "LoRA output_dir=$LORA_OUTPUT_DIR"
echo "Config=$GENERATED_CONFIG_PATH"
echo "Use Python overlay=$USE_PYTHON_OVERLAY"

if [[ -z "$LLAMAFACTORY_PATH" && "$HAS_LLAMFACTORY_MODULE" != "1" ]]; then
  echo "LLaMA-Factory is not available in the current Python environment." >&2
  echo "Please activate the train env or install LLaMA-Factory before training." >&2
  echo "Expected one of:" >&2
  echo "  1. \`llamafactory-cli\` in PATH" >&2
  echo "  2. Python module \`llamafactory\` importable by \`$PYTHON_BIN\`" >&2
  exit 1
fi

if [[ -z "$LLAMAFACTORY_PATH" ]]; then
  exec "$PYTHON_BIN" -m llamafactory.cli train "$GENERATED_CONFIG_PATH"
fi

exec "$LLAMAFACTORY_BIN" train "$GENERATED_CONFIG_PATH"
