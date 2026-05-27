#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/src/config/llama_factory_online_teacher_chain_500_plus_sample50_config.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
FILTERED_OVERLAY_DIR="${FILTERED_OVERLAY_DIR:-$ROOT_DIR/outputs/.cache/python_overlay_filtered_llamafactory}"
export PYTHON_OVERLAY_DIR
export FILTERED_OVERLAY_DIR

DATASET_DIR="$ROOT_DIR/data/processed/llama_factory/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1"
MODEL_DIR="$ROOT_DIR/model/qwen3.5-4b"
OUTPUT_DIR="$ROOT_DIR/model/lora/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1_qwen35_4b_lr3e5_epoch1"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/outputs/.cache}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$TRAIN_GPUS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON_BIN" - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]) or 1)
PY
)}"
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export DATASETS_CACHE="${DATASETS_CACHE:-$HF_DATASETS_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_DIR/matplotlib}"

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"

if [[ -d "$PYTHON_OVERLAY_DIR/llamafactory" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os
import shutil
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
    "trl",
    "trl-",
)

dst.mkdir(parents=True, exist_ok=True)
for child in list(dst.iterdir()):
    if child.is_symlink() or child.is_file():
        child.unlink()
    elif child.is_dir():
        shutil.rmtree(child)

for child in src.iterdir():
    if child.name.startswith(allow_prefixes):
        (dst / child.name).symlink_to(child)
PY
  export PYTHONPATH="$FILTERED_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

for path in \
  "$CONFIG_PATH" \
  "$MODEL_DIR/config.json" \
  "$DATASET_DIR/dataset_info.json" \
  "$DATASET_DIR/train.json" \
  "$DATASET_DIR/val.json"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

LLAMAFACTORY_PATH="$(command -v "$LLAMAFACTORY_BIN" || true)"
HAS_LLAMFACTORY_MODULE="$("$PYTHON_BIN" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("llamafactory") else "0")
PY
)"

echo "Config: $CONFIG_PATH"
echo "Dataset: $DATASET_DIR"
echo "Base model: $MODEL_DIR"
echo "LoRA output: $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PYTHON_OVERLAY_DIR=$PYTHON_OVERLAY_DIR"
echo "FILTERED_OVERLAY_DIR=$FILTERED_OVERLAY_DIR"

if [[ -z "$LLAMAFACTORY_PATH" && "$HAS_LLAMFACTORY_MODULE" != "1" ]]; then
  cat >&2 <<'EOF'
LLaMA-Factory is not installed in the active Python environment.

Install it in the train env first:
  conda activate train
  pip install -U "llamafactory[torch,metrics]"

If pip install is unavailable, use source install:
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
  pip install -e "third_party/LLaMA-Factory[torch,metrics]"
EOF
  exit 1
fi

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "CHECK_ONLY=1, training command is available."
  exit 0
fi

if [[ -n "$LLAMAFACTORY_PATH" ]]; then
  exec "$LLAMAFACTORY_BIN" train "$CONFIG_PATH"
fi

exec "$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH"
