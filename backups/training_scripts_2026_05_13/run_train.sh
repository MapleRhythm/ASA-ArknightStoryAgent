#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_BIN="${LLAMAFACTORY_BIN:-llamafactory-cli}"
SOURCE_DIR="${SOURCE_DIR:-}"
SOURCE_BASENAME="${SOURCE_BASENAME:-}"
DATASET_DIR="${DATASET_DIR:-}"
CONFIG_TEMPLATE_PATH="${CONFIG_TEMPLATE_PATH:-$ROOT_DIR/src/config/llama_factory_config.yaml}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2}"
WANDB_PROJECT="${WANDB_PROJECT:-goldenglow-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-}"
REPORT_TO="${REPORT_TO:-}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
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
GENERATED_CONFIG_PATH="${GENERATED_CONFIG_PATH:-$CACHE_DIR/llama_factory_${SOURCE_BASENAME}.yaml}"
USE_PYTHON_OVERLAY=0

cd "$ROOT_DIR"

if [[ -z "$SOURCE_DIR" ]]; then
  echo "SOURCE_DIR is required. Refusing to fall back to any default dataset." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "SOURCE_DIR does not exist: $SOURCE_DIR" >&2
  exit 1
fi

if [[ -z "$SOURCE_BASENAME" ]]; then
  SOURCE_BASENAME="$(basename "$SOURCE_DIR")"
fi

if [[ -z "$DATASET_DIR" ]]; then
  DATASET_DIR="$ROOT_DIR/data/processed/llama_factory/$SOURCE_BASENAME"
fi

if [[ -z "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME="${SOURCE_BASENAME}_qwen35_4b_train"
fi

if [[ -z "$WANDB_RUN_GROUP" ]]; then
  WANDB_RUN_GROUP="$SOURCE_BASENAME"
fi

if [[ -z "$LORA_OUTPUT_DIR" ]]; then
  LORA_OUTPUT_DIR="$ROOT_DIR/model/lora/${SOURCE_BASENAME}_qwen35_4b"
fi

if [[ -z "$REPORT_TO" ]]; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from importlib.metadata import PackageNotFoundError, version
try:
    version("wandb")
except PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    REPORT_TO="wandb"
  else
    REPORT_TO="none"
  fi
fi

mkdir -p "$WANDB_DIR" "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" "$DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$MPLCONFIGDIR"
mkdir -p "$PYTHON_OVERLAY_DIR"
mkdir -p "$(dirname "$LORA_OUTPUT_DIR")"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_RUN_NAME
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
export SOURCE_BASENAME
export DATASET_DIR
export CONFIG_TEMPLATE_PATH
export GENERATED_CONFIG_PATH
export LORA_OUTPUT_DIR
export PYTHON_OVERLAY_DIR
export FILTERED_OVERLAY_DIR
export REPORT_TO
export OVERWRITE_OUTPUT_DIR
export TOKENIZERS_PARALLELISM=false
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

if [[ -d "$PYTHON_OVERLAY_DIR/llamafactory" ]]; then
  USE_PYTHON_OVERLAY=1
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
export NPROC_PER_NODE="$VISIBLE_GPU_COUNT"

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
report_to = os.environ["REPORT_TO"]
overwrite_output_dir = os.environ["OVERWRITE_OUTPUT_DIR"].lower()

replacements = {
    "dataset_dir": dataset_dir.as_posix(),
    "dataset": f"{source_basename}_train",
    "eval_dataset": f"{source_basename}_val",
    "output_dir": lora_output_dir.as_posix(),
    "run_name": wandb_run_name,
    "report_to": report_to,
    "overwrite_output_dir": overwrite_output_dir,
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

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch

output_dir = Path(os.environ["LORA_OUTPUT_DIR"])
overwrite_output_dir = os.environ["OVERWRITE_OUTPUT_DIR"].strip().lower() == "true"
has_checkpoint = any(output_dir.glob("checkpoint-*"))
version_parts = str(torch.__version__).split("+", 1)[0].split(".")
torch_major = int(version_parts[0]) if len(version_parts) > 0 and version_parts[0].isdigit() else 0
torch_minor = int(version_parts[1]) if len(version_parts) > 1 and version_parts[1].isdigit() else 0
torch_is_old = (torch_major, torch_minor) < (2, 6)

if has_checkpoint and torch_is_old and not overwrite_output_dir:
    raise SystemExit(
        "Existing checkpoint detected in LORA_OUTPUT_DIR, but current torch<2.6 cannot safely resume "
        "optimizer/scheduler state with current transformers. Use a fresh output dir or set "
        "OVERWRITE_OUTPUT_DIR=true to start from scratch."
    )
PY

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Training processes=$VISIBLE_GPU_COUNT"
echo "Source dataset=$SOURCE_DIR"
echo "LLaMA-Factory dataset_dir=$DATASET_DIR"
echo "LoRA output_dir=$LORA_OUTPUT_DIR"
echo "Config=$GENERATED_CONFIG_PATH"
echo "Report to=$REPORT_TO"
echo "Overwrite output dir=$OVERWRITE_OUTPUT_DIR"
echo "Use Python overlay=$USE_PYTHON_OVERLAY"

if [[ -z "$LLAMAFACTORY_PATH" && "$HAS_LLAMFACTORY_MODULE" != "1" ]]; then
  echo "LLaMA-Factory is not available in the current Python environment." >&2
  echo "Please activate the train env or install LLaMA-Factory before training." >&2
  echo "Expected one of:" >&2
  echo "  1. \`llamafactory-cli\` in PATH" >&2
  echo "  2. Python module \`llamafactory\` importable by \`$PYTHON_BIN\`" >&2
  exit 1
fi

if [[ "$USE_PYTHON_OVERLAY" == "1" ]]; then
  exec "$PYTHON_BIN" -m llamafactory.cli train "$GENERATED_CONFIG_PATH"
fi

if [[ -z "$LLAMAFACTORY_PATH" ]]; then
  exec "$PYTHON_BIN" -m llamafactory.cli train "$GENERATED_CONFIG_PATH"
fi

exec "$LLAMAFACTORY_BIN" train "$GENERATED_CONFIG_PATH"
