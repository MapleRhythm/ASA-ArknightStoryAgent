#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
elif [[ -x "/mnt/store/zhb/conda_envs/train/bin/python" ]]; then
  PYTHON_BIN="/mnt/store/zhb/conda_envs/train/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

SFT_CONFIG="${SFT_CONFIG:-src/config/llama_factory_schema_sft_patch_v1_from_soda_lora_config.yaml}"
DPO_CONFIG="${DPO_CONFIG:-src/config/llama_factory_soda_conclusion_dpo_v1_from_schema_sft_config.yaml}"
SFT_DATA="${SFT_DATA:-data/processed/llama_factory/schema_sft_patch_v1}"
DPO_DATA="${DPO_DATA:-data/processed/llama_factory/soda_conclusion_dpo_v1}"
SFT_OUT="${SFT_OUT:-model/lora/schema_sft_patch_v1_from_soda_lora_qwen35_4b_lr8e6_epoch1}"
DPO_OUT="${DPO_OUT:-model/lora/soda_conclusion_dpo_v1_from_schema_sft_qwen35_4b_lr5e7_beta002_ftx008_epoch2}"
LOG_DIR="${LOG_DIR:-logs/schema_sft_conclusion_dpo_flow_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_SFT="${RUN_SFT:-1}"
RUN_DPO="${RUN_DPO:-1}"
RUN_EVAL="${RUN_EVAL:-0}"

mkdir -p "$LOG_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "[error] missing directory: $path" >&2
    exit 2
  fi
}

patch_trl_optional_dependency_probe() {
  local path="$ROOT_DIR/.python_packages/train/trl/import_utils.py"
  [[ -f "$path" ]] || return 0
  "$PYTHON_BIN" - "$path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if "def _as_available(value):" not in text:
    marker = 'LIGER_KERNEL_MIN_VERSION = "0.5.8"\n'
    replacement = marker + '''

def _as_available(value):
    """Normalize newer Transformers `(available, version)` return values to bool."""
    if isinstance(value, tuple):
        return value[0]
    return value
'''
    if marker not in text:
        raise SystemExit(f"cannot patch {path}: marker not found")
    text = text.replace(marker, replacement, 1)

replacements = {
    '_deepspeed_available = _is_package_available("deepspeed")': '_deepspeed_available = _as_available(_is_package_available("deepspeed"))',
    '_fastapi_available = _is_package_available("fastapi")': '_fastapi_available = _as_available(_is_package_available("fastapi"))',
    '_joblib_available = _is_package_available("joblib")': '_joblib_available = _as_available(_is_package_available("joblib"))',
    '_llm_blender_available = _is_package_available("llm_blender")': '_llm_blender_available = _as_available(_is_package_available("llm_blender"))',
    '_math_verify_available = _is_package_available("math_verify")': '_math_verify_available = _as_available(_is_package_available("math_verify"))',
    '_mergekit_available = _is_package_available("mergekit")': '_mergekit_available = _as_available(_is_package_available("mergekit"))',
    '_pydantic_available = _is_package_available("pydantic")': '_pydantic_available = _as_available(_is_package_available("pydantic"))',
    '_requests_available = _is_package_available("requests")': '_requests_available = _as_available(_is_package_available("requests"))',
    '_unsloth_available = _is_package_available("unsloth")': '_unsloth_available = _as_available(_is_package_available("unsloth"))',
    '_uvicorn_available = _is_package_available("uvicorn")': '_uvicorn_available = _as_available(_is_package_available("uvicorn"))',
    '_vllm_ascend_available = _is_package_available("vllm_ascend")': '_vllm_ascend_available = _as_available(_is_package_available("vllm_ascend"))',
    '_weave_available = _is_package_available("weave")': '_weave_available = _as_available(_is_package_available("weave"))',
}
for src, dst in replacements.items():
    text = text.replace(src, dst)
path.write_text(text, encoding="utf-8")
PY
}

summarize_dataset() {
  "$PYTHON_BIN" - "$SFT_DATA/summary.json" "$DPO_DATA/summary.json" <<'PY'
import json
import sys
from pathlib import Path

for path in map(Path, sys.argv[1:]):
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[dataset] {path.parent.name}: {json.dumps(payload.get('records'), ensure_ascii=False)}")
PY
}

train_config() {
  local name="$1"
  local config="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[stage] train $name config=$config cuda=$CUDA_VISIBLE_DEVICES log=$log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m llamafactory.cli train "$config" 2>&1 | tee "$log"
}

validate_adapter() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path/adapter_config.json" || ! -f "$path/adapter_model.safetensors" ]]; then
    echo "[error] $name adapter is incomplete: $path" >&2
    exit 3
  fi
  echo "[ok] $name adapter=$path"
}

echo "[config] root=$ROOT_DIR"
echo "[config] python=$PYTHON_BIN"
echo "[config] cuda=$CUDA_VISIBLE_DEVICES"
echo "[config] log_dir=$LOG_DIR"

require_file "$SFT_CONFIG"
require_file "$DPO_CONFIG"
require_dir "$SFT_DATA"
require_dir "$DPO_DATA"
require_file "$SFT_DATA/dataset_info.json"
require_file "$DPO_DATA/dataset_info.json"
require_file "$SFT_DATA/train.json"
require_file "$DPO_DATA/train.json"

patch_trl_optional_dependency_probe
summarize_dataset

"$PYTHON_BIN" - <<'PY'
import sys
from trl.import_utils import is_mergekit_available
print("[check] mergekit_available=", is_mergekit_available(), type(is_mergekit_available()).__name__)
from trl import DPOTrainer
print("[check] DPOTrainer import ok")
PY

if [[ "$RUN_SFT" == "1" ]]; then
  train_config "schema_sft_patch_v1" "$SFT_CONFIG"
fi
validate_adapter "schema_sft_patch_v1" "$SFT_OUT"

if [[ "$RUN_DPO" == "1" ]]; then
  train_config "soda_conclusion_dpo_v1" "$DPO_CONFIG"
fi
validate_adapter "soda_conclusion_dpo_v1" "$DPO_OUT"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[stage] eval final adapter"
  RUN_NAME="${RUN_NAME:-eval_soda_conclusion_dpo_v1_$(date +%Y%m%d_%H%M%S)}" \
  LORA_PATH="$DPO_OUT" \
  GPUS="${GPUS:-$CUDA_VISIBLE_DEVICES}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.48}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" \
  MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}" \
  DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
  RUN_EVAL50="${RUN_EVAL50:-1}" \
  RUN_HARD="${RUN_HARD:-1}" \
  bash scripts/run_eval50_hard10_gpu_abstain_flow.sh
fi

echo "[done] schema SFT + conclusion DPO flow complete"
