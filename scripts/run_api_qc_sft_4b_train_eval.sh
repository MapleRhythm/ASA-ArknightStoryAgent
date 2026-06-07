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
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

SCHEMA_CONFIG="${SCHEMA_CONFIG:-src/config/llama_factory_schema_sft_patch_v1_api_qc_from_soda_lora_config.yaml}"
CONCLUSION_CONFIG="${CONCLUSION_CONFIG:-src/config/llama_factory_conclusion_chosen_sft_v1_api_qc_from_schema_sft_cutoff3072_config.yaml}"
SCHEMA_OUT="${SCHEMA_OUT:-model/lora/schema_sft_patch_v1_api_qc_from_soda_lora_qwen35_4b_lr8e6_epoch1}"
CONCLUSION_OUT="${CONCLUSION_OUT:-model/lora/conclusion_chosen_sft_v1_api_qc_from_schema_sft_qwen35_4b_lr2e6_epoch2_cutoff3072}"
LOG_DIR="${LOG_DIR:-logs/api_qc_sft_4b_train_eval_$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_SCHEMA="${RUN_SCHEMA:-1}"
RUN_CONCLUSION="${RUN_CONCLUSION:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p "$LOG_DIR"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[error] missing file: $path" >&2
    exit 2
  fi
}

require_adapter() {
  local name="$1"
  local path="$2"
  if [[ ! -f "$path/adapter_config.json" || ! -f "$path/adapter_model.safetensors" ]]; then
    echo "[error] incomplete adapter for $name: $path" >&2
    exit 3
  fi
  echo "[ok] $name adapter=$path"
}

train_config() {
  local name="$1"
  local config="$2"
  local log="$LOG_DIR/${name}.log"
  echo "[stage] train $name config=$config cuda=$CUDA_VISIBLE_DEVICES log=$log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m llamafactory.cli train "$config" 2>&1 | tee "$log"
}

echo "[config] root=$ROOT_DIR"
echo "[config] python=$PYTHON_BIN"
echo "[config] cuda=$CUDA_VISIBLE_DEVICES"
echo "[config] log_dir=$LOG_DIR"
echo "[config] schema_config=$SCHEMA_CONFIG"
echo "[config] conclusion_config=$CONCLUSION_CONFIG"

require_file "$SCHEMA_CONFIG"
require_file "$CONCLUSION_CONFIG"

if [[ "$RUN_SCHEMA" == "1" && -e "$SCHEMA_OUT" && ! -f "$SCHEMA_OUT/adapter_config.json" ]]; then
  echo "[error] schema output exists but is not a complete adapter: $SCHEMA_OUT" >&2
  exit 4
fi
if [[ "$RUN_CONCLUSION" == "1" && -e "$CONCLUSION_OUT" && ! -f "$CONCLUSION_OUT/adapter_config.json" ]]; then
  echo "[error] conclusion output exists but is not a complete adapter: $CONCLUSION_OUT" >&2
  exit 4
fi

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

for path in [
    Path("data/processed/llama_factory/schema_sft_patch_v1_api_qc_v1/summary.json"),
    Path("data/processed/llama_factory/conclusion_chosen_sft_v1_api_qc_v1/summary.json"),
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"[dataset] {path.parent.name} output_splits={payload.get('output_splits')} actions={payload.get('actions')}")
PY

if [[ "$RUN_SCHEMA" == "1" ]]; then
  train_config "schema_sft_patch_v1_api_qc" "$SCHEMA_CONFIG"
fi
require_adapter "schema_sft_patch_v1_api_qc" "$SCHEMA_OUT"

if [[ "$RUN_CONCLUSION" == "1" ]]; then
  train_config "conclusion_chosen_sft_v1_api_qc" "$CONCLUSION_CONFIG"
fi
require_adapter "conclusion_chosen_sft_v1_api_qc" "$CONCLUSION_OUT"

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "[stage] eval final adapter"
  RUN_NAME="${RUN_NAME:-eval_api_qc_sft_4b_$(date +%Y%m%d_%H%M%S)}" \
  LORA_PATH="$CONCLUSION_OUT" \
  GPUS="${EVAL_GPUS:-$CUDA_VISIBLE_DEVICES}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.52}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}" \
  MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}" \
  DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
  RUN_EVAL50="${RUN_EVAL50:-1}" \
  RUN_HARD="${RUN_HARD:-1}" \
  bash scripts/run_eval50_hard10_gpu_abstain_flow.sh 2>&1 | tee "$LOG_DIR/eval.log"
fi

echo "[done] api qc sft 4b train/eval flow complete"
