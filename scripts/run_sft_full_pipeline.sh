#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/qwen3.5-4b}"
SFT_LORA="${SFT_LORA:-$ROOT_DIR/model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_inference_gpu.json}"

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$ROOT_DIR"

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
  echo "Base model not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$SFT_LORA/adapter_config.json" ]]; then
  echo "SFT LoRA not found: $SFT_LORA" >&2
  exit 1
fi

answer_only_args=()
if [[ "${ANSWER_ONLY:-0}" == "1" ]]; then
  answer_only_args+=(--answer-only)
fi

"$PYTHON_BIN" scripts/run_cpu_inference.py \
  --runtime-config "$RUNTIME_CONFIG" \
  --backend vllm \
  --device cuda:0 \
  --base-model "$BASE_MODEL" \
  --lora-path "$SFT_LORA" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.62}" \
  --ctx-size "${CTX_SIZE:-6144}" \
  --max-tokens "${MAX_TOKENS:-512}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-0.9}" \
  --repeat-penalty "${REPEAT_PENALTY:-1.05}" \
  "${answer_only_args[@]}" \
  "$@"
