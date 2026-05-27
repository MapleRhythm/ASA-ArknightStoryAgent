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

KTO_CONFIG="$ROOT_DIR/src/config/llama_factory_action_target_hard_negative_kto_v1_config.yaml"
DEFAULT_BASE_MODEL="$(awk -F': *' '$1 == "model_name_or_path" {print $2; exit}' "$KTO_CONFIG")"
DEFAULT_KTO_LORA="$(awk -F': *' '$1 == "output_dir" {print $2; exit}' "$KTO_CONFIG")"
if [[ -z "$DEFAULT_BASE_MODEL" || -z "$DEFAULT_KTO_LORA" ]]; then
  echo "Failed to parse model_name_or_path/output_dir from $KTO_CONFIG" >&2
  exit 1
fi
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/$DEFAULT_BASE_MODEL}"
KTO_LORA="${KTO_LORA:-$ROOT_DIR/$DEFAULT_KTO_LORA}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_inference_gpu.json}"

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$ROOT_DIR"

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
  echo "Merged SFT base model not found: $BASE_MODEL" >&2
  echo "Run scripts/run_action_target_hard_negative_kto_train.sh first; it will merge the SFT base if needed." >&2
  exit 1
fi

if [[ ! -f "$KTO_LORA/adapter_config.json" ]]; then
  echo "KTO LoRA not found: $KTO_LORA" >&2
  echo "Run scripts/run_action_target_hard_negative_kto_train.sh first." >&2
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
  --lora-path "$KTO_LORA" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.62}" \
  --ctx-size "${CTX_SIZE:-6144}" \
  --max-tokens "${MAX_TOKENS:-512}" \
  --temperature "${TEMPERATURE:-0.2}" \
  --top-p "${TOP_P:-0.9}" \
  --repeat-penalty "${REPEAT_PENALTY:-1.05}" \
  "${answer_only_args[@]}" \
  "$@"
