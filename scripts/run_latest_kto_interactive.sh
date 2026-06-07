#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export GOLDENGLOW_USE_TRAIN_OVERRIDE="${GOLDENGLOW_USE_TRAIN_OVERRIDE:-1}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/.vendor/train_override:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime_inference_gpu.json}"
LORA_PATH="${LORA_PATH:-model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.52}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}"
CTX_SIZE="${CTX_SIZE:-10000}"
MAX_TOKENS="${MAX_TOKENS:-1536}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
PROMPT_EVIDENCE_MAX_CHARS_PER_DOC="${PROMPT_EVIDENCE_MAX_CHARS_PER_DOC:-1000}"
PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS="${PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS:-12000}"
ANSWER_GROUNDING_MODE="${ANSWER_GROUNDING_MODE:-quote}"
CONCLUSION_PROMPT_MODE="${CONCLUSION_PROMPT_MODE:-minimal}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}"
QUESTION="${1:-}"

if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "[error] missing runtime config: $RUNTIME_CONFIG" >&2
  exit 2
fi

if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
  echo "[error] invalid LoRA path: $LORA_PATH" >&2
  echo "Pass LORA_PATH=/path/to/adapter_dir if needed." >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" scripts/run_cpu_inference.py
  --runtime-config "$RUNTIME_CONFIG"
  --device cuda
  --backend vllm
  --lora-path "$LORA_PATH"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-retrieval-rounds "$MAX_RETRIEVAL_ROUNDS"
  --ctx-size "$CTX_SIZE"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --prompt-evidence-max-chars-per-doc "$PROMPT_EVIDENCE_MAX_CHARS_PER_DOC"
  --prompt-conclusion-evidence-max-total-chars "$PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS"
  --answer-grounding-mode "$ANSWER_GROUNDING_MODE"
  --conclusion-prompt-mode "$CONCLUSION_PROMPT_MODE"
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  cmd+=(--enforce-eager)
fi
if [[ "$DISABLE_WEB_CONTEXT" == "1" ]]; then
  cmd+=(--disable-web-context)
fi
if [[ -n "$QUESTION" ]]; then
  cmd+=("$QUESTION")
fi

echo "[config] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] python=$PYTHON_BIN"
echo "[config] runtime_config=$RUNTIME_CONFIG"
echo "[config] lora_path=$LORA_PATH"
echo "[config] max_retrieval_rounds=$MAX_RETRIEVAL_ROUNDS grounding=$ANSWER_GROUNDING_MODE conclusion_prompt=$CONCLUSION_PROMPT_MODE"
echo "[hint] 输入 exit / quit / q 退出交互。"

export CUDA_VISIBLE_DEVICES
exec "${cmd[@]}"
