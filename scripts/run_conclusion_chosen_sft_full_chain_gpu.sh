#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/qwen3.5-4b}"
LORA_PATH="${LORA_PATH:-$ROOT_DIR/model/lora/conclusion_chosen_sft_v1_from_schema_sft_qwen35_4b_lr2e6_epoch2}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_inference_gpu.json}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "$BASE_MODEL/config.json" ]]; then
  echo "[error] base model not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$LORA_PATH/adapter_config.json" ]]; then
  echo "[error] LoRA adapter not found: $LORA_PATH" >&2
  exit 1
fi

ANSWER_ARGS=()
if [[ "${ANSWER_ONLY:-0}" == "1" ]]; then
  ANSWER_ARGS+=(--answer-only)
fi

WEB_ARGS=()
if [[ "${ENABLE_WEB_CONTEXT:-0}" == "1" ]]; then
  WEB_ARGS+=(--enable-web-context)
else
  WEB_ARGS+=(--disable-web-context)
fi

EAGER_ARGS=()
if [[ "${ENFORCE_EAGER:-1}" == "1" ]]; then
  EAGER_ARGS+=(--enforce-eager)
fi

COMMON_ARGS=(
  --runtime-config "$RUNTIME_CONFIG"
  --backend vllm
  --device cuda
  --base-model "$BASE_MODEL"
  --lora-path "$LORA_PATH"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.52}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-4096}"
  --ctx-size "${CTX_SIZE:-12000}"
  --max-retrieval-rounds "${MAX_RETRIEVAL_ROUNDS:-2}"
  --prompt-evidence-max-chars-per-doc "${PROMPT_EVIDENCE_MAX_CHARS_PER_DOC:-1800}"
  --prompt-conclusion-evidence-max-total-chars "${PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS:-24000}"
  --max-tokens "${MAX_TOKENS:-512}"
  --temperature "${TEMPERATURE:-0.2}"
  --top-p "${TOP_P:-0.9}"
  --repeat-penalty "${REPEAT_PENALTY:-1.05}"
  "${WEB_ARGS[@]}"
  "${EAGER_ARGS[@]}"
)

if [[ "${ONE_SHOT:-auto}" != "0" && $# -gt 0 && "${1:-}" != --* ]]; then
  QUESTION="$1"
  shift
  RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/full_chain_runs/conclusion_chosen_sft_$RUN_ID}"
  LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/full_chain_runs}"
  mkdir -p "$OUT_DIR" "$LOG_DIR"
  QUESTION_FILE="$OUT_DIR/question.txt"
  OUTPUT_FILE="$OUT_DIR/answer.jsonl"
  LOG_FILE="$LOG_DIR/conclusion_chosen_sft_$RUN_ID.log"
  printf '%s\n' "$QUESTION" > "$QUESTION_FILE"

  RUN_CMD=(
    "$PYTHON_BIN" scripts/run_cpu_inference.py
    "${COMMON_ARGS[@]}"
    --questions-file "$QUESTION_FILE"
    --batch-output "$OUTPUT_FILE"
    "$@"
  )
  if [[ "${STREAM_LOG:-0}" == "1" ]]; then
    "${RUN_CMD[@]}" 2>&1 | tee "$LOG_FILE"
  else
    "${RUN_CMD[@]}" >"$LOG_FILE" 2>&1
  fi

  if [[ "${ANSWER_ONLY:-0}" == "1" ]]; then
    "$PYTHON_BIN" - "$OUTPUT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    row = json.loads(handle.readline())
print(row.get("answer") or "")
if row.get("error"):
    print(f"[error] {row['error']}", file=sys.stderr)
PY
  else
    "$PYTHON_BIN" - "$OUTPUT_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    row = json.loads(handle.readline())
print(json.dumps(row, ensure_ascii=False, indent=2))
PY
  fi
  echo "[saved] output=$OUTPUT_FILE log=$LOG_FILE" >&2
  exit 0
fi

exec "$PYTHON_BIN" scripts/run_cpu_inference.py \
  "${COMMON_ARGS[@]}" \
  "${ANSWER_ARGS[@]}" \
  "$@"
