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

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running API verifier.}"

INPUT_DIR="${SODA_INPUT_DIR:-data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel}"
OUTPUT_DIR="${SODA_VERIFIER_OUT_DIR:-data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1}"
MAX_VERIFIER_PROMPTS="${SODA_MAX_VERIFIER_PROMPTS:-}"
RUN_TEACHER_FULL_CHAIN="${SODA_RUN_TEACHER_FULL_CHAIN:-0}"
MAX_TEACHER_FULL_CHAIN_QUESTIONS="${SODA_MAX_TEACHER_FULL_CHAIN_QUESTIONS:-}"
KEEP_UNVERIFIED_CONCLUSION="${SODA_KEEP_UNVERIFIED_CONCLUSION:-1}"
SAVE_API_REQUEST_LOGS="${SODA_SAVE_API_REQUEST_LOGS:-0}"
OVERWRITE_VERIFIER="${SODA_OVERWRITE_VERIFIER:-0}"
OVERWRITE_TEACHER_FULL_CHAIN="${SODA_OVERWRITE_TEACHER_FULL_CHAIN:-0}"
LOG_PATH="${SODA_VERIFIER_LOG:-logs/soda_api_verifier_v1.log}"

cd "$ROOT_DIR"
mkdir -p logs

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$INPUT_DIR/audit_records.jsonl" ]]; then
  echo "[error] Missing input audit records: $INPUT_DIR/audit_records.jsonl" >&2
  exit 2
fi

cmd=(
  "$PYTHON_BIN" scripts/build_soda_api_verifier_dataset.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$MAX_VERIFIER_PROMPTS" ]]; then
  cmd+=(--max-verifier-prompts "$MAX_VERIFIER_PROMPTS")
fi
if [[ "$KEEP_UNVERIFIED_CONCLUSION" == "1" ]]; then
  cmd+=(--keep-unverified-conclusion)
fi
if [[ "$SAVE_API_REQUEST_LOGS" == "1" ]]; then
  cmd+=(--save-api-request-logs)
fi
if [[ "$OVERWRITE_VERIFIER" == "1" ]]; then
  cmd+=(--overwrite-verifier)
fi
if [[ "$RUN_TEACHER_FULL_CHAIN" == "1" ]]; then
  cmd+=(--run-teacher-full-chain)
fi
if [[ "$OVERWRITE_TEACHER_FULL_CHAIN" == "1" ]]; then
  cmd+=(--overwrite-teacher-full-chain)
fi
if [[ -n "$MAX_TEACHER_FULL_CHAIN_QUESTIONS" ]]; then
  cmd+=(--max-teacher-full-chain-questions "$MAX_TEACHER_FULL_CHAIN_QUESTIONS")
fi

"${cmd[@]}" 2>&1 | tee "$LOG_PATH"
