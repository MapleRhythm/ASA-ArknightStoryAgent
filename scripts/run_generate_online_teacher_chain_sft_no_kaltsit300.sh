#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/llama_factory/teacher_online_chain_short_prompt_v1_ds_flash_no_kaltsit300_v1}"
API_BASE="${API_BASE:-https://api.deepseek.com}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_API_KEY}"
MODEL="${MODEL:-deepseek-v4-flash}"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing API key env: $API_KEY_ENV" >&2
  echo "Example: export $API_KEY_ENV=sk-..." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"$PYTHON_BIN" scripts/generate_online_teacher_chain_sft.py \
  --teacher-config configs/sft_teacher_prompt_supplement_merged_v1.json \
  --api-type chat_completions \
  --api-base "$API_BASE" \
  --api-key-env "$API_KEY_ENV" \
  --model "$MODEL" \
  --question-source teacher_complex \
  --max-questions 300 \
  --teacher-complex-questions 300 \
  --teacher-complex-questions-per-request 4 \
  --output-dir "$OUTPUT_DIR" \
  --parallel 4 \
  --parallel-retrieval \
  --device cuda:0 \
  --max-rounds 3 \
  --validation-retries 2 \
  --api-retries 3 \
  --retry-sleep 30 \
  --grounding-mode soft \
  --resume
