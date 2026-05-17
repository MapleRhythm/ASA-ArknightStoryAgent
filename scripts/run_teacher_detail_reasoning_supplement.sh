#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/sft_teacher_prompt_supplement_merged_v1.json}"
BASE_DIR="${BASE_DIR:-data/processed/sft_data/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/sft_data/tool_detail_reasoning_supplement_teacher_v1}"
MERGED_OUTPUT_DIR="${MERGED_OUTPUT_DIR:-data/processed/sft_data/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1}"
LLAMA_FACTORY_OUTPUT_DIR="${LLAMA_FACTORY_OUTPUT_DIR:-data/processed/llama_factory/teacher_v2_plus_prompt_supplement_merged_v1_run6_fixed_plus_detail_reasoning_teacher_v1}"
TARGET_TOTAL="${TARGET_TOTAL:-300}"
SAMPLES_PER_REQUEST="${SAMPLES_PER_REQUEST:-3}"
MAX_REQUESTS="${MAX_REQUESTS:-160}"
BADCASE_CANDIDATE_LIMIT="${BADCASE_CANDIDATE_LIMIT:-240}"
CONCURRENCY="${CONCURRENCY:-6}"
API_KEY_ENV="${API_KEY_ENV:-TEACHER_API_KEY}"
API_BASE="${API_BASE:-https://api.svips.org}"
ENDPOINT_PATH="${ENDPOINT_PATH:-/v1/chat/completions}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

if [[ -z "${!API_KEY_ENV:-}" ]]; then
  echo "Missing teacher API key env var: ${API_KEY_ENV}" >&2
  echo "Set it first, for example:" >&2
  echo "  export ${API_KEY_ENV}=<your_key>" >&2
  exit 2
fi

if [[ "$OVERWRITE_OUTPUT_DIR" == "true" ]]; then
  rm -rf "$OUTPUT_DIR" "$MERGED_OUTPUT_DIR" "$LLAMA_FACTORY_OUTPUT_DIR"
fi

exec "$PYTHON_BIN" scripts/generate_tool_detail_reasoning_supplement_from_teacher.py \
  --config "$CONFIG_PATH" \
  --base-dir "$BASE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --merged-output-dir "$MERGED_OUTPUT_DIR" \
  --llama-factory-output-dir "$LLAMA_FACTORY_OUTPUT_DIR" \
  --target-total "$TARGET_TOTAL" \
  --samples-per-request "$SAMPLES_PER_REQUEST" \
  --max-requests "$MAX_REQUESTS" \
  --badcase-candidate-limit "$BADCASE_CANDIDATE_LIMIT" \
  --concurrency "$CONCURRENCY" \
  --api-key-env "$API_KEY_ENV" \
  --api-base "$API_BASE" \
  --endpoint-path "$ENDPOINT_PATH" \
  --export-llama-factory
