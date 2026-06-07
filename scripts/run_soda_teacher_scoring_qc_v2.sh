#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  : "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running teacher scoring.}"
fi

INPUT_DIR="${INPUT_DIR:-data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored}"
DATASET_NAME="${DATASET_NAME:-soda_mix_eval50_clean_extra300_v1_qc_v2_teacher_scored}"
MODEL="${MODEL:-deepseek-chat}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MAX_EVIDENCE_CHARS="${MAX_EVIDENCE_CHARS:-3000}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-12000}"
MIN_MARGIN="${MIN_MARGIN:-1.5}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.65}"
MIN_BEST_SCORE="${MIN_BEST_SCORE:-3.5}"
MIN_SFT_SCORE="${MIN_SFT_SCORE:-4.0}"

mkdir -p logs

args=(
  scripts/score_soda_kto_pairs_with_teacher.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --dataset-name "$DATASET_NAME"
  --batch-size "$BATCH_SIZE"
  --max-evidence-chars "$MAX_EVIDENCE_CHARS"
  --api-key-env DEEPSEEK_API_KEY
  --api-base "${API_BASE:-https://api.deepseek.com}"
  --model "$MODEL"
  --temperature "${TEMPERATURE:-0}"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --min-margin "$MIN_MARGIN"
  --min-confidence "$MIN_CONFIDENCE"
  --min-best-score "$MIN_BEST_SCORE"
  --min-sft-score "$MIN_SFT_SCORE"
)

if [[ -n "${TASK_TYPE:-}" ]]; then
  args+=(--task-type "$TASK_TYPE")
fi
if [[ -n "${LIMIT_GROUPS:-}" ]]; then
  args+=(--limit-groups "$LIMIT_GROUPS")
fi
if [[ -n "${LIMIT_BATCHES:-}" ]]; then
  args+=(--limit-batches "$LIMIT_BATCHES")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

python "${args[@]}" 2>&1 | tee "logs/soda_teacher_scoring_qc_v2.log"
