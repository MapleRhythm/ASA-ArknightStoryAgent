#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
else
  PYTHON_BIN="python"
fi

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

EXTRA_LIMIT="${SODA_EXTRA_LIMIT:-300}"
RUN_ID="${SODA_EXTRA_RUN_ID:-extra300_v1_soda_lora}"
QUESTIONS_FILE="${SODA_EXTRA_QUESTIONS_FILE:-data/processed/soda_extra_hard_questions_v1.jsonl}"
CLEAN_DIR="${SODA_EXTRA_CLEAN_DIR:-data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_clean_v1}"
EXTRA_MERGED_DIR="${SODA_EXTRA_MERGED_DIR:-data/processed/llama_factory/soda_eval50_len1800_api_verifier_${RUN_ID}_merged}"
FINAL_DIR="${SODA_EXTRA_FINAL_DIR:-data/processed/llama_factory/soda_mix_eval50_clean_extra300_v1}"
FINAL_DATASET_NAME="${SODA_EXTRA_FINAL_DATASET_NAME:-soda_mix_eval50_clean_extra300_v1}"
STUDENT_LORA_PATH="${SODA_EXTRA_STUDENT_LORA_PATH:-model/lora/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3}"

mkdir -p logs outputs/soda_flow_reports

echo "[stage] clean current eval50 verifier dataset"
"$PYTHON_BIN" scripts/clean_soda_api_verifier_dataset.py \
  --output-dir "$CLEAN_DIR" \
  --dataset-name "$(basename "$CLEAN_DIR")"

echo "[stage] build extra question pool"
"$PYTHON_BIN" scripts/build_soda_extra_question_pool.py \
  --output "$QUESTIONS_FILE" \
  --limit "$EXTRA_LIMIT"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[stop] DEEPSEEK_API_KEY is not set. Export it in your shell, then rerun this script." >&2
  echo "[ready] clean_dir=$CLEAN_DIR" >&2
  echo "[ready] questions_file=$QUESTIONS_FILE" >&2
  exit 4
fi

echo "[stage] 3gpu student rollout + teacher replay + evidence-only verifier"
SODA_PARALLEL_QUESTIONS_FILE="$QUESTIONS_FILE" \
SODA_PARALLEL_LISTWISE_FILE="${SODA_EXTRA_LISTWISE_FILE:-data/processed/soda_550_questions_listwise.jsonl}" \
SODA_PARALLEL_RUN_ID="$RUN_ID" \
SODA_PARALLEL_GPUS="${SODA_EXTRA_GPUS:-0,1,2}" \
SODA_PARALLEL_RUN_TEACHER_FULL_CHAIN="${SODA_EXTRA_RUN_TEACHER_FULL_CHAIN:-0}" \
SODA_PARALLEL_GPU_MEMORY_UTILIZATION="${SODA_EXTRA_GPU_MEMORY_UTILIZATION:-0.50}" \
SODA_PARALLEL_DATASET_NAME="soda_eval50_len1800_api_verifier_${RUN_ID}" \
SODA_PARALLEL_VERIFIER_BASE="data/processed/llama_factory/soda_eval50_len1800_api_verifier_${RUN_ID}" \
SODA_PARALLEL_ROLLOUT_BASE="data/processed/llama_factory/soda_eval50_len1800_blackbox_${RUN_ID}" \
SODA_FLOW_LORA_PATH="$STUDENT_LORA_PATH" \
bash scripts/run_soda_eval50_len1800_api_verifier_flow_3gpu.sh

echo "[stage] merge clean eval50 + extra verifier dataset"
"$PYTHON_BIN" scripts/merge_soda_api_verifier_shards.py \
  --input-dir "$CLEAN_DIR" \
  --input-dir "$EXTRA_MERGED_DIR" \
  --output-dir "$FINAL_DIR" \
  --dataset-name "$FINAL_DATASET_NAME" \
  --seed "${SODA_EXTRA_SEED:-20260601}" \
  --val-ratio "${SODA_EXTRA_VAL_RATIO:-0.08}" \
  --overwrite

"$PYTHON_BIN" scripts/analyze_soda_api_verifier_dataset.py \
  --dataset-dir "$FINAL_DIR" \
  --output "outputs/soda_flow_reports/${FINAL_DATASET_NAME}_audit.md"

echo "[done] clean_dir=$CLEAN_DIR"
echo "[done] extra_merged_dir=$EXTRA_MERGED_DIR"
echo "[done] final_dir=$FINAL_DIR"
