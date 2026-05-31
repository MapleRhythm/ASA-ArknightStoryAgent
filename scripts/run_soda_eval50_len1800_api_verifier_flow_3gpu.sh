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

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this flow.}"

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

QUESTIONS_FILE="${SODA_PARALLEL_QUESTIONS_FILE:-data/processed/eval50_recall_questions_for_soda.jsonl}"
LISTWISE_FILE="${SODA_PARALLEL_LISTWISE_FILE:-data/processed/eval50_recall_questions_listwise.jsonl}"
GPUS_CSV="${SODA_PARALLEL_GPUS:-0,1,2}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

RUN_ID="${SODA_PARALLEL_RUN_ID:-v1_noweb_gpu3}"
SHARD_ROOT="${SODA_PARALLEL_SHARD_ROOT:-data/processed/llama_factory/soda_eval50_len1800_shards_${RUN_ID}}"
ROLLOUT_BASE="${SODA_PARALLEL_ROLLOUT_BASE:-data/processed/llama_factory/soda_eval50_len1800_blackbox_${RUN_ID}}"
VERIFIER_BASE="${SODA_PARALLEL_VERIFIER_BASE:-data/processed/llama_factory/soda_eval50_len1800_api_verifier_${RUN_ID}}"
MERGED_ROLLOUT_DIR="${SODA_PARALLEL_MERGED_ROLLOUT_DIR:-${ROLLOUT_BASE}_merged}"
MERGED_VERIFIER_DIR="${SODA_PARALLEL_MERGED_VERIFIER_DIR:-${VERIFIER_BASE}_merged}"
DATASET_NAME="${SODA_PARALLEL_DATASET_NAME:-soda_eval50_len1800_api_verifier_${RUN_ID}}"
MAX_ROUNDS="${SODA_PARALLEL_MAX_ROUNDS:-2}"
GPU_MEMORY_UTILIZATION="${SODA_PARALLEL_GPU_MEMORY_UTILIZATION:-0.62}"
RUN_TEACHER_FULL_CHAIN="${SODA_PARALLEL_RUN_TEACHER_FULL_CHAIN:-1}"
TEACHER_FULL_CHAIN_LIMIT="${SODA_PARALLEL_TEACHER_FULL_CHAIN_LIMIT:-}"
VAL_RATIO="${SODA_PARALLEL_VAL_RATIO:-0.08}"
SEED="${SODA_PARALLEL_SEED:-20260531}"

if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "[error] No GPUs configured in SODA_PARALLEL_GPUS" >&2
  exit 2
fi
if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "[error] Missing questions file: $QUESTIONS_FILE" >&2
  exit 2
fi

mkdir -p "$SHARD_ROOT" logs outputs/soda_flow_reports

"$PYTHON_BIN" - "$QUESTIONS_FILE" "$SHARD_ROOT" "${GPUS[@]}" <<'PY'
import sys
from pathlib import Path

questions_file = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
gpus = sys.argv[3:]
lines = [line for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()]
handles = {}
try:
    for gpu in gpus:
        path = shard_root / f"questions_gpu{gpu}.jsonl"
        handles[gpu] = path.open("w", encoding="utf-8")
    for index, line in enumerate(lines):
        gpu = gpus[index % len(gpus)]
        handles[gpu].write(line + "\n")
finally:
    for handle in handles.values():
        handle.close()
for gpu in gpus:
    path = shard_root / f"questions_gpu{gpu}.jsonl"
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[shard] gpu={gpu} questions={count} file={path}", flush=True)
PY

declare -a PIDS=()
declare -a SHARD_ROLLOUT_DIRS=()
declare -a SHARD_VERIFIER_DIRS=()

for gpu in "${GPUS[@]}"; do
  shard_questions="$SHARD_ROOT/questions_gpu${gpu}.jsonl"
  rollout_dir="${ROLLOUT_BASE}_gpu${gpu}"
  verifier_dir="${VERIFIER_BASE}_gpu${gpu}"
  SHARD_ROLLOUT_DIRS+=("$rollout_dir")
  SHARD_VERIFIER_DIRS+=("$verifier_dir")
  (
    export SODA_FLOW_QUESTIONS_FILE="$shard_questions"
    export SODA_FLOW_LISTWISE_FILE="$LISTWISE_FILE"
    export SODA_FLOW_ROLLOUT_DIR="$rollout_dir"
    export SODA_FLOW_VERIFIER_DIR="$verifier_dir"
    export SODA_FLOW_GEN_CUDA_VISIBLE_DEVICES="$gpu"
    export SODA_FLOW_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
    export SODA_FLOW_MAX_ROUNDS="$MAX_ROUNDS"
    export SODA_FLOW_LIMIT=
    export SODA_FLOW_VERIFIER_LIMIT=
    export SODA_FLOW_DISABLE_WEB_CONTEXT=1
    export SODA_FLOW_RUN_TEACHER_FULL_CHAIN="$RUN_TEACHER_FULL_CHAIN"
    export SODA_FLOW_TEACHER_FULL_CHAIN_LIMIT="$TEACHER_FULL_CHAIN_LIMIT"
    export SODA_FLOW_LOG_PREFIX="soda_eval50_len1800_${RUN_ID}_gpu${gpu}"
    export SODA_FLOW_REPORT_PREFIX="eval50_len1800_${RUN_ID}_gpu${gpu}"
    bash scripts/run_soda_eval50_len1800_api_verifier_flow.sh
  ) > "logs/soda_eval50_len1800_${RUN_ID}_gpu${gpu}.wrapper.log" 2>&1 &
  PIDS+=("$!")
  echo "[launched] gpu=$gpu pid=${PIDS[-1]} questions=$shard_questions"
done

failed=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  gpu="${GPUS[$index]}"
  if wait "$pid"; then
    echo "[shard-done] gpu=$gpu"
  else
    status=$?
    echo "[shard-failed] gpu=$gpu status=$status log=logs/soda_eval50_len1800_${RUN_ID}_gpu${gpu}.wrapper.log" >&2
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "[error] one or more SODA shards failed; inspect logs before merge." >&2
  exit 3
fi

rollout_merge_cmd=("$PYTHON_BIN" scripts/merge_soda_blackbox_shards.py --output-dir "$MERGED_ROLLOUT_DIR" --seed "$SEED" --val-ratio "$VAL_RATIO" --overwrite)
for dir in "${SHARD_ROLLOUT_DIRS[@]}"; do
  rollout_merge_cmd+=(--input-dir "$dir")
done
"${rollout_merge_cmd[@]}"

verifier_merge_cmd=(
  "$PYTHON_BIN" scripts/merge_soda_api_verifier_shards.py
  --output-dir "$MERGED_VERIFIER_DIR"
  --dataset-name "$DATASET_NAME"
  --seed "$SEED"
  --val-ratio "$VAL_RATIO"
  --overwrite
)
for dir in "${SHARD_VERIFIER_DIRS[@]}"; do
  verifier_merge_cmd+=(--input-dir "$dir")
done
"${verifier_merge_cmd[@]}"

"$PYTHON_BIN" scripts/analyze_soda_api_verifier_dataset.py \
  --dataset-dir "$MERGED_VERIFIER_DIR" \
  --output "outputs/soda_flow_reports/eval50_len1800_${RUN_ID}_merged_api_verifier_audit.md"

"$PYTHON_BIN" scripts/analyze_soda_gold_evidence_topk.py \
  --audit-records "$MERGED_ROLLOUT_DIR/audit_records.jsonl" \
  --output "outputs/soda_flow_reports/eval50_len1800_${RUN_ID}_merged_gold_topk.json"

echo "[done] merged_rollout_dir=$MERGED_ROLLOUT_DIR"
echo "[done] merged_verifier_dir=$MERGED_VERIFIER_DIR"
