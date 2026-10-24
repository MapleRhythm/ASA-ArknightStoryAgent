#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
REPO=${REMOTE_STORE}/ASA-ArknightStoryAgent
PY=${REMOTE_STORE}/asa_train/env/train/bin/python
INDEX=$BASE/qwen3_embedding_4b_1024_sparse_v2
EMBED=$BASE/Qwen3-Embedding-4B
RERANK=${REMOTE_HOME}/asa_listwise_v2_model
SCRIPT=$BASE/scripts/evaluate_retrieval_recall.py
OUT=$BASE/qwen3_embedding_4b_ablation_20260825/rerank_fusion

run_one() {
  local data=$1 name=$2 batch=$3
  local output=$OUT/${name}.json
  local log=$OUT/${name}.log
  [[ -s "$output" ]] && { echo "skip existing $output"; return; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/src" "$PY" "$SCRIPT" \
    --listwise "$data" --output "$output" --device cuda \
    --index-dir "$INDEX" --embedding-model "$EMBED" --reranker-model "$RERANK" \
    --top-ks 1,5,10,20 --dense-top-k 120 --sparse-top-k 120 \
    --dense-weight 1.25 --sparse-weight 1.0 \
    --dense-min-quota 40 --sparse-min-quota 80 \
    --fusion-top-k 120 --reranker-candidate-top-k 120 \
    --minirag-top-k 0 --minirag-weight 0 --rerank-batch-size "$batch" \
    --include-records --hit-mode segment --segment-coverage-threshold 0.9 \
    --tag "$name" >"$log" 2>&1
}

baseline_pid=1437575
while kill -0 "$baseline_pid" 2>/dev/null; do sleep 10; done

run_one ${REMOTE_HOME}/asa_hard59_listwise.jsonl hard59_tuned_dw125_q40_80_b16 16
run_one ${REMOTE_HOME}/asa_easy200_listwise.jsonl easy200_tuned_dw125_q40_80_b16 16
echo "[$(date '+%F %T')] tuned sequential rerank complete"
