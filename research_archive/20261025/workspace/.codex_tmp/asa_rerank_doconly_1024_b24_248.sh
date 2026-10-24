#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
OUT="$BASE/qwen3_embedding_4b_ablation_20260825/rerank_fusion"
PY=${REMOTE_STORE}/asa_train/env/train/bin/python
EVAL=${REMOTE_HOME}/asa_evaluate_retrieval_recall_doconly_248.py
INDEX="$BASE/qwen3_embedding_4b_1024_sparse_v2"
EMBED="$BASE/Qwen3-Embedding-4B"
RERANK=${REMOTE_HOME}/asa_listwise_v2_model
export PYTHONPATH=${REMOTE_STORE}/ASA-ArknightStoryAgent/src

run_eval() {
  local input=$1 stem=$2
  "$PY" "$EVAL" --listwise "$input" --output "$OUT/${stem}.json" \
    --device cuda --index-dir "$INDEX" --embedding-model "$EMBED" --reranker-model "$RERANK" \
    --reranker-max-length 1024 --document-only-rerank --top-ks 1,5,10,20 \
    --dense-top-k 120 --sparse-top-k 120 --dense-weight 1.25 --sparse-weight 1.0 \
    --dense-min-quota 40 --sparse-min-quota 80 --fusion-top-k 120 \
    --reranker-candidate-top-k 120 --minirag-top-k 0 --minirag-weight 0 \
    --rerank-batch-size 24 --include-records --hit-mode segment \
    --segment-coverage-threshold 0.9 --tag "$stem" >"$OUT/${stem}.log" 2>&1
}

run_eval ${REMOTE_HOME}/asa_hard59_listwise.jsonl hard59_tuned_dw125_q40_80_doconly_ml1024_b24
run_eval ${REMOTE_HOME}/asa_easy200_listwise.jsonl easy200_tuned_dw125_q40_80_doconly_ml1024_b24
