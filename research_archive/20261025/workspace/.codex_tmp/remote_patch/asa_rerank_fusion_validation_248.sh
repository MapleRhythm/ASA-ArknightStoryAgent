#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
REPO=${REMOTE_STORE}/ASA-ArknightStoryAgent
PY=$BASE/runtime_env/bin/python
[[ -x "$PY" ]] || PY=${REMOTE_STORE}/asa_train/env/train/bin/python
INDEX=$BASE/qwen3_embedding_4b_1024_sparse_v2
EMBED=$BASE/Qwen3-Embedding-4B
RERANK=${REMOTE_HOME}/asa_listwise_v2_model
SCRIPT=$BASE/scripts/evaluate_retrieval_recall.py
OUT=$BASE/qwen3_embedding_4b_ablation_20260825/rerank_fusion
HARD=${REMOTE_HOME}/asa_hard59_listwise.jsonl
EASY=${REMOTE_HOME}/asa_easy200_listwise.jsonl

mkdir -p "$OUT"
test -s "$RERANK/model.safetensors"

run_one() {
  local data=$1 name=$2 dense_weight=$3 dense_quota=$4 sparse_quota=$5
  local output=$OUT/${name}.json
  local log=$OUT/${name}.log
  [[ -s "$output" ]] && { echo "skip existing $output"; return; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO/src" "$PY" "$SCRIPT" \
    --listwise "$data" --output "$output" --device cuda \
    --index-dir "$INDEX" --embedding-model "$EMBED" \
    --reranker-model "$RERANK" \
    --top-ks 1,5,10,20 \
    --dense-top-k 120 --sparse-top-k 120 \
    --dense-weight "$dense_weight" --sparse-weight 1.0 \
    --dense-min-quota "$dense_quota" --sparse-min-quota "$sparse_quota" \
    --fusion-top-k 120 --reranker-candidate-top-k 120 \
    --minirag-top-k 0 --minirag-weight 0 \
    --rerank-batch-size 8 --include-records \
    --hit-mode segment --segment-coverage-threshold 0.9 \
    --tag "$name" >"$log" 2>&1
}

run_pair() {
  local suffix=$1 dense_weight=$2 dense_quota=$3 sparse_quota=$4
  run_one "$HARD" "hard59_${suffix}" "$dense_weight" "$dense_quota" "$sparse_quota" &
  local hard_pid=$!
  run_one "$EASY" "easy200_${suffix}" "$dense_weight" "$dense_quota" "$sparse_quota" &
  local easy_pid=$!
  wait "$hard_pid"
  wait "$easy_pid"
}

run_pair baseline_dw05_q20_100 0.5 20 100
run_pair tuned_dw125_q40_80 1.25 40 80
echo "[$(date '+%F %T')] rerank fusion validation complete"
