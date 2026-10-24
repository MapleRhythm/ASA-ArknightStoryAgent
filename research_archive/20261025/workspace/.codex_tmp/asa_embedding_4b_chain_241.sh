#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
REPO=${REMOTE_HOME}/ASA-ArknightStoryAgent
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
SOURCE_INDEX=$BASE/indexes/qwen3_embedding_0.6b_1024_sparse_v2
INDEX_4B=$BASE/indexes/qwen3_embedding_4b_1024_sparse_v2
MODEL_06=$BASE/models/Qwen3-Embedding-0.6B
MODEL_4B=$BASE/models/Qwen3-Embedding-4B
OUT_DIR=$BASE/runs/qwen3_embedding_4b_ablation_20260825
GPU_LOCK=$BASE/locks/gpu0_asa_experiments.lock
DISABLED_RERANKER=$BASE/models/__disabled_reranker__

mkdir -p "$OUT_DIR" "$BASE/locks" "$BASE/logs"

wait_gpu0_idle() {
  while true; do
    IFS=, read -r memory_used utilization < <(
      nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    memory_used=${memory_used// /}
    utilization=${utilization// /}
    if [[ "$memory_used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
      && (( memory_used < 1000 && utilization < 10 )); then
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for GPU0: memory=${memory_used}MiB util=${utilization}%"
    sleep 30
  done
}

if [[ ! -s "$INDEX_4B/faiss.index" ]]; then
  exec 9>"$GPU_LOCK"
  flock 9
  wait_gpu0_idle
  cd "$REPO"
  if ! CUDA_VISIBLE_DEVICES=0 GOLDENGLOW_USE_TRAIN_OVERRIDE=1 "$PY" \
    "$BASE/scripts/build_dense_sidecar_index.py" \
    --source-index "$SOURCE_INDEX" \
    --output-dir "$INDEX_4B" \
    --embedding-model "$MODEL_4B" \
    --device cuda --batch-size 8 \
    --embedding-dim 1024 --max-seq-length 1024; then
    if [[ -s "$INDEX_4B/faiss.index" ]]; then
      echo "index appeared despite nonzero builder exit"
    else
      echo "batch-size 8 failed; retrying batch-size 4" >&2
      CUDA_VISIBLE_DEVICES=0 GOLDENGLOW_USE_TRAIN_OVERRIDE=1 "$PY" \
        "$BASE/scripts/build_dense_sidecar_index.py" \
        --source-index "$SOURCE_INDEX" \
        --output-dir "$INDEX_4B" \
        --embedding-model "$MODEL_4B" \
        --device cuda --batch-size 4 \
        --embedding-dim 1024 --max-seq-length 1024
    fi
  fi
  flock -u 9
fi

[[ -s "$INDEX_4B/faiss.index" ]] || { echo "4B FAISS index missing" >&2; exit 3; }

run_recall() {
  local dataset=$1
  local index_dir=$2
  local embedding_model=$3
  local output=$4
  local tag=$5
  if [[ -s "$output" ]]; then
    echo "skip existing result: $output"
    return 0
  fi
  exec 9>"$GPU_LOCK"
  flock 9
  wait_gpu0_idle
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0 GOLDENGLOW_USE_TRAIN_OVERRIDE=1 "$PY" \
    scripts/evaluate_retrieval_recall.py \
    --listwise "$dataset" --output "$output" --device cuda \
    --index-dir "$index_dir" --embedding-model "$embedding_model" \
    --reranker-model "$DISABLED_RERANKER" \
    --top-ks 1,5,10,20,50,120 \
    --dense-top-k 120 --sparse-top-k 120 \
    --dense-weight 0.5 --sparse-weight 1.0 \
    --dense-min-quota 20 --sparse-min-quota 100 \
    --fusion-top-k 120 --reranker-candidate-top-k 120 \
    --minirag-top-k 0 --minirag-weight 0 \
    --skip-rerank --oracle-sources --include-records \
    --hit-mode segment --segment-coverage-threshold 0.9 \
    --tag "$tag"
  flock -u 9
  sleep 5
}

HARD=$REPO/data/processed/evidence_chain_reranker/failure_hard_pool_v1/failure_hard_pool_listwise.jsonl
EASY=$REPO/data/processed/evidence_chain_reranker/failure_hard_pool_v1/sample200_listwise.jsonl

run_recall "$HARD" "$SOURCE_INDEX" "$MODEL_06" \
  "$OUT_DIR/hard59_qwen3_0.6b_segment.json" hard59_qwen3_0.6b_1024_segment
run_recall "$HARD" "$INDEX_4B" "$MODEL_4B" \
  "$OUT_DIR/hard59_qwen3_4b_segment.json" hard59_qwen3_4b_1024_segment
run_recall "$EASY" "$SOURCE_INDEX" "$MODEL_06" \
  "$OUT_DIR/easy200_qwen3_0.6b_segment.json" easy200_qwen3_0.6b_1024_segment
run_recall "$EASY" "$INDEX_4B" "$MODEL_4B" \
  "$OUT_DIR/easy200_qwen3_4b_segment.json" easy200_qwen3_4b_1024_segment

echo "[$(date '+%F %T')] embedding 4B ablation complete"
