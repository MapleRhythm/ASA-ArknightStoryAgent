#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
REPO=${REMOTE_STORE}/ASA-ArknightStoryAgent
PY=${REMOTE_STORE}/asa_train/env/train/bin/python
SOURCE_INDEX=$BASE/qwen3_embedding_0.6b_1024_sparse_v2
INDEX_4B=$BASE/qwen3_embedding_4b_1024_sparse_v2
MODEL_06=$BASE/Qwen3-Embedding-0.6B
MODEL_4B=$BASE/Qwen3-Embedding-4B
OUT=$BASE/qwen3_embedding_4b_ablation_20260825
HARD=$BASE/ablation_inputs/failure_hard_pool_listwise.jsonl
EASY=$BASE/ablation_inputs/sample200_listwise.jsonl
DISABLED=$BASE/__disabled_reranker__

mkdir -p "$OUT"
exec > >(tee -a "$OUT/ablation_248.log") 2>&1

run_recall() {
  local dataset=$1 index=$2 model=$3 output=$4 tag=$5
  [[ -s "$output" ]] && { echo "skip existing $output"; return; }
  local cache=$OUT/cache_${tag}
  mkdir -p "$cache"
  CUDA_VISIBLE_DEVICES=0 XDG_CACHE_HOME="$cache" HF_HOME="$cache/huggingface" \
    TORCH_HOME="$cache/torch" PYTHONPATH="$REPO/src" "$PY" \
    "$BASE/scripts/evaluate_retrieval_recall.py" \
    --listwise "$dataset" --output "$output" --device cuda \
    --index-dir "$index" --embedding-model "$model" \
    --reranker-model "$DISABLED" \
    --top-ks 1,5,10,20,50,120 \
    --dense-top-k 120 --sparse-top-k 120 \
    --dense-weight 0.5 --sparse-weight 1.0 \
    --dense-min-quota 20 --sparse-min-quota 100 \
    --fusion-top-k 120 --reranker-candidate-top-k 120 \
    --minirag-top-k 0 --minirag-weight 0 \
    --skip-rerank --oracle-sources --include-records \
    --hit-mode segment --segment-coverage-threshold 0.9 \
    --tag "$tag"
  rm -rf "$cache"
}

test -s "$INDEX_4B/faiss.index"
run_recall "$HARD" "$SOURCE_INDEX" "$MODEL_06" \
  "$OUT/hard59_qwen3_0.6b_segment.json" hard59_qwen3_0.6b_1024_segment
run_recall "$HARD" "$INDEX_4B" "$MODEL_4B" \
  "$OUT/hard59_qwen3_4b_segment.json" hard59_qwen3_4b_1024_segment
run_recall "$EASY" "$SOURCE_INDEX" "$MODEL_06" \
  "$OUT/easy200_qwen3_0.6b_segment.json" easy200_qwen3_0.6b_1024_segment
run_recall "$EASY" "$INDEX_4B" "$MODEL_4B" \
  "$OUT/easy200_qwen3_4b_segment.json" easy200_qwen3_4b_1024_segment
echo "[$(date '+%F %T')] 248 ablation complete"
