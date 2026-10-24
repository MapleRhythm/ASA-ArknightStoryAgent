#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
PY=${REMOTE_STORE}/asa_train/env/train/bin/python
SOURCE_INDEX=$BASE/qwen3_embedding_0.6b_1024_sparse_v2
MODEL_4B=$BASE/Qwen3-Embedding-4B
OUTPUT_INDEX=$BASE/qwen3_embedding_4b_1024_sparse_v2
SCRIPT=$BASE/scripts/build_dense_sidecar_index.py
LOG_DIR=$BASE/logs/qwen3_embedding_4b_ablation_20260825

mkdir -p "$LOG_DIR" "$OUTPUT_INDEX"
exec > >(tee -a "$LOG_DIR/build_4b_index.log") 2>&1

echo "[$(date '+%F %T')] 248 build start"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

build_index() {
  local batch_size=$1
  local resume_offset
  resume_offset=$(grep -o 'encoded [0-9]*/[0-9]*' "$LOG_DIR/build_4b_index.log" 2>/dev/null \
    | tail -n 1 | sed -E 's/encoded ([0-9]+)\/.*/\1/' || true)
  resume_offset=${resume_offset:-0}
  echo "[$(date '+%F %T')] batch=$batch_size resume_offset=$resume_offset"
  CUDA_VISIBLE_DEVICES=0 ASA_DENSE_RESUME_OFFSET="$resume_offset" "$PY" "$SCRIPT" \
    --source-index "$SOURCE_INDEX" \
    --output-dir "$OUTPUT_INDEX" \
    --embedding-model "$MODEL_4B" \
    --device cuda --batch-size "$batch_size" \
    --embedding-dim 1024 --max-seq-length 1024
}

if [[ ! -s "$OUTPUT_INDEX/faiss.index" ]]; then
  if ! build_index 16; then
    if [[ ! -s "$OUTPUT_INDEX/faiss.index" ]] && ! build_index 8; then
      [[ -s "$OUTPUT_INDEX/faiss.index" ]] || build_index 4
    fi
  fi
fi

test -s "$OUTPUT_INDEX/faiss.index"
echo "[$(date '+%F %T')] 248 build complete"
du -sh "$OUTPUT_INDEX"
