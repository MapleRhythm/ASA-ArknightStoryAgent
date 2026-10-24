#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
KEY=$BASE/.sync_248_to_241_ed25519
SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
SCP=(scp -O -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
BLIND=$BASE/blind67_retry
BUILD_LOG=$BASE/logs/qwen3_embedding_4b_ablation_20260825
ABLATION=$BASE/qwen3_embedding_4b_ablation_20260825
PRIMARY=${REMOTE_ASA_ROOT}/rank_v8_coverage
LOG=$BASE/logs/finalize_watcher_248.log

exec > >(tee -a "$LOG") 2>&1

wait_pidfile() {
  local path=$1 label=$2
  while true; do
    pid=$(cat "$path" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[$(date '+%F %T')] waiting $label pid=$pid"
      sleep 30
      continue
    fi
    return 0
  done
}

sync_blind() {
  wait_pidfile "$BLIND/retry_248.pid" blind67
  count=$(${REMOTE_STORE}/asa_train/env/train/bin/python - "$BLIND/blind67_labels.jsonl" <<'PY'
import json, sys
keys = set()
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        keys.add((str(row.get("source_name") or ""), str(row.get("query") or "")))
print(len(keys))
PY
)
  echo "[$(date '+%F %T')] blind67 finished unique=$count"
  (( count == 67 )) || return 2
  pdir=$PRIMARY/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
  "${SCP[@]}" "$BLIND/blind67_labels.jsonl" ${REMOTE_USER}@${REMOTE_HOST}:$pdir/blind67_labels.from248.tmp
  "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} \
    "mv '$pdir/blind67_labels.from248.tmp' '$pdir/blind67_labels.jsonl'; \
     nohup /tmp/asa_blind67_finalize_241.sh \
       >'$PRIMARY/logs/resume_20260825/blind67_finalize_after_248.log' 2>&1 </dev/null & \
     echo \$! >'$PRIMARY/logs/resume_20260825/blind67_finalize_after_248.pid'"
  echo "[$(date '+%F %T')] blind67 synced; 241 finalizer launched"
}

sync_embedding_and_ablation() {
  wait_pidfile "$BUILD_LOG/build.pid" embedding-build
  src=$BASE/qwen3_embedding_4b_1024_sparse_v2
  test -s "$src/faiss.index"
  test -s "$src/index_meta.json"
  pdst=$PRIMARY/indexes/qwen3_embedding_4b_1024_sparse_v2
  "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p '$pdst'"
  for name in faiss.index index_meta.json; do
    "${SCP[@]}" "$src/$name" ${REMOTE_USER}@${REMOTE_HOST}:$pdst/$name.tmp
    "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} "mv '$pdst/$name.tmp' '$pdst/$name'"
  done
  "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} "bash -s" <<'REMOTE'
set -euo pipefail
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
SRC=$BASE/indexes/qwen3_embedding_0.6b_1024_sparse_v2
DST=$BASE/indexes/qwen3_embedding_4b_1024_sparse_v2
for name in documents.jsonl bm25_tokens.pkl sparse_index.pkl operator_aliases.json; do
  test -e "$DST/$name" || ln -s "$SRC/$name" "$DST/$name"
done
REMOTE
  mkdir -p "$ABLATION"
  nohup "$BASE/scripts/asa_embedding_4b_ablation_248.sh" \
    >"$ABLATION/launcher.log" 2>&1 </dev/null &
  echo $! > "$ABLATION/ablation.pid"
  wait_pidfile "$ABLATION/ablation.pid" retrieval-ablation
  for name in hard59_qwen3_0.6b_segment.json hard59_qwen3_4b_segment.json \
    easy200_qwen3_0.6b_segment.json easy200_qwen3_4b_segment.json; do
    test -s "$ABLATION/$name"
  done
  pout=$PRIMARY/runs/qwen3_embedding_4b_ablation_20260825
  "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p '$pout'"
  for name in hard59_qwen3_0.6b_segment.json hard59_qwen3_4b_segment.json \
    easy200_qwen3_0.6b_segment.json easy200_qwen3_4b_segment.json; do
    "${SCP[@]}" "$ABLATION/$name" ${REMOTE_USER}@${REMOTE_HOST}:$pout/$name.tmp
    "${SSH[@]}" ${REMOTE_USER}@${REMOTE_HOST} "mv '$pout/$name.tmp' '$pout/$name'"
  done
  echo "[$(date '+%F %T')] embedding index and four ablation results synced to 241"
}

sync_blind & blind_job=$!
sync_embedding_and_ablation & embedding_job=$!
blind_status=0
embedding_status=0
wait "$blind_job" || blind_status=$?
wait "$embedding_job" || embedding_status=$?
echo "[$(date '+%F %T')] watcher complete blind=$blind_status embedding=$embedding_status"
(( blind_status == 0 && embedding_status == 0 ))
