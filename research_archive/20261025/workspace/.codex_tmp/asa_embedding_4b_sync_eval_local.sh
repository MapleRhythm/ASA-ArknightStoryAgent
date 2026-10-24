#!/usr/bin/env bash
set -euo pipefail

SSH_ASKPASS=${ASA_ROOT}/.codex_tmp/ssh_askpass_asa.sh
export SSH_ASKPASS SSH_ASKPASS_REQUIRE=force DISPLAY=:0

REMOTE_BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror
REMOTE_LOG=$REMOTE_BASE/logs/qwen3_embedding_4b_ablation_20260825
REMOTE_ABLATION_SCRIPT=$REMOTE_BASE/scripts/asa_embedding_4b_ablation_248.sh
PRIMARY_BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
PRIMARY_INDEX=$PRIMARY_BASE/indexes/qwen3_embedding_4b_1024_sparse_v2
LOG=${ASA_ROOT}/.codex_tmp/asa_embedding_4b_sync_eval_local.log

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] waiting for 248 4B index"
while setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
  ${REMOTE_USER}@${REMOTE_HOST} "pid=\$(cat '$REMOTE_LOG/build.pid' 2>/dev/null || true); if test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null; then echo running; else echo done; fi" \
  | grep -q '^running$'; do
  sleep 60
done

setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "test -s '$REMOTE_BASE/qwen3_embedding_4b_1024_sparse_v2/faiss.index' && \
   test -s '$REMOTE_BASE/qwen3_embedding_4b_1024_sparse_v2/index_meta.json'"

setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "nohup '$REMOTE_ABLATION_SCRIPT' \
     >'$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/launcher.log' 2>&1 </dev/null & \
   echo \$! >'$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/ablation.pid'"

while setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
  ${REMOTE_USER}@${REMOTE_HOST} "pid=\$(cat '$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/ablation.pid' 2>/dev/null || true); if test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null; then echo running; else echo done; fi" \
  | grep -q '^running$'; do
  sleep 60
done
setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "test -s '$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/hard59_qwen3_0.6b_segment.json' && \
   test -s '$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/hard59_qwen3_4b_segment.json' && \
   test -s '$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/easy200_qwen3_0.6b_segment.json' && \
   test -s '$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/easy200_qwen3_4b_segment.json'"

PRIMARY_OUT=$PRIMARY_BASE/runs/qwen3_embedding_4b_ablation_20260825
setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "mkdir -p '$PRIMARY_OUT'"
for name in hard59_qwen3_0.6b_segment.json hard59_qwen3_4b_segment.json \
  easy200_qwen3_0.6b_segment.json easy200_qwen3_4b_segment.json; do
  tmp=$(mktemp)
  setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    ${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_BASE/qwen3_embedding_4b_ablation_20260825/$name "$tmp"
  setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$tmp" \
    ${REMOTE_USER}@${REMOTE_HOST}:$PRIMARY_OUT/$name.tmp
  setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
    "mv '$PRIMARY_OUT/$name.tmp' '$PRIMARY_OUT/$name'"
  rm -f "$tmp"
done

setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "mkdir -p '$PRIMARY_INDEX'"
for name in faiss.index index_meta.json; do
  tmp=$(mktemp)
  setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
    ${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_BASE/qwen3_embedding_4b_1024_sparse_v2/$name "$tmp"
  setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$tmp" \
    ${REMOTE_USER}@${REMOTE_HOST}:$PRIMARY_INDEX/$name.tmp
  setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
    "mv '$PRIMARY_INDEX/$name.tmp' '$PRIMARY_INDEX/$name'"
  rm -f "$tmp"
done

setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} 'bash -s' <<'REMOTE'
set -euo pipefail
BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
SRC=$BASE/indexes/qwen3_embedding_0.6b_1024_sparse_v2
DST=$BASE/indexes/qwen3_embedding_4b_1024_sparse_v2
for name in documents.jsonl bm25_tokens.pkl sparse_index.pkl operator_aliases.json; do
  test -e "$DST/$name" || ln -s "$SRC/$name" "$DST/$name"
done
test -s "$DST/faiss.index"
REMOTE

echo "[$(date '+%F %T')] synced 4B index and all 248 recall results to 241"
