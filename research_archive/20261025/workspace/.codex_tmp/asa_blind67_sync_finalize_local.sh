#!/usr/bin/env bash
set -euo pipefail

SSH_ASKPASS=${ASA_ROOT}/.codex_tmp/ssh_askpass_asa.sh
REMOTE_BASE=${REMOTE_STORE}/rank_v8_coverage_241_mirror/blind67_retry
PRIMARY_BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
PRIMARY_DIR=$PRIMARY_BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
LOG=${ASA_ROOT}/.codex_tmp/asa_blind67_sync_finalize_local.log

export SSH_ASKPASS SSH_ASKPASS_REQUIRE=force DISPLAY=:0
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] waiting for 248 blind67 retry"

while setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
  ${REMOTE_USER}@${REMOTE_HOST} "pid=\$(cat '$REMOTE_BASE/retry_248.pid' 2>/dev/null || true); if test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null; then echo running; else echo done; fi" \
  | grep -q '^running$'; do
  sleep 30
done

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
  ${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_BASE/blind67_labels.jsonl "$tmp"

count=$(python3 - "$tmp" <<'PY'
import json, sys
keys = set()
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        keys.add((str(row.get("source_name") or ""), str(row.get("query") or "")))
print(len(keys))
PY
)
echo "[$(date '+%F %T')] fetched unique=$count"
if (( count != 67 )); then
  echo "248 blind67 retry incomplete; preserving primary labels" >&2
  exit 2
fi

setsid -w scp -O -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$tmp" \
  ${REMOTE_USER}@${REMOTE_HOST}:$PRIMARY_DIR/blind67_labels.from248.tmp
setsid -w ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ${REMOTE_USER}@${REMOTE_HOST} \
  "mv '$PRIMARY_DIR/blind67_labels.from248.tmp' '$PRIMARY_DIR/blind67_labels.jsonl'; \
   nohup /tmp/asa_blind67_finalize_241.sh \
     >'$PRIMARY_BASE/logs/resume_20260825/blind67_finalize_after_248.log' 2>&1 </dev/null & \
   echo \$! >'$PRIMARY_BASE/logs/resume_20260825/blind67_finalize_after_248.pid'"
echo "[$(date '+%F %T')] synced 67 rows and launched 241 finalizer"
