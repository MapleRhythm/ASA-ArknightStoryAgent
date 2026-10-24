#!/usr/bin/env bash
set -euo pipefail

MODEL=${REMOTE_HOME}/asa_listwise_v2_model/model.safetensors
EXPECTED_SIZE=2271071852
RSYNC_PIDFILE=${REMOTE_HOME}/asa_listwise_v2_model.rsync.pid
LOG=${REMOTE_HOME}/asa_rerank_resume.log

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] waiting for listwise model"
if [[ -s "$RSYNC_PIDFILE" ]]; then
  pid=$(cat "$RSYNC_PIDFILE")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
  done
fi

actual_size=$(stat -c %s "$MODEL")
if [[ "$actual_size" -ne "$EXPECTED_SIZE" ]]; then
  echo "model size mismatch: actual=$actual_size expected=$EXPECTED_SIZE"
  exit 2
fi

echo "[$(date '+%F %T')] model complete; launching rerank validation"
exec bash ${REMOTE_HOME}/asa_rerank_fusion_validation_248.sh
