#!/usr/bin/env bash
set -euo pipefail

MOUNT_ROOT=${REMOTE_STORAGE}
PROJECT_ROOT="$MOUNT_ROOT/zhb_asa/rank_v8_coverage"
RUN_ROOT="$PROJECT_ROOT/runs/query_answerability_bge_v1_20260821"
LOG_ROOT="$RUN_ROOT/logs"
MANIFEST_ROOT="$RUN_ROOT/manifests"
COMPLETION="$PROJECT_ROOT/schedule/complete_missing_query_answerability_v3_5.sh"
BGE="$PROJECT_ROOT/schedule/run_bge_document_v1_when_healthy.sh"

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

if ! mountpoint -q "$MOUNT_ROOT"; then
  echo "output disk is not mounted at $MOUNT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT" || ! -w "$PROJECT_ROOT" ]]; then
  echo "mounted project root is absent or not writable: $PROJECT_ROOT" >&2
  exit 1
fi
if [[ -z "${BIGMODEL_API_KEY:-}" ]]; then
  echo "BIGMODEL_API_KEY is not set" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT" "$MANIFEST_ROOT"
exec 9>"$MANIFEST_ROOT/completion_then_bge.lock"
if ! flock -n 9; then
  log "another completion-then-BGE supervisor owns the lock"
  exit 0
fi
printf '%s\n' "$$" >"$MANIFEST_ROOT/completion_then_bge.pid"

log "waiting for 241 load and mounted-label I/O to recover before completion"
while true; do
  read -r load_one _ < /proc/loadavg
  load_integer=${load_one%%.*}
  io_ready=0
  if (( load_integer < 245 )) && timeout 20 /usr/bin/python3 -c \
    'from pathlib import Path; print(len(Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/data/query_answerability_v3/input/train_safe.jsonl").open("rb").readline()))' \
    >/dev/null 2>&1; then
    io_ready=1
  fi
  if (( io_ready == 1 )); then
    log "completion health gate passed: load=$load_one"
    break
  fi
  log "completion health gate waiting: load=$load_one label_io=$io_ready"
  sleep 60
done

log "starting append-only missing-query completion"
set +e
"$COMPLETION" >>"$LOG_ROOT/completion_then_bge.log" 2>&1
completion_status=$?
set -e
unset BIGMODEL_API_KEY
if (( completion_status != 0 )); then
  log "completion failed with status=$completion_status; BGE training will not start"
  exit "$completion_status"
fi
log "completion succeeded; handing over to the BGE health-gated supervisor"
exec "$BGE"
