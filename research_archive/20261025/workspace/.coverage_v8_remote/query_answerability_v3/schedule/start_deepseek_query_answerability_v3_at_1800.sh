#!/usr/bin/env bash
set -euo pipefail

MOUNT_ROOT=${REMOTE_STORAGE}
WORK_DIR="$MOUNT_ROOT/zhb_asa/rank_v8_coverage"
LOG_ROOT="$WORK_DIR/logs/query_answerability_v3/deepseek_flash_v3"
STATE_ROOT="$WORK_DIR/run/query_answerability_v3/deepseek_flash_v3"
START_AT="${START_AT:-2026-08-19 18:00:00}"
if ! mountpoint -q "$MOUNT_ROOT"; then
  echo "output disk is not mounted at $MOUNT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$WORK_DIR" || ! -w "$WORK_DIR" ]]; then
  echo "mounted work directory is absent or not writable: $WORK_DIR" >&2
  exit 1
fi
mkdir -p "$LOG_ROOT" "$STATE_ROOT"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf '%s missing DEEPSEEK_API_KEY\n' "$(date '+%F %T')" >> "$LOG_ROOT/scheduler.log"
  exit 2
fi
start_epoch="$(date -d "$START_AT" +%s)"
now="$(date +%s)"
if ((now < start_epoch)); then
  printf '%s waiting until %s\n' "$(date '+%F %T')" "$START_AT" >> "$LOG_ROOT/scheduler.log"
  sleep "$((start_epoch-now))"
fi
exec "$WORK_DIR/schedule/run_deepseek_query_answerability_v3.sh"
