#!/usr/bin/env bash
set -euo pipefail

MOUNT_ROOT=${REMOTE_STORAGE}
WORK_DIR="$MOUNT_ROOT/zhb_asa/rank_v8_coverage"
DATA_ROOT="$WORK_DIR/data/query_answerability_v3"
RUN_ROOT="$DATA_ROOT/deepseek_flash_v3"
LOG_ROOT="$WORK_DIR/logs/query_answerability_v3/deepseek_flash_v3"
STATE_ROOT="$WORK_DIR/run/query_answerability_v3/deepseek_flash_v3"
LABEL_SCRIPT="$WORK_DIR/scripts/label_query_answerability_candidates.py"
PYTHON_BIN=python3

if ! mountpoint -q "$MOUNT_ROOT"; then
  echo "output disk is not mounted at $MOUNT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$WORK_DIR" || ! -w "$WORK_DIR" ]]; then
  echo "mounted work directory is absent or not writable: $WORK_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$STATE_ROOT" "$WORK_DIR/tmp/query_answerability_v3"
export TMPDIR="$WORK_DIR/tmp/query_answerability_v3"
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf '%s missing DEEPSEEK_API_KEY\n' "$(date '+%F %T')" >> "$LOG_ROOT/launcher.log"
  exit 2
fi

common_args=(
  --model deepseek-v4-flash
  --endpoint https://api.deepseek.com/chat/completions
  --api-key-env DEEPSEEK_API_KEY
  --provider deepseek
  --temperature 0
  --reasoning-effort high
  --max-workers 1
  --max-tokens 32768
  --timeout 900
  --retries 5
  --expected-candidates 120
)

printf '%s starting host=241 protocol=query_answerability_v3 prompt=query_answerability_v3.1_pointwise\n' \
  "$(date '+%F %T')" >> "$LOG_ROOT/launcher.log"

"$PYTHON_BIN" "$LABEL_SCRIPT" \
  --input "$DATA_ROOT/input/train_safe.jsonl" \
  --output "$RUN_ROOT/train_teacher.jsonl" \
  --failures-output "$RUN_ROOT/train_failures.jsonl" \
  "${common_args[@]}" > "$LOG_ROOT/train.log" 2>&1 &
train_pid=$!

"$PYTHON_BIN" "$LABEL_SCRIPT" \
  --input "$DATA_ROOT/input/val_safe.jsonl" \
  --output "$RUN_ROOT/val_teacher.jsonl" \
  --failures-output "$RUN_ROOT/val_failures.jsonl" \
  "${common_args[@]}" > "$LOG_ROOT/val.log" 2>&1 &
val_pid=$!

printf '%s %s\n' "$train_pid" "$val_pid" > "$STATE_ROOT/worker.pids"
wait "$train_pid"
train_status=$?
wait "$val_pid"
val_status=$?
unset DEEPSEEK_API_KEY
printf '%s stopped train_status=%s val_status=%s train_rows=%s val_rows=%s\n' \
  "$(date '+%F %T')" "$train_status" "$val_status" \
  "$(wc -l < "$RUN_ROOT/train_teacher.jsonl" 2>/dev/null || echo 0)" \
  "$(wc -l < "$RUN_ROOT/val_teacher.jsonl" 2>/dev/null || echo 0)" \
  >> "$LOG_ROOT/launcher.log"
