#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=${REMOTE_STORE}/asa_train/work/rank_v8_coverage
RUN_ROOT="$WORK_DIR/data/answer_support_v2/deepseek_flash_v2_2_pointwise"
LOG_ROOT="$WORK_DIR/logs/answer_support_v2/deepseek_flash_v2_2_pointwise"
STATE_ROOT="$WORK_DIR/run/answer_support_v2/deepseek_flash_v2_2_pointwise"
PYTHON_BIN=${REMOTE_STORE}/asa_train/env/train/bin/python3
LABEL_SCRIPT="$WORK_DIR/scripts/label_coverage_candidates_with_deepseek_v2.py"
START_AT="2026-08-19 12:01:00"
STOP_AT="2026-08-19 13:55:00"

mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$STATE_ROOT"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  printf '%s missing DEEPSEEK_API_KEY\n' "$(date '+%F %T')" >> "$LOG_ROOT/scheduler.log"
  exit 2
fi

wait_until() {
  local target="$1"
  local target_epoch now delay
  target_epoch="$(date -d "$target" +%s)"
  now="$(date +%s)"
  delay=$((target_epoch - now))
  if ((delay > 0)); then
    sleep "$delay"
  fi
}

wait_until "$START_AT"

stop_epoch="$(date -d "$STOP_AT" +%s)"
now="$(date +%s)"
if ((now >= stop_epoch)); then
  printf '%s start window already expired\n' "$(date '+%F %T')" >> "$LOG_ROOT/scheduler.log"
  exit 3
fi

printf '%s starting protocol=answer_support_v2 prompt=answer_support_v2.2_pointwise\n' \
  "$(date '+%F %T')" >> "$LOG_ROOT/scheduler.log"

common_args=(
  --model deepseek-v4-flash
  --endpoint https://api.deepseek.com/chat/completions
  --api-key-env DEEPSEEK_API_KEY
  --max-workers 1
  --claim-max-tokens 2048
  --max-tokens 16384
  --timeout 900
  --retries 3
  --expected-candidates 120
)

"$PYTHON_BIN" "$LABEL_SCRIPT" \
  --input "$WORK_DIR/data/answer_support_v2/input/train_safe.jsonl" \
  --output "$RUN_ROOT/train_teacher.jsonl" \
  --failures-output "$RUN_ROOT/train_failures.jsonl" \
  --claims-output "$RUN_ROOT/train_claims.jsonl" \
  "${common_args[@]}" > "$LOG_ROOT/train.log" 2>&1 &
train_pid=$!

"$PYTHON_BIN" "$LABEL_SCRIPT" \
  --input "$WORK_DIR/data/answer_support_v2/input/val_safe.jsonl" \
  --output "$RUN_ROOT/val_teacher.jsonl" \
  --failures-output "$RUN_ROOT/val_failures.jsonl" \
  --claims-output "$RUN_ROOT/val_claims.jsonl" \
  "${common_args[@]}" > "$LOG_ROOT/val.log" 2>&1 &
val_pid=$!

printf '%s %s\n' "$train_pid" "$val_pid" > "$STATE_ROOT/worker.pids"

remaining=$((stop_epoch - $(date +%s)))
if ((remaining > 0)); then
  sleep "$remaining"
fi

for pid in "$train_pid" "$val_pid"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

deadline=$((SECONDS + 30))
while ((SECONDS < deadline)); do
  alive=0
  for pid in "$train_pid" "$val_pid"; do
    kill -0 "$pid" 2>/dev/null && alive=1
  done
  if ((alive == 0)); then
    break
  fi
  sleep 1
done

for pid in "$train_pid" "$val_pid"; do
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

wait "$train_pid" 2>/dev/null || true
wait "$val_pid" 2>/dev/null || true

unset DEEPSEEK_API_KEY
printf '%s stopped train_rows=%s val_rows=%s\n' \
  "$(date '+%F %T')" \
  "$(wc -l < "$RUN_ROOT/train_teacher.jsonl" 2>/dev/null || echo 0)" \
  "$(wc -l < "$RUN_ROOT/val_teacher.jsonl" 2>/dev/null || echo 0)" \
  >> "$LOG_ROOT/scheduler.log"
