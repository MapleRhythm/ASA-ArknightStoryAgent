#!/usr/bin/env bash
set -euo pipefail

MOUNT_ROOT=${REMOTE_STORAGE}
PROJECT_ROOT="$MOUNT_ROOT/zhb_asa/rank_v8_coverage"
DATA_ROOT="$PROJECT_ROOT/data/query_answerability_v3"
INPUT_ROOT="$DATA_ROOT/input"
DS_ROOT="$DATA_ROOT/deepseek_flash_high_v3_5"
GLM_ROOT="$DATA_ROOT/glm_5_3_high_coding_v3_5_remainder"
OUTPUT_ROOT="$DATA_ROOT/completion_v3_5"
RUN_ROOT="$PROJECT_ROOT/runs/query_answerability_bge_v1_20260821"
LOG_ROOT="$RUN_ROOT/logs/completion_v3_5"
MANIFEST_ROOT="$RUN_ROOT/manifests"
BUILDER="$PROJECT_ROOT/scripts/build_missing_query_answerability_inputs.py"
LABELER="$PROJECT_ROOT/scripts/label_query_answerability_candidates.py"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

if ! mountpoint -q "$MOUNT_ROOT"; then
  log "refusing to write: output disk is not mounted at $MOUNT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT" || ! -w "$PROJECT_ROOT" ]]; then
  log "mounted project root is absent or not writable: $PROJECT_ROOT" >&2
  exit 1
fi
if [[ -z "${BIGMODEL_API_KEY:-}" ]]; then
  log "BIGMODEL_API_KEY is not set; the key is never accepted as a CLI argument" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/input" "$LOG_ROOT" "$MANIFEST_ROOT" "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
# This host sometimes inherits stale localhost proxy variables while no proxy
# listener exists.  The BigModel endpoint is directly reachable; prevent
# urllib from routing teacher requests to a refused 127.0.0.1 socket.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec 9>"$MANIFEST_ROOT/completion_v3_5.lock"
if ! flock -n 9; then
  log "another completion process owns the lock"
  exit 0
fi

for required in \
  "$PYTHON_BIN" "$BUILDER" "$LABELER" \
  "$INPUT_ROOT/train_safe.jsonl" "$INPUT_ROOT/val_safe.jsonl" \
  "$DS_ROOT/train_teacher.jsonl" "$DS_ROOT/val_teacher.jsonl" \
  "$GLM_ROOT/train_teacher.jsonl" "$GLM_ROOT/val_teacher.jsonl"; do
  if [[ ! -e "$required" ]]; then
    log "missing required path: $required" >&2
    exit 1
  fi
done

build_missing() {
  local split="$1"
  local -a completed_args=(
    --completed "$DS_ROOT/${split}_teacher.jsonl"
    --completed "$GLM_ROOT/${split}_teacher.jsonl"
  )
  if [[ -s "$OUTPUT_ROOT/${split}_teacher.jsonl" ]]; then
    completed_args+=(--completed "$OUTPUT_ROOT/${split}_teacher.jsonl")
  fi
  "$PYTHON_BIN" "$BUILDER" \
    --input "$INPUT_ROOT/${split}_safe.jsonl" \
    "${completed_args[@]}" \
    --output "$OUTPUT_ROOT/input/${split}_missing.jsonl" \
    --manifest "$MANIFEST_ROOT/${split}_missing_before_completion.json"
}

build_missing train
build_missing val

common_args=(
  --model glm-5.3
  --endpoint https://open.bigmodel.cn/api/coding/paas/v4/chat/completions
  --api-key-env BIGMODEL_API_KEY
  --provider bigmodel-coding-plan
  --temperature 0
  --reasoning-effort high
  --max-workers 3
  --max-tokens 32768
  --timeout 900
  --retries 5
  --expected-candidates 120
)

for split in train val; do
  missing="$OUTPUT_ROOT/input/${split}_missing.jsonl"
  if [[ ! -s "$missing" ]]; then
    log "$split has no missing query groups"
    continue
  fi
  log "labelling $(wc -l <"$missing") missing $split query groups into a new shard"
  "$PYTHON_BIN" "$LABELER" \
    --input "$missing" \
    --output "$OUTPUT_ROOT/${split}_teacher.jsonl" \
    --failures-output "$OUTPUT_ROOT/${split}_failures.jsonl" \
    "${common_args[@]}" \
    >"$LOG_ROOT/${split}.log" 2>&1
done

# Rebuild the manifests after labelling.  They are the authoritative proof
# that no frozen query remains missing; non-zero missing counts block training.
build_missing train
build_missing val
unset BIGMODEL_API_KEY
log "completion pass finished"
