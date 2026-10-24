#!/usr/bin/env bash
set -euo pipefail

# Run GLM-5.3 full-candidate teacher labelling without touching existing
# DeepSeek/Flash outputs.  This script is meant to be executed from the ASA
# project root on the mounted disk, for example:
#
#   cd ${REMOTE_STORE}/ASA-ArknightStoryAgent
#   export BIGMODEL_API_KEY=...
#   bash scripts/run_glm_5_3_full_candidate_labeling.sh
#
# Optional overrides:
#   GLM_MODEL=glm-5.3
#   GLM_MAX_WORKERS=1
#   GLM_MAX_TOKENS=8192
#   GLM_TIMEOUT=300
#   GLM_RETRIES=3
#   GLM_OUTPUT_DIR=data/processed/evidence_chain_reranker/rank_v8_coverage/glm_5_3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${BIGMODEL_API_KEY:-}" ]]; then
  echo "BIGMODEL_API_KEY is required; export it in the shell environment." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
GLM_MODEL="${GLM_MODEL:-glm-5.3}"
GLM_ENDPOINT="${GLM_ENDPOINT:-https://open.bigmodel.cn/api/paas/v4/chat/completions}"
GLM_MAX_WORKERS="${GLM_MAX_WORKERS:-1}"
GLM_MAX_TOKENS="${GLM_MAX_TOKENS:-8192}"
GLM_TIMEOUT="${GLM_TIMEOUT:-300}"
GLM_RETRIES="${GLM_RETRIES:-3}"
GLM_OUTPUT_DIR="${GLM_OUTPUT_DIR:-data/processed/evidence_chain_reranker/rank_v8_coverage/glm_5_3}"

LABEL_SCRIPT="scripts/label_coverage_candidates_with_glm.py"
if [[ ! -f "$LABEL_SCRIPT" ]]; then
  echo "Missing $LABEL_SCRIPT under $ROOT_DIR" >&2
  exit 2
fi

mkdir -p "$GLM_OUTPUT_DIR" "$GLM_OUTPUT_DIR/logs"

find_input() {
  local split="$1"
  local expected_lines="$2"
  "$PYTHON_BIN" - "$split" "$expected_lines" <<'PY'
import json
import sys
from pathlib import Path

split = sys.argv[1]
expected_lines = int(sys.argv[2])
root = Path.cwd()
preferred_dirs = [
    root / "data/processed/evidence_chain_reranker/rank_v8_coverage",
    root / "data/processed/evidence_chain_reranker",
    root / "data/processed",
]
patterns = [
    f"*{split}*candidate*.jsonl",
    f"*{split}*raw*.jsonl",
    f"*{split}*.jsonl",
]

seen = set()
candidates = []
for base in preferred_dirs:
    if not base.exists():
        continue
    for pattern in patterns:
        for path in base.rglob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                line_count = 0
                all_120 = True
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        line_count += 1
                        record = json.loads(line)
                        if len(record.get("candidates") or []) != 120:
                            all_120 = False
                            break
                if line_count == expected_lines and all_120:
                    candidates.append(path)
            except Exception:
                continue

if not candidates:
    print(
        f"No {split} JSONL with {expected_lines} lines and 120 candidates per line found.",
        file=sys.stderr,
    )
    sys.exit(1)

ranked = sorted(
    candidates,
    key=lambda p: (
        "rank_v8_coverage" not in str(p),
        "raw" not in p.name and "candidate" not in p.name,
        len(str(p)),
        str(p),
    ),
)
print(ranked[0])
PY
}

TRAIN_INPUT="${TRAIN_INPUT:-$(find_input train 533)}"
VAL_INPUT="${VAL_INPUT:-$(find_input val 116)}"

echo "project=$ROOT_DIR"
echo "model=$GLM_MODEL"
echo "endpoint=$GLM_ENDPOINT"
echo "train_input=$TRAIN_INPUT"
echo "val_input=$VAL_INPUT"
echo "output_dir=$GLM_OUTPUT_DIR"

COMMON_ARGS=(
  --model "$GLM_MODEL"
  --endpoint "$GLM_ENDPOINT"
  --api-key-env BIGMODEL_API_KEY
  --max-workers "$GLM_MAX_WORKERS"
  --max-tokens "$GLM_MAX_TOKENS"
  --timeout "$GLM_TIMEOUT"
  --retries "$GLM_RETRIES"
  --expected-candidates 120
  --temperature 0
)

SMOKE_OUTPUT="$GLM_OUTPUT_DIR/smoke_teacher.jsonl"
rm -f "$SMOKE_OUTPUT" "$GLM_OUTPUT_DIR/smoke_teacher.summary.json" "$GLM_OUTPUT_DIR/smoke_teacher.failures.jsonl"

echo "Starting GLM smoke test with one full 120-candidate train query..."
"$PYTHON_BIN" "$LABEL_SCRIPT" \
  --input "$TRAIN_INPUT" \
  --output "$SMOKE_OUTPUT" \
  --limit 1 \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "$GLM_OUTPUT_DIR/logs/smoke.log"

echo "Smoke passed. Starting full train and val labelling."
TRAIN_LOG="$GLM_OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"
VAL_LOG="$GLM_OUTPUT_DIR/logs/val_$(date +%Y%m%d_%H%M%S).log"

(
  "$PYTHON_BIN" "$LABEL_SCRIPT" \
    --input "$TRAIN_INPUT" \
    --output "$GLM_OUTPUT_DIR/train_teacher.jsonl" \
    "${COMMON_ARGS[@]}"
) >"$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!

(
  "$PYTHON_BIN" "$LABEL_SCRIPT" \
    --input "$VAL_INPUT" \
    --output "$GLM_OUTPUT_DIR/val_teacher.jsonl" \
    "${COMMON_ARGS[@]}"
) >"$VAL_LOG" 2>&1 &
VAL_PID=$!

cat >"$GLM_OUTPUT_DIR/run_state.json" <<EOF_STATE
{
  "model": "$GLM_MODEL",
  "endpoint": "$GLM_ENDPOINT",
  "train_input": "$TRAIN_INPUT",
  "val_input": "$VAL_INPUT",
  "train_output": "$GLM_OUTPUT_DIR/train_teacher.jsonl",
  "val_output": "$GLM_OUTPUT_DIR/val_teacher.jsonl",
  "train_log": "$TRAIN_LOG",
  "val_log": "$VAL_LOG",
  "train_pid": $TRAIN_PID,
  "val_pid": $VAL_PID,
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF_STATE

echo "train_pid=$TRAIN_PID train_log=$TRAIN_LOG"
echo "val_pid=$VAL_PID val_log=$VAL_LOG"
echo "state=$GLM_OUTPUT_DIR/run_state.json"
