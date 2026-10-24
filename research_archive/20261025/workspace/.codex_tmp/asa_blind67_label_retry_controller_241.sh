#!/usr/bin/env bash
set -u

BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
INPUT=$BASE/data/query_answerability_v3/blind67_relabel_input.jsonl
LABEL_DIR=$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
LABELS=$LABEL_DIR/blind67_labels.jsonl
FAILURES=$LABEL_DIR/blind67_labels.failures.jsonl

unique_count() {
  "$PY" - "$LABELS" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
keys = set()
if p.exists():
    for line in p.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            keys.add((str(row.get("source_name") or ""), str(row.get("query") or "")))
print(len(keys))
PY
}

source ~/.bigmodel_env
for round in 1 2 3 4; do
  count=$(unique_count)
  echo "[$(date '+%F %T')] retry_round=$round unique=$count"
  if (( count == 67 )); then
    exit 0
  fi
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy \
    "$PY" "$BASE/scripts/label_query_answerability_candidates.py" \
    --input "$INPUT" \
    --output "$LABELS" \
    --failures-output "$FAILURES" \
    --model glm-5.3 \
    --endpoint https://open.bigmodel.cn/api/coding/paas/v4/chat/completions \
    --api-key-env BIGMODEL_API_KEY \
    --provider bigmodel-coding-plan \
    --temperature 0 --reasoning-effort high \
    --max-tokens 32768 --max-workers 1 \
    >"$LABEL_DIR/label_retry_20260825_round${round}.log" 2>&1 || true
  sleep 60
done

count=$(unique_count)
echo "[$(date '+%F %T')] retries exhausted unique=$count"
(( count == 67 ))
