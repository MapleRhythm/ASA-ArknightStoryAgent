#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
INPUT=$BASE/data/query_answerability_v3/blind67_relabel_input.jsonl
LABEL_DIR=$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
LABELS=$LABEL_DIR/blind67_labels.jsonl
FAILURES=$LABEL_DIR/blind67_labels.failures.jsonl
LOGS=$BASE/logs/resume_20260825

mkdir -p "$BASE/scripts" "$LOGS" "$BASE/locks"

install_checked() {
  local source=$1
  local target=$2
  if [[ -e "$target" ]] && ! cmp -s "$source" "$target"; then
    echo "refusing to replace different existing script: $target" >&2
    exit 4
  fi
  cp "$source" "$target"
  chmod +x "$target"
}

install_checked /tmp/build_dense_sidecar_index.py "$BASE/scripts/build_dense_sidecar_index.py"
install_checked /tmp/audit_query_answerability_single_split.py "$BASE/scripts/audit_query_answerability_single_split.py"
install_checked /tmp/audit_query_answerability_labels.py "$BASE/scripts/audit_query_answerability_labels.py"

echo "[$(date '+%F %T')] resume bootstrap on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

unique_count=$(
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
)
echo "blind67_unique=$unique_count"

if (( unique_count < 67 )) \
  && ! pgrep -f '[l]abel_query_answerability_candidates.py.*blind67_relabel_input|[a]sa_blind67_label_retry_controller_241.sh' >/dev/null; then
  nohup bash /tmp/asa_blind67_label_retry_controller_241.sh \
    >"$LABEL_DIR/label_retry_controller_20260825.log" 2>&1 < /dev/null &
  echo $! >"$LABEL_DIR/label_retry_controller_20260825.pid"
  echo "started blind67 retry controller pid=$!"
fi

if ! pgrep -f '[a]sa_blind67_finalize_241.sh' >/dev/null \
  && [[ ! -s "$BASE/runs/query_listwise_v2_20260823/blind67_same_protocol_20260825/comparison.json" ]]; then
  nohup bash /tmp/asa_blind67_finalize_241.sh \
    >"$LOGS/blind67_finalize.log" 2>&1 < /dev/null &
  echo $! >"$LOGS/blind67_finalize.pid"
  echo "started blind67 finalizer pid=$!"
fi

if ! pgrep -f '[a]sa_embedding_4b_chain_241.sh' >/dev/null \
  && [[ ! -s "$BASE/runs/qwen3_embedding_4b_ablation_20260825/easy200_qwen3_4b_segment.json" ]]; then
  nohup bash /tmp/asa_embedding_4b_chain_241.sh \
    >"$LOGS/embedding_4b_chain.log" 2>&1 < /dev/null &
  echo $! >"$LOGS/embedding_4b_chain.pid"
  echo "started embedding chain pid=$!"
fi

pgrep -af '[l]abel_query_answerability_candidates.py|[a]sa_blind67_label_retry_controller_241.sh|[a]sa_blind67_finalize_241.sh|[a]sa_embedding_4b_chain_241.sh' || true
