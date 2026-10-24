#!/usr/bin/env bash
set -euo pipefail

BASE=${REMOTE_ASA_ROOT}/rank_v8_coverage
REPO=${REMOTE_HOME}/ASA-ArknightStoryAgent
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
INPUT=$BASE/data/query_answerability_v3/blind67_relabel_input.jsonl
LABEL_DIR=$BASE/data/query_answerability_v3/blind67_glm_5_3_high_v3_5_20260824
LABELS=$LABEL_DIR/blind67_labels.jsonl
OUT_DIR=$BASE/runs/query_listwise_v2_20260823/blind67_same_protocol_20260825
MATERIALIZED_DIR=$LABEL_DIR/materialized
MATERIALIZED=$MATERIALIZED_DIR/blind67_same_protocol.jsonl
GPU_LOCK=$BASE/locks/gpu0_asa_experiments.lock

mkdir -p "$OUT_DIR" "$MATERIALIZED_DIR" "$BASE/locks"

unique_rows() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
keys = set()
if p.exists():
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row.get("source_name") or ""), str(row.get("query") or "")))
print(len(keys))
PY
}

wait_gpu0_idle() {
  while true; do
    IFS=, read -r memory_used utilization < <(
      nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
    )
    memory_used=${memory_used// /}
    utilization=${utilization// /}
    if [[ "$memory_used" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
      && (( memory_used < 1000 && utilization < 10 )); then
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for GPU0: memory=${memory_used}MiB util=${utilization}%"
    sleep 30
  done
}

while pgrep -f '[l]abel_query_answerability_candidates.py.*blind67_relabel_input|[a]sa_blind67_label_retry_controller_241.sh' >/dev/null; do
  echo "[$(date '+%F %T')] waiting for blind67 teacher"
  sleep 30
done

count=$(unique_rows "$LABELS")
echo "[$(date '+%F %T')] blind67 unique labels=$count"
if (( count != 67 )); then
  echo "blind67 remains incomplete; refusing to materialize/evaluate" >&2
  exit 2
fi

"$PY" "$BASE/scripts/materialize_priority_teacher_shards.py" \
  --input "$INPUT" \
  --teacher "$LABELS" \
  --output "$MATERIALIZED" \
  --manifest "$MATERIALIZED_DIR/blind67_same_protocol.manifest.json" \
  --expected-candidates 120

"$PY" "$BASE/scripts/audit_query_answerability_single_split.py" \
  --labels "$MATERIALIZED" \
  --input "$INPUT" \
  --report "$MATERIALIZED_DIR/blind67_same_protocol.audit.json" \
  --expected-candidates 120 \
  --audit-module-dir "$BASE/scripts"

evaluate_model() {
  local model=$1
  local output=$2
  if [[ -s "$output" ]]; then
    echo "skip existing result: $output"
    return 0
  fi
  exec 9>"$GPU_LOCK"
  flock 9
  wait_gpu0_idle
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$BASE/scripts/evaluate_query_answerability_reranker.py" \
    --input "$MATERIALIZED" \
    --model "$model" \
    --output "$output" \
    --device cuda --max-length 1024 --batch-size 16 \
    --cutoff 1 --cutoff 5 --cutoff 10 --cutoff 32
  flock -u 9
}

evaluate_model \
  "$REPO/model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch" \
  "$OUT_DIR/v6_result.json"
evaluate_model \
  "$BASE/runs/query_listwise_v2_20260823/full_train" \
  "$OUT_DIR/listwise_v2_result.json"

"$PY" - "$OUT_DIR/v6_result.json" "$OUT_DIR/listwise_v2_result.json" \
  "$OUT_DIR/comparison.json" <<'PY'
import json, sys
from pathlib import Path
v6 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
new = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
v6m, newm = v6["aggregate"], new["aggregate"]
keys = sorted(set(v6m) & set(newm))
delta = {
    key: float(newm[key]) - float(v6m[key])
    for key in keys
    if key != "records" and isinstance(v6m[key], (int, float))
}
result = {
    "v6": v6m,
    "listwise_v2": newm,
    "delta_listwise_v2_minus_v6": delta,
    "latency_seconds": {
        "v6": v6.get("latency_seconds", {}),
        "listwise_v2": new.get("latency_seconds", {}),
    },
}
Path(sys.argv[3]).write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

echo "[$(date '+%F %T')] blind67 same-protocol evaluation complete"
