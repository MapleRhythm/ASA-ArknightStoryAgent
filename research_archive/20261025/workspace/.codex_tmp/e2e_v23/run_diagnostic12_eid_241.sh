#!/usr/bin/env bash
set -euo pipefail

RUN=${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_no_minirag_diagnostic12_eid_20260909.v24
REPO=${REMOTE_HOME}/ASA-ArknightStoryAgent
REL=$REPO/release/ASA-ArknightStoryAgent
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python
GPU_INDEX=2
LOCK=${REMOTE_ASA_ROOT}/rank_v8_coverage/locks/gpu2_asa_e2e.lock

if [[ -e "$RUN/results.jsonl" || -e "$RUN/exit_code.txt" ]]; then
  echo "run output already exists; refusing to overwrite" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
flock -n 9 || {
  echo "ASA GPU2 experiment lock is already held" >&2
  exit 4
}

: >"$RUN/gpu_idle_samples.csv"
echo "timestamp,memory_used_mib,utilization_percent,compute_processes" >>"$RUN/gpu_idle_samples.csv"
for _ in $(seq 1 12); do
  IFS=, read -r memory_used utilization < <(
    nvidia-smi -i "$GPU_INDEX" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory_used=${memory_used// /}
  utilization=${utilization// /}
  process_count=$(
    nvidia-smi -i "$GPU_INDEX" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l
  )
  timestamp=$(date --iso-8601=seconds)
  echo "$timestamp,$memory_used,$utilization,$process_count" \
    | tee -a "$RUN/gpu_idle_samples.csv"
  if (( memory_used >= 256 || utilization >= 10 || process_count != 0 )); then
    echo "GPU2 was not continuously idle; refusing to launch" >&2
    exit 5
  fi
  sleep 5
done

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$REPO/.python_packages/train:$REL/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

set +e
"$PY" "$REL/scripts/run_cpu_inference.py" \
  --runtime-config "$RUN/runtime.json" \
  --questions-file "$RUN/questions.txt" \
  --batch-output "$RUN/results.jsonl" \
  2>&1 | tee "$RUN/run.log"
run_rc=${PIPESTATUS[0]}
set -e
echo "$run_rc" >"$RUN/exit_code.txt"

set +e
"$PY" - "$RUN/results.jsonl" "$RUN/postrun_counts.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
payload = {
    "rows": len(rows),
    "answers": sum(bool(row.get("answer")) for row in rows),
    "errors": sum(bool(row.get("error")) for row in rows),
    "error_types": Counter(
        str(row.get("error") or "").split(":", 1)[0] for row in rows if row.get("error")
    ),
}
Path(sys.argv[2]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False))
if len(rows) != 12 or payload["errors"]:
    raise SystemExit(6)
PY
postrun_rc=$?
set -e

if (( run_rc != 0 )); then
  exit "$run_rc"
fi
exit "$postrun_rc"
