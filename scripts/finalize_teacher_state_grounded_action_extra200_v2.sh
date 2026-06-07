#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

LOG_DIR="logs/teacher_state_grounded_action_sft_extra200_v2"
BASE_DIR="data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_parallel"
RAW_MERGED="data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_merged"
COMPACT="data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_compact_prompt_top10_qkeep"
FILTERED="data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_compact_prompt_top10_qkeep_len10000"

for gpu in 1 2; do
  pid_file="$LOG_DIR/gpu${gpu}.pid"
  [[ -f "$pid_file" ]] || { echo "missing pid file: $pid_file" >&2; exit 1; }
done

while true; do
  running=0
  for gpu in 1 2; do
    pid="$(cat "$LOG_DIR/gpu${gpu}.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      running=1
    fi
  done
  [[ "$running" -eq 0 ]] && break
  sleep 60
done

for gpu in 1 2; do
  summary="$BASE_DIR/gpu${gpu}/summary.json"
  [[ -f "$summary" ]] || { echo "missing shard summary: $summary" >&2; exit 1; }
done

"$PY" - <<'PY'
import json
import random
from collections import Counter
from pathlib import Path

base = Path("data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_parallel")
out = Path("data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_merged")
dataset_name = "teacher_state_grounded_action_sft_extra200_v2_merged"
out.mkdir(parents=True, exist_ok=True)

records = []
source_shards = []
for shard in ["gpu1", "gpu2"]:
    shard_dir = base / shard
    shard_records = []
    for split in ["train.json", "val.json"]:
        path = shard_dir / split
        if path.exists():
            shard_records.extend(json.loads(path.read_text(encoding="utf-8")))
    summary = json.loads((shard_dir / "summary.json").read_text(encoding="utf-8"))
    source_shards.append({
        "name": shard,
        "questions": summary.get("questions"),
        "records_total": len(shard_records),
        "actions": summary.get("actions", {}),
    })
    records.extend(shard_records)

rng = random.Random(20260603)
rng.shuffle(records)
val_count = max(1, round(len(records) * 0.08)) if len(records) > 10 else 0
val = records[:val_count]
train = records[val_count:]

role_tags = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}

def dataset_info(name):
    def entry(file_name):
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": role_tags,
        }
    return {f"{name}_train": entry("train.json"), f"{name}_val": entry("val.json")}

def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

write(out / "train.json", train)
write(out / "val.json", val)
write(out / "dataset_info.json", dataset_info(dataset_name))
actions = Counter()
for record in records:
    try:
        payload = json.loads(record["conversations"][-1]["value"])
        actions[payload.get("next_action", "unknown")] += 1
    except Exception:
        actions["unknown"] += 1
write(out / "summary.json", {
    "output_dir": str(out),
    "dataset_name": dataset_name,
    "records_total": len(records),
    "records_train": len(train),
    "records_val": len(val),
    "actions": dict(actions),
    "source_shards": source_shards,
})
print(json.dumps(json.loads((out / "summary.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
PY

"$PY" scripts/convert_grounded_action_sft_to_short_schema.py \
  --input-dir "$RAW_MERGED" \
  --output-dir "$COMPACT" \
  --dataset-name teacher_state_grounded_action_sft_extra200_v2_compact_prompt_top10_qkeep \
  --mode compact_prompt \
  --top-k-evidence 10 \
  --overwrite

"$PY" scripts/filter_sharegpt_by_token_len.py \
  --input-dir "$COMPACT" \
  --output-dir "$FILTERED" \
  --dataset-name teacher_state_grounded_action_sft_extra200_v2_compact_prompt_top10_qkeep_len10000 \
  --tokenizer model/qwen3.5-4b \
  --cutoff-len 10000 \
  --overwrite

"$PY" - <<'PY'
import json
from pathlib import Path

target = Path("data/processed/llama_factory/teacher_state_grounded_action_sft_extra200_v2_compact_prompt_top10_qkeep_len10000")
summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
print(json.dumps(summary, ensure_ascii=False, indent=2))
train = json.loads((target / "train.json").read_text(encoding="utf-8"))
if train:
    sample = train[0]
    payload = json.loads(sample["conversations"][-1]["value"])
    print(json.dumps({
        "sample_id": sample.get("id"),
        "token_len": sample.get("token_len"),
        "prompt_preview": sample["conversations"][0]["value"][:800],
        "assistant": payload,
    }, ensure_ascii=False, indent=2)[:3000])
PY
