#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
else
  PYTHON_BIN="python"
fi

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
normalize_pythonpath() {
  local overlay="$ROOT_DIR/.python_packages/train"
  local src_path="$ROOT_DIR/src"
  local existing="${PYTHONPATH:-}"
  local filtered=""
  local entry
  IFS=':' read -r -a entries <<< "$existing"
  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    [[ "$entry" == "$overlay" ]] && continue
    [[ "$entry" == "$src_path" ]] && continue
    filtered="${filtered:+$filtered:}$entry"
  done
  if [[ "${GOLDENGLOW_USE_TRAIN_OVERRIDE:-}" =~ ^(0|false|False|no|off)$ ]]; then
    export PYTHONPATH="$src_path${filtered:+:$filtered}"
  else
    export PYTHONPATH="$overlay:$src_path${filtered:+:$filtered}"
  fi
}

normalize_pythonpath

RUN_NAME="${RUN_NAME:-eval50_hard10_gpu_abstain_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-outputs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-logs/$RUN_NAME}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime_inference_gpu.json}"
LORA_PATH="${LORA_PATH:-model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1}"
EVAL50_QUESTIONS="${EVAL50_QUESTIONS:-outputs/eval_soda_api_verifier_v2/eval50_questions.txt}"
HARD_QUESTIONS="${HARD_QUESTIONS:-outputs/eval_soda_api_verifier_v2/hard10_questions.txt}"
GPUS="${GPUS:-0,1,2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.52}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}"
DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
RUN_EVAL50="${RUN_EVAL50:-1}"
RUN_HARD="${RUN_HARD:-1}"
QUESTION_TIMEOUT_SECONDS="${QUESTION_TIMEOUT_SECONDS:-0}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "[error] missing runtime config: $RUNTIME_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$LORA_PATH" || ! -f "$LORA_PATH/adapter_config.json" ]]; then
  echo "[error] invalid LoRA path: $LORA_PATH" >&2
  exit 2
fi
if [[ "$RUN_EVAL50" == "1" && ! -f "$EVAL50_QUESTIONS" ]]; then
  echo "[error] missing eval50 questions: $EVAL50_QUESTIONS" >&2
  exit 2
fi
if [[ "$RUN_HARD" == "1" && ! -f "$HARD_QUESTIONS" ]]; then
  echo "[error] missing hard questions: $HARD_QUESTIONS" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "[error] GPUS is empty" >&2
  exit 2
fi
SHARDS="${#GPU_ARRAY[@]}"

echo "[config] run_name=$RUN_NAME"
echo "[config] out_dir=$OUT_DIR"
echo "[config] log_dir=$LOG_DIR"
echo "[config] gpus=$GPUS shards=$SHARDS"
echo "[config] lora_path=$LORA_PATH"

split_questions() {
  local src="$1"
  local prefix="$2"
  "$PYTHON_BIN" - "$src" "$prefix" "$SHARDS" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
prefix = Path(sys.argv[2])
shards = int(sys.argv[3])
questions = [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
base, rem = divmod(len(questions), shards)
start = 0
for idx in range(shards):
    count = base + (1 if idx < rem else 0)
    part = questions[start:start + count]
    start += count
    path = Path(f"{prefix}_shard{idx}_questions.txt")
    path.write_text("\n".join(part) + ("\n" if part else ""), encoding="utf-8")
    print(f"[split] {path} questions={len(part)}")
PY
}

run_dataset_parallel() {
  local name="$1"
  local questions_file="$2"

  echo "[stage] split $name"
  split_questions "$questions_file" "$OUT_DIR/${name}"
  rm -f "$OUT_DIR/${name}"_shard*_answers.jsonl "$OUT_DIR/${name}_answers.jsonl"

  local pids=()
  for shard in $(seq 0 $((SHARDS - 1))); do
    local gpu="${GPU_ARRAY[$shard]}"
    local shard_questions="$OUT_DIR/${name}_shard${shard}_questions.txt"
    local shard_output="$OUT_DIR/${name}_shard${shard}_answers.jsonl"
    local shard_log="$LOG_DIR/${name}_shard${shard}.log"
    local runner_log="$LOG_DIR/${name}_shard${shard}.runner.log"

    if [[ ! -s "$shard_questions" ]]; then
      : > "$shard_output"
      echo "[skip] $name shard=$shard gpu=$gpu empty"
      continue
    fi

    (
      set +e
      echo "[$(date)] start name=$name shard=$shard gpu=$gpu"
      cmd=(
        "$PYTHON_BIN" scripts/run_cpu_inference.py
        --runtime-config "$RUNTIME_CONFIG"
        --backend vllm
        --lora-path "$LORA_PATH"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
        --max-retrieval-rounds "$MAX_RETRIEVAL_ROUNDS"
        --questions-file "$shard_questions"
        --batch-output "$shard_output"
      )
      if [[ -n "$MAX_TOKENS" ]]; then
        cmd+=(--max-tokens "$MAX_TOKENS")
      fi
      if [[ "$ENFORCE_EAGER" == "1" ]]; then
        cmd+=(--enforce-eager)
      fi
      if [[ "$DISABLE_WEB_CONTEXT" == "1" ]]; then
        cmd+=(--disable-web-context)
      fi
      if [[ "$QUESTION_TIMEOUT_SECONDS" != "0" ]]; then
        timeout "$QUESTION_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$shard_log" 2>&1
      else
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$shard_log" 2>&1
      fi
      status=$?
      echo "[$(date)] done name=$name shard=$shard gpu=$gpu status=$status"
      exit "$status"
    ) > "$runner_log" 2>&1 &
    pids+=("$!")
    echo "[launch] $name shard=$shard gpu=$gpu pid=${pids[-1]}"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "[error] one or more $name shards failed" >&2
    for f in "$LOG_DIR/${name}"_shard*.runner.log "$LOG_DIR/${name}"_shard*.log; do
      [[ -f "$f" ]] || continue
      echo "--- $f" >&2
      tail -n 60 "$f" >&2 || true
    done
    exit 3
  fi

  echo "[stage] merge $name"
  for shard in $(seq 0 $((SHARDS - 1))); do
    cat "$OUT_DIR/${name}_shard${shard}_answers.jsonl" >> "$OUT_DIR/${name}_answers.jsonl"
  done
  wc -l "$OUT_DIR/${name}_answers.jsonl"
}

if [[ "$RUN_EVAL50" == "1" ]]; then
  run_dataset_parallel "eval50" "$EVAL50_QUESTIONS"
fi

if [[ "$RUN_HARD" == "1" ]]; then
  run_dataset_parallel "hard10" "$HARD_QUESTIONS"
fi

echo "[stage] analyze"
"$PYTHON_BIN" - "$OUT_DIR" <<'PY'
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

out_dir = Path(sys.argv[1])
markers = (
    "现有检索证据不足",
    "现有证据不足",
    "不足以确认",
    "无法确认",
    "无法判断",
    "不能确认",
    "没有足够",
    "未能找到",
    "无法回答",
    "当前证据只能确认",
)
marker_re = re.compile("|".join(re.escape(item) for item in markers))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def final_action(row: dict) -> str:
    trace = row.get("retrieval_trace") or []
    if not trace:
        return ""
    return str((trace[-1] or {}).get("planner_action") or "")


def is_abstain_like(row: dict) -> bool:
    return final_action(row) == "abstain" or bool(marker_re.search(str(row.get("answer") or "")))


def compact_top3(row: dict) -> list[dict]:
    existing = row.get("abstain_evidence_top3")
    if isinstance(existing, list) and existing:
        return existing[:3]
    evidence = row.get("evidence") or []
    output = []
    for item in evidence[:3]:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("clean_text") or "").split())
        if len(text) > 900:
            text = text[:899].rstrip() + "…"
        output.append(
            {
                "rank": len(output) + 1,
                "id": item.get("id"),
                "activity_name": item.get("activity_name"),
                "story_name": item.get("story_name"),
                "stage_code": item.get("stage_code"),
                "avg_tag": item.get("avg_tag"),
                "source_path": item.get("source_path"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "clean_text": text,
            }
        )
    return output


def summarize(name: str, rows: list[dict]) -> dict:
    rounds = Counter(len(row.get("retrieval_trace") or []) for row in rows)
    actions = Counter(final_action(row) for row in rows)
    seqs = Counter(">".join(str(step.get("planner_action") or "") for step in (row.get("retrieval_trace") or [])) for row in rows)
    errors = sum(1 for row in rows if str(row.get("error") or "").strip())
    abstain_like = sum(1 for row in rows if is_abstain_like(row))
    avg_elapsed = round(sum(float(row.get("elapsed_sec") or 0) for row in rows) / len(rows), 3) if rows else 0
    return {
        "name": name,
        "count": len(rows),
        "errors": errors,
        "abstain_like": abstain_like,
        "rounds": dict(sorted(rounds.items())),
        "final_actions": dict(actions),
        "action_sequences": dict(seqs.most_common()),
        "avg_elapsed_sec": avg_elapsed,
    }


summaries = []
all_abstain = []
for name in ("eval50", "hard10"):
    rows = read_jsonl(out_dir / f"{name}_answers.jsonl")
    if not rows:
        continue
    summaries.append(summarize(name, rows))
    for row in rows:
        if not is_abstain_like(row):
            continue
        record = {
            "dataset": name,
            "question": row.get("question"),
            "final_action": final_action(row),
            "answer": row.get("answer"),
            "abstain_evidence_top3": compact_top3(row),
        }
        all_abstain.append(record)

(out_dir / "summary.json").write_text(
    json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(out_dir / "abstain_top3_evidence.jsonl").write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in all_abstain) + ("\n" if all_abstain else ""),
    encoding="utf-8",
)

parts = ["# Eval Abstain Top3 Evidence\n\n", f"count: {len(all_abstain)}\n"]
for idx, record in enumerate(all_abstain, 1):
    parts.append(f"\n## {idx}. [{record['dataset']}] {record['question']}\n")
    parts.append(f"- final_action: `{record['final_action']}`\n")
    parts.append(f"- answer: {record['answer']}\n")
    for ev in record["abstain_evidence_top3"]:
        parts.append(f"\n### Evidence {ev.get('rank')}\n")
        parts.append(f"- id: `{ev.get('id')}`\n")
        parts.append(f"- activity/story: {ev.get('activity_name') or ''} / {ev.get('story_name') or ''}\n")
        parts.append(f"- rerank_score: `{ev.get('rerank_score')}`\n")
        parts.append(f"\n```text\n{ev.get('clean_text') or ''}\n```\n")
(out_dir / "abstain_top3_evidence.md").write_text("".join(parts), encoding="utf-8")

summary_md = ["# Eval Summary\n\n"]
for item in summaries:
    summary_md.append(f"## {item['name']}\n")
    summary_md.append(f"- count: {item['count']}\n")
    summary_md.append(f"- errors: {item['errors']}\n")
    summary_md.append(f"- abstain_like: {item['abstain_like']}\n")
    summary_md.append(f"- avg_elapsed_sec: {item['avg_elapsed_sec']}\n")
    summary_md.append(f"- rounds: `{item['rounds']}`\n")
    summary_md.append(f"- final_actions: `{item['final_actions']}`\n")
    summary_md.append(f"- action_sequences: `{item['action_sequences']}`\n\n")
(out_dir / "summary.md").write_text("".join(summary_md), encoding="utf-8")

print(json.dumps({"summaries": summaries, "abstain_records": len(all_abstain)}, ensure_ascii=False, indent=2))
PY

echo "[done] answers:"
[[ -f "$OUT_DIR/eval50_answers.jsonl" ]] && echo "  $OUT_DIR/eval50_answers.jsonl"
[[ -f "$OUT_DIR/hard10_answers.jsonl" ]] && echo "  $OUT_DIR/hard10_answers.jsonl"
echo "[done] reports:"
echo "  $OUT_DIR/summary.md"
echo "  $OUT_DIR/summary.json"
echo "  $OUT_DIR/abstain_top3_evidence.md"
echo "  $OUT_DIR/abstain_top3_evidence.jsonl"
