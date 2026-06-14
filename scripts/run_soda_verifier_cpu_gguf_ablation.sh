#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/qwen3.5-4b}"
RAW_LORA="${RAW_LORA:-$ROOT_DIR/model/lora/soda_blackbox_deepseek_v1_550_parallel_qwen35_4b_lr2e6_beta001_epoch1}"
VERIFIER_LORA="${VERIFIER_LORA:-$ROOT_DIR/model/lora/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/model/gguf_ablation/soda_verifier}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/soda_verifier_cpu_gguf_ablation}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"
CONVERT_BIN="${CONVERT_BIN:-$LLAMA_CPP_DIR/convert_hf_to_gguf.py}"

EVAL50_QUESTIONS="${EVAL50_QUESTIONS:-$ROOT_DIR/outputs/eval_soda_api_verifier_v2/eval50_questions.txt}"
HARD_QUESTIONS="${HARD_QUESTIONS:-$ROOT_DIR/outputs/eval_soda_api_verifier_v2/hard10_questions.txt}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_inference.json}"

RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_RAW="${RUN_RAW:-1}"
RUN_VERIFIER="${RUN_VERIFIER:-1}"
RUN_EVAL50="${RUN_EVAL50:-1}"
RUN_HARD="${RUN_HARD:-1}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
CLEAN_F16="${CLEAN_F16:-1}"
CLEAN_MERGED="${CLEAN_MERGED:-0}"

CTX_SIZE="${CTX_SIZE:-8192}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}"
THREADS="${THREADS:-$(nproc)}"
NO_RERANKER="${NO_RERANKER:-1}"
DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}"
LLAMA_DEVICE="${LLAMA_DEVICE:-}"

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "[error] missing $label: $path" >&2
    exit 2
  fi
}

require_path "base model" "$BASE_MODEL"
require_path "raw LoRA" "$RAW_LORA/adapter_config.json"
require_path "verifier LoRA" "$VERIFIER_LORA/adapter_config.json"
require_path "GGUF converter" "$CONVERT_BIN"
require_path "llama.cpp quantizer" "$QUANTIZE_BIN"
require_path "runtime config" "$RUNTIME_CONFIG"

mkdir -p "$WORK_DIR" "$OUTPUT_ROOT"

export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

safe_rm_generated() {
  local path="$1"
  case "$path" in
    "$ROOT_DIR"/model/gguf_ablation/*)
      rm -rf "$path"
      ;;
    *)
      echo "[error] refusing to remove non-ablation path: $path" >&2
      exit 3
      ;;
  esac
}

export_q4() {
  local label="$1"
  local lora_path="$2"
  local merged_dir="$WORK_DIR/${label}_merged_hf"
  local f16_gguf="$WORK_DIR/${label}-f16.gguf"
  local q4_gguf="$WORK_DIR/${label}-q4_k_m.gguf"

  if [[ "$FORCE_EXPORT" == "1" ]]; then
    safe_rm_generated "$merged_dir"
    rm -f "$f16_gguf" "$q4_gguf"
  fi

  if [[ -f "$q4_gguf" ]]; then
    echo "[skip] existing q4 gguf: $q4_gguf" >&2
    echo "$q4_gguf"
    return
  fi

  echo "[export] merge $label" >&2
  "$PYTHON_BIN" scripts/merge_lora_to_base.py \
    --base-model "$BASE_MODEL" \
    --lora-path "$lora_path" \
    --output-dir "$merged_dir" \
    --dtype float16 \
    --device cpu >&2

  echo "[export] convert $label to f16 gguf" >&2
  "$PYTHON_BIN" "$CONVERT_BIN" \
    "$merged_dir" \
    --outfile "$f16_gguf.tmp" \
    --outtype f16 >&2
  mv "$f16_gguf.tmp" "$f16_gguf"

  echo "[export] quantize $label to q4_k_m" >&2
  "$QUANTIZE_BIN" "$f16_gguf" "$q4_gguf.tmp" Q4_K_M >&2
  mv "$q4_gguf.tmp" "$q4_gguf"

  if [[ "$CLEAN_F16" == "1" ]]; then
    rm -f "$f16_gguf"
  fi
  if [[ "$CLEAN_MERGED" == "1" ]]; then
    safe_rm_generated "$merged_dir"
  fi

  echo "$q4_gguf"
}

summarize_output() {
  local out_dir="$1"
  "$PYTHON_BIN" - "$out_dir" <<'PY'
from __future__ import annotations

from collections import Counter
import json
import re
import sys
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
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def final_action(row: dict) -> str:
    trace = row.get("retrieval_trace")
    if isinstance(trace, list) and trace:
        last = trace[-1] if isinstance(trace[-1], dict) else {}
        return str(last.get("planner_action") or "")
    return str(row.get("final_action") or "")


def summarize(name: str, rows: list[dict]) -> dict:
    actions = Counter()
    sequences = Counter()
    errors = 0
    abstain_like = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        action = final_action(row)
        actions[action] += 1
        if action == "abstain" or marker_re.search(str(row.get("answer") or "")):
            abstain_like += 1
        trace = row.get("retrieval_trace")
        if isinstance(trace, list) and trace:
            seq = ">".join(str((item or {}).get("planner_action") or "") for item in trace if isinstance(item, dict))
        else:
            seq = action
        sequences[seq] += 1
    return {
        "name": name,
        "count": len(rows),
        "errors": errors,
        "abstain_like": abstain_like,
        "final_actions": dict(actions),
        "action_sequences": dict(sequences),
    }


payload = {"summaries": []}
for name in ("eval50", "hard10"):
    rows = read_jsonl(out_dir / f"{name}_answers.jsonl")
    if rows:
        payload["summaries"].append(summarize(name, rows))

(out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(out_dir / "summary.json")
PY
}

run_dataset() {
  local label="$1"
  local gguf="$2"
  local dataset="$3"
  local questions="$4"
  local out_dir="$OUTPUT_ROOT/${label}"
  local output="$out_dir/${dataset}_answers.jsonl"

  mkdir -p "$out_dir"
  echo "[eval] $label $dataset"
  cmd=(
    "$PYTHON_BIN" scripts/run_cpu_inference.py
    --runtime-config "$RUNTIME_CONFIG"
    --backend llama.cpp
    --gguf-model "$gguf"
    --device cpu
    --llama-gpu-layers 0
    --threads "$THREADS"
    --ctx-size "$CTX_SIZE"
    --max-tokens "$MAX_TOKENS"
    --max-retrieval-rounds "$MAX_RETRIEVAL_ROUNDS"
    --questions-file "$questions"
    --batch-output "$output"
  )
  if [[ "$NO_RERANKER" == "1" ]]; then
    cmd+=(--no-reranker)
  fi
  if [[ "$DISABLE_WEB_CONTEXT" == "1" ]]; then
    cmd+=(--disable-web-context)
  fi
  if [[ -n "$LLAMA_DEVICE" ]]; then
    cmd+=(--llama-device "$LLAMA_DEVICE")
  fi
  "${cmd[@]}"
}

run_arm() {
  local label="$1"
  local gguf="$2"
  if [[ "$RUN_EVAL50" == "1" ]]; then
    require_path "eval50 questions" "$EVAL50_QUESTIONS"
    run_dataset "$label" "$gguf" "eval50" "$EVAL50_QUESTIONS"
  fi
  if [[ "$RUN_HARD" == "1" ]]; then
    require_path "hard questions" "$HARD_QUESTIONS"
    run_dataset "$label" "$gguf" "hard10" "$HARD_QUESTIONS"
  fi
  summarize_output "$OUTPUT_ROOT/${label}"
}

RAW_GGUF="$WORK_DIR/raw_soda_550-q4_k_m.gguf"
VERIFIER_GGUF="$WORK_DIR/verifier_soda_v2-q4_k_m.gguf"

if [[ "$RUN_EXPORT" == "1" ]]; then
  if [[ "$RUN_RAW" == "1" ]]; then
    RAW_GGUF="$(export_q4 raw_soda_550 "$RAW_LORA" | tail -n 1)"
  fi
  if [[ "$RUN_VERIFIER" == "1" ]]; then
    VERIFIER_GGUF="$(export_q4 verifier_soda_v2 "$VERIFIER_LORA" | tail -n 1)"
  fi
fi

if [[ "$RUN_RAW" == "1" ]]; then
  require_path "raw q4 gguf" "$RAW_GGUF"
  run_arm "raw_soda_550_cpu_gguf" "$RAW_GGUF"
fi

if [[ "$RUN_VERIFIER" == "1" ]]; then
  require_path "verifier q4 gguf" "$VERIFIER_GGUF"
  run_arm "verifier_soda_v2_cpu_gguf" "$VERIFIER_GGUF"
fi

"$PYTHON_BIN" scripts/analyze_soda_verifier_ablation.py
echo "[done] cpu gguf ablation output: $OUTPUT_ROOT"
