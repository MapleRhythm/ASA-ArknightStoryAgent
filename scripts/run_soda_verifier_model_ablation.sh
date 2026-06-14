#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
  PYTHON_BIN="python"
fi

export PYTHON_BIN
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
export GPUS="${GPUS:-0,1}"
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime_inference_gpu.json}"
export EVAL50_QUESTIONS="${EVAL50_QUESTIONS:-outputs/eval_soda_api_verifier_v2/eval50_questions.txt}"
export HARD_QUESTIONS="${HARD_QUESTIONS:-outputs/eval_soda_api_verifier_v2/hard10_questions.txt}"
export MAX_RETRIEVAL_ROUNDS="${MAX_RETRIEVAL_ROUNDS:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.52}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

RUN_RAW="${RUN_RAW:-1}"
RUN_VERIFIER="${RUN_VERIFIER:-1}"
RUN_TEACHER_SCORE="${RUN_TEACHER_SCORE:-0}"

RAW_LORA="${RAW_LORA:-model/lora/soda_blackbox_deepseek_v1_550_parallel_qwen35_4b_lr2e6_beta001_epoch1}"
VERIFIER_LORA="${VERIFIER_LORA:-model/lora/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_qwen35_4b_lr1e6_beta001_epoch3}"
RAW_OUT_DIR="${RAW_OUT_DIR:-outputs/eval50_hard10_raw_soda_550_verifier_ablation}"
VERIFIER_OUT_DIR="${VERIFIER_OUT_DIR:-outputs/eval50_hard10_verifier_soda_v2_ablation}"
TEACHER_SCORE_OUT_DIR="${TEACHER_SCORE_OUT_DIR:-outputs/runtime_teacher_scores/soda_verifier_model_ablation}"
TEACHER_SCORE_DATASETS="${TEACHER_SCORE_DATASETS:-eval50,hard10}"
TEACHER_SCORE_PROMPT_STYLE="${TEACHER_SCORE_PROMPT_STYLE:-compact}"
TEACHER_SCORE_BATCH_SIZE="${TEACHER_SCORE_BATCH_SIZE:-8}"
TEACHER_SCORE_WORKERS="${TEACHER_SCORE_WORKERS:-2}"
TEACHER_SCORE_MAX_ITEMS="${TEACHER_SCORE_MAX_ITEMS:-0}"

check_lora() {
  local path="$1"
  if [[ ! -f "$path/adapter_config.json" ]]; then
    echo "[error] missing LoRA adapter_config.json: $path" >&2
    exit 2
  fi
}

check_lora "$RAW_LORA"
check_lora "$VERIFIER_LORA"

if [[ "$RUN_RAW" == "1" || "$RUN_VERIFIER" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - shell preflight
    print(f"[error] cannot import torch for CUDA preflight: {exc}", file=sys.stderr)
    raise SystemExit(4)

if not torch.cuda.is_available():
    print("[error] CUDA is unavailable; strict model-effect ablation requires vLLM GPU runtime.", file=sys.stderr)
    print("[hint] check nvidia-smi / driver-library compatibility, then rerun this wrapper.", file=sys.stderr)
    raise SystemExit(4)

print(f"[config] cuda_device_count={torch.cuda.device_count()}")

try:
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.lora.request import LoRARequest  # noqa: F401
except Exception as exc:  # pragma: no cover - shell preflight
    print("[error] vLLM runtime imports failed.", file=sys.stderr)
    print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
    print("[hint] install or repair vLLM in PYTHON_BIN environment before running the ablation.", file=sys.stderr)
    raise SystemExit(4)
PY
fi

echo "[config] python=$PYTHON_BIN"
echo "[config] gpus=$GPUS"
echo "[config] runtime_config=$RUNTIME_CONFIG"
echo "[config] eval50_questions=$EVAL50_QUESTIONS"
echo "[config] hard_questions=$HARD_QUESTIONS"

if [[ "$RUN_RAW" == "1" ]]; then
  echo "[stage] raw SODA eval"
  RUN_NAME="eval50_hard10_raw_soda_550_verifier_ablation" \
    OUT_DIR="$RAW_OUT_DIR" \
    LORA_PATH="$RAW_LORA" \
    bash scripts/run_eval50_hard10_gpu_abstain_flow.sh
fi

if [[ "$RUN_VERIFIER" == "1" ]]; then
  echo "[stage] verifier-aware SODA eval"
  RUN_NAME="eval50_hard10_verifier_soda_v2_ablation" \
    OUT_DIR="$VERIFIER_OUT_DIR" \
    LORA_PATH="$VERIFIER_LORA" \
    bash scripts/run_eval50_hard10_gpu_abstain_flow.sh
fi

if [[ "$RUN_TEACHER_SCORE" == "1" ]]; then
  if [[ ! -f "$RAW_OUT_DIR/eval50_answers.jsonl" || ! -f "$VERIFIER_OUT_DIR/eval50_answers.jsonl" ]]; then
    echo "[error] strict eval outputs are incomplete; cannot run teacher scorer" >&2
    echo "[hint] expected: $RAW_OUT_DIR/eval50_answers.jsonl and $VERIFIER_OUT_DIR/eval50_answers.jsonl" >&2
    exit 5
  fi
  echo "[stage] evidence-only teacher scoring"
  score_args=(
    "$PYTHON_BIN" scripts/score_runtime_answers_with_teacher.py
    --run raw_soda "$RAW_OUT_DIR"
    --run verifier_soda "$VERIFIER_OUT_DIR"
    --datasets "$TEACHER_SCORE_DATASETS"
    --output-dir "$TEACHER_SCORE_OUT_DIR"
    --include-all-actions
    --prompt-style "$TEACHER_SCORE_PROMPT_STYLE"
    --batch-size "$TEACHER_SCORE_BATCH_SIZE"
    --workers "$TEACHER_SCORE_WORKERS"
  )
  if [[ "$TEACHER_SCORE_MAX_ITEMS" != "0" ]]; then
    score_args+=(--max-items "$TEACHER_SCORE_MAX_ITEMS")
  fi
  "${score_args[@]}"
fi

echo "[stage] rebuild ablation report"
"$PYTHON_BIN" scripts/analyze_soda_verifier_ablation.py

echo "[done] report=outputs/soda_verifier_ablation_20260609/report.md"
