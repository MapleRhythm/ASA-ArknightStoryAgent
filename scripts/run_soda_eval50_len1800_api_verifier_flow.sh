#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
else
  PYTHON_BIN="python"
fi

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this flow.}"

cd "$ROOT_DIR"
mkdir -p logs outputs/eval_multiround_retrieval outputs/soda_flow_reports

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

QUESTIONS_FILE="${SODA_FLOW_QUESTIONS_FILE:-data/processed/eval50_recall_questions_for_soda.jsonl}"
LISTWISE_FILE="${SODA_FLOW_LISTWISE_FILE:-data/processed/eval50_recall_questions_listwise.jsonl}"
ROLLOUT_DIR="${SODA_FLOW_ROLLOUT_DIR:-data/processed/llama_factory/soda_eval50_len1800_blackbox_v1}"
VERIFIER_DIR="${SODA_FLOW_VERIFIER_DIR:-data/processed/llama_factory/soda_eval50_len1800_api_verifier_v1}"
RUNTIME_CONFIG="${SODA_FLOW_RUNTIME_CONFIG:-configs/runtime_inference_gpu.json}"
TEACHER_RUNTIME_CONFIG="${SODA_FLOW_TEACHER_RUNTIME_CONFIG:-api-mode/runtime_deepseek_api.json}"
DISABLE_WEB_CONTEXT="${SODA_FLOW_DISABLE_WEB_CONTEXT:-1}"
LOG_PREFIX="${SODA_FLOW_LOG_PREFIX:-soda_eval50_len1800}"
REPORT_PREFIX="${SODA_FLOW_REPORT_PREFIX:-eval50_len1800}"
SEED="${SODA_FLOW_SEED:-20260531}"
MAX_ROUNDS="${SODA_FLOW_MAX_ROUNDS:-2}"
FLOW_LIMIT="${SODA_FLOW_LIMIT-2}"
VERIFIER_LIMIT="${SODA_FLOW_VERIFIER_LIMIT:-}"
TEACHER_FULL_CHAIN_LIMIT="${SODA_FLOW_TEACHER_FULL_CHAIN_LIMIT:-}"
GEN_CUDA_VISIBLE_DEVICES="${SODA_FLOW_GEN_CUDA_VISIBLE_DEVICES:-0}"
GEN_GPU_MEMORY_UTILIZATION="${SODA_FLOW_GPU_MEMORY_UTILIZATION:-0.45}"
FLOW_DEVICE="${SODA_FLOW_DEVICE:-cuda}"
STUDENT_BACKEND="${SODA_FLOW_STUDENT_BACKEND:-}"
STUDENT_BASE_MODEL="${SODA_FLOW_BASE_MODEL:-}"
STUDENT_LORA_PATH="${SODA_FLOW_LORA_PATH:-}"
STUDENT_NO_LORA="${SODA_FLOW_NO_LORA:-0}"
STUDENT_GGUF_MODEL="${SODA_FLOW_GGUF_MODEL:-}"
STUDENT_LORA_GGUF="${SODA_FLOW_LORA_GGUF:-}"
STUDENT_LLAMA_CLI="${SODA_FLOW_LLAMA_CLI:-}"
STUDENT_THREADS="${SODA_FLOW_THREADS:-}"
STUDENT_LLAMA_DEVICE="${SODA_FLOW_LLAMA_DEVICE:-}"
STUDENT_LLAMA_GPU_LAYERS="${SODA_FLOW_LLAMA_GPU_LAYERS:-}"
STUDENT_CTX_SIZE="${SODA_FLOW_CTX_SIZE:-}"
STUDENT_MAX_TOKENS="${SODA_FLOW_MAX_TOKENS:-}"
STUDENT_NO_RERANKER="${SODA_FLOW_NO_RERANKER:-0}"

RUN_RECALL_EVAL="${SODA_FLOW_RUN_RECALL_EVAL:-0}"
RUN_ROLLOUT="${SODA_FLOW_RUN_ROLLOUT:-1}"
RUN_VERIFIER="${SODA_FLOW_RUN_VERIFIER:-1}"
RUN_TEACHER_FULL_CHAIN="${SODA_FLOW_RUN_TEACHER_FULL_CHAIN:-0}"
RUN_AUDIT="${SODA_FLOW_RUN_AUDIT:-1}"

if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "[error] Missing questions file: $QUESTIONS_FILE" >&2
  exit 2
fi
if [[ ! -f "$LISTWISE_FILE" ]]; then
  echo "[error] Missing listwise file: $LISTWISE_FILE" >&2
  exit 2
fi
if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "[error] Missing runtime config: $RUNTIME_CONFIG" >&2
  exit 2
fi

EFFECTIVE_RUNTIME_CONFIG="$RUNTIME_CONFIG"
if [[ "$DISABLE_WEB_CONTEXT" == "1" ]]; then
  EFFECTIVE_RUNTIME_CONFIG="outputs/soda_flow_reports/${REPORT_PREFIX}_$(basename "${RUNTIME_CONFIG%.json}")_soda_noweb.json"
  "$PYTHON_BIN" - "$RUNTIME_CONFIG" "$EFFECTIVE_RUNTIME_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
inference = payload.setdefault("inference", {})
web_context = inference.setdefault("web_context", {})
web_context["enabled"] = False
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  echo "[config] web_context disabled for SODA flow: $EFFECTIVE_RUNTIME_CONFIG"
fi

if [[ "$RUN_RECALL_EVAL" == "1" ]]; then
  echo "[stage] recall_eval"
  recall_cmd=(
    "$PYTHON_BIN" scripts/evaluate_multiround_retrieval_recall.py
    --listwise "$LISTWISE_FILE" \
    --output "outputs/eval_multiround_retrieval/eval50_len1800_full_runtime_round${MAX_ROUNDS}.json" \
    --runtime-config "$EFFECTIVE_RUNTIME_CONFIG" \
    --top-ks 1,5,10,20,50 \
    --max-rounds "$MAX_ROUNDS" \
    --planner-mode conclusion \
    --device "$FLOW_DEVICE" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
    --tag "eval50_len1800_full_runtime_round${MAX_ROUNDS}"
  )
  if [[ -n "$STUDENT_BACKEND" ]]; then recall_cmd+=(--backend "$STUDENT_BACKEND"); fi
  if [[ -n "$STUDENT_BASE_MODEL" ]]; then recall_cmd+=(--base-model "$STUDENT_BASE_MODEL"); fi
  if [[ -n "$STUDENT_LORA_PATH" ]]; then recall_cmd+=(--lora-path "$STUDENT_LORA_PATH"); fi
  if [[ "$STUDENT_NO_LORA" == "1" ]]; then recall_cmd+=(--no-lora); fi
  if [[ -n "$STUDENT_GGUF_MODEL" ]]; then recall_cmd+=(--gguf-model "$STUDENT_GGUF_MODEL"); fi
  if [[ -n "$STUDENT_LORA_GGUF" ]]; then recall_cmd+=(--lora-gguf "$STUDENT_LORA_GGUF"); fi
  if [[ -n "$STUDENT_LLAMA_CLI" ]]; then recall_cmd+=(--llama-cli "$STUDENT_LLAMA_CLI"); fi
  if [[ -n "$STUDENT_THREADS" ]]; then recall_cmd+=(--threads "$STUDENT_THREADS"); fi
  if [[ -n "$STUDENT_LLAMA_DEVICE" ]]; then recall_cmd+=(--llama-device "$STUDENT_LLAMA_DEVICE"); fi
  if [[ -n "$STUDENT_LLAMA_GPU_LAYERS" ]]; then recall_cmd+=(--llama-gpu-layers "$STUDENT_LLAMA_GPU_LAYERS"); fi
  if [[ -n "$STUDENT_CTX_SIZE" ]]; then recall_cmd+=(--ctx-size "$STUDENT_CTX_SIZE"); fi
  if [[ -n "$STUDENT_MAX_TOKENS" ]]; then recall_cmd+=(--max-tokens "$STUDENT_MAX_TOKENS"); fi
  if [[ "$STUDENT_NO_RERANKER" == "1" ]]; then recall_cmd+=(--no-reranker); fi
  CUDA_VISIBLE_DEVICES="$GEN_CUDA_VISIBLE_DEVICES" "${recall_cmd[@]}" \
    2>&1 | tee "logs/${LOG_PREFIX}_full_runtime_round${MAX_ROUNDS}.log"
else
  echo "[skip] recall_eval (set SODA_FLOW_RUN_RECALL_EVAL=1 to run GPU full-runtime eval)"
fi

rollout_cmd=(
  "$PYTHON_BIN" scripts/generate_soda_blackbox_distillation.py
  --output-dir "$ROLLOUT_DIR"
  --runtime-config "$EFFECTIVE_RUNTIME_CONFIG"
  --teacher-runtime-config "$TEACHER_RUNTIME_CONFIG"
  --questions-file "$QUESTIONS_FILE"
  --seed "$SEED"
  --max-rounds "$MAX_ROUNDS"
  --device "$FLOW_DEVICE"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION"
  --skip-existing
)
if [[ -n "$FLOW_LIMIT" ]]; then
  rollout_cmd+=(--limit "$FLOW_LIMIT")
fi
if [[ -n "$STUDENT_BACKEND" ]]; then rollout_cmd+=(--student-backend "$STUDENT_BACKEND"); fi
if [[ -n "$STUDENT_BASE_MODEL" ]]; then rollout_cmd+=(--base-model "$STUDENT_BASE_MODEL"); fi
if [[ -n "$STUDENT_LORA_PATH" ]]; then rollout_cmd+=(--lora-path "$STUDENT_LORA_PATH"); fi
if [[ "$STUDENT_NO_LORA" == "1" ]]; then rollout_cmd+=(--no-lora); fi
if [[ -n "$STUDENT_GGUF_MODEL" ]]; then rollout_cmd+=(--gguf-model "$STUDENT_GGUF_MODEL"); fi
if [[ -n "$STUDENT_LORA_GGUF" ]]; then rollout_cmd+=(--lora-gguf "$STUDENT_LORA_GGUF"); fi
if [[ -n "$STUDENT_LLAMA_CLI" ]]; then rollout_cmd+=(--llama-cli "$STUDENT_LLAMA_CLI"); fi
if [[ -n "$STUDENT_THREADS" ]]; then rollout_cmd+=(--threads "$STUDENT_THREADS"); fi
if [[ -n "$STUDENT_LLAMA_DEVICE" ]]; then rollout_cmd+=(--llama-device "$STUDENT_LLAMA_DEVICE"); fi
if [[ -n "$STUDENT_LLAMA_GPU_LAYERS" ]]; then rollout_cmd+=(--llama-gpu-layers "$STUDENT_LLAMA_GPU_LAYERS"); fi
if [[ -n "$STUDENT_CTX_SIZE" ]]; then rollout_cmd+=(--ctx-size "$STUDENT_CTX_SIZE"); fi
if [[ -n "$STUDENT_MAX_TOKENS" ]]; then rollout_cmd+=(--max-tokens "$STUDENT_MAX_TOKENS"); fi
if [[ "$STUDENT_NO_RERANKER" == "1" ]]; then rollout_cmd+=(--no-reranker); fi

if [[ "$RUN_ROLLOUT" == "1" ]]; then
  echo "[stage] student_rollout_and_teacher_replay"
  CUDA_VISIBLE_DEVICES="$GEN_CUDA_VISIBLE_DEVICES" "${rollout_cmd[@]}" \
    2>&1 | tee "logs/${LOG_PREFIX}_rollout.log"
else
  echo "[skip] student_rollout_and_teacher_replay"
fi

verifier_cmd=(
  "$PYTHON_BIN" scripts/build_soda_api_verifier_dataset.py
  --input-dir "$ROLLOUT_DIR"
  --output-dir "$VERIFIER_DIR"
  --runtime-config "$EFFECTIVE_RUNTIME_CONFIG"
  --teacher-runtime-config "$TEACHER_RUNTIME_CONFIG"
  --keep-unverified-conclusion
)
if [[ -n "$VERIFIER_LIMIT" ]]; then
  verifier_cmd+=(--max-verifier-prompts "$VERIFIER_LIMIT")
fi
if [[ "$RUN_TEACHER_FULL_CHAIN" == "1" ]]; then
  verifier_cmd+=(--run-teacher-full-chain)
fi
if [[ -n "$TEACHER_FULL_CHAIN_LIMIT" ]]; then
  verifier_cmd+=(--max-teacher-full-chain-questions "$TEACHER_FULL_CHAIN_LIMIT")
fi

if [[ "$RUN_VERIFIER" == "1" ]]; then
  if [[ ! -s "$ROLLOUT_DIR/audit_records.jsonl" ]]; then
    echo "[error] Missing or empty rollout audit records: $ROLLOUT_DIR/audit_records.jsonl" >&2
    if [[ -f "$ROLLOUT_DIR/build_summary.json" ]]; then
      "$PYTHON_BIN" - "$ROLLOUT_DIR/build_summary.json" <<'PY' >&2
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text(encoding="utf-8"))
print("[rollout-summary]", json.dumps(summary, ensure_ascii=False, indent=2))
PY
    fi
    echo "[hint] Fix student rollout first, or set SODA_FLOW_RUN_VERIFIER=0 to skip relabeling." >&2
    exit 3
  fi
  echo "[stage] api_evidence_only_verifier_and_relabel"
  CUDA_VISIBLE_DEVICES="$GEN_CUDA_VISIBLE_DEVICES" "${verifier_cmd[@]}" \
    2>&1 | tee "logs/${LOG_PREFIX}_api_verifier.log"
else
  echo "[skip] api_evidence_only_verifier_and_relabel"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  echo "[stage] audit"
  "$PYTHON_BIN" scripts/analyze_soda_api_verifier_dataset.py \
    --dataset-dir "$VERIFIER_DIR" \
    --output "outputs/soda_flow_reports/${REPORT_PREFIX}_api_verifier_audit.md"
  "$PYTHON_BIN" scripts/analyze_soda_gold_evidence_topk.py \
    --audit-records "$ROLLOUT_DIR/audit_records.jsonl" \
    --output "outputs/soda_flow_reports/${REPORT_PREFIX}_gold_topk.json"
else
  echo "[skip] audit"
fi

echo "[done] rollout_dir=$ROLLOUT_DIR"
echo "[done] verifier_dir=$VERIFIER_DIR"
