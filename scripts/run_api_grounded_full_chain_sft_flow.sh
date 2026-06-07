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
elif [[ -x "/mnt/store/zhb/conda_envs/train/bin/python" ]]; then
  PYTHON_BIN="/mnt/store/zhb/conda_envs/train/bin/python"
else
  PYTHON_BIN="python"
fi

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this flow.}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${RUN_ID:-api_grounded_full_chain_sft_v1}"
QUESTIONS_FILE="${QUESTIONS_FILE:-outputs/eval_soda_api_verifier_v2/eval50_questions.txt}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-configs/runtime_inference_gpu.json}"
TEACHER_RUNTIME_CONFIG="${TEACHER_RUNTIME_CONFIG:-api-mode/runtime_deepseek_api.json}"
WORK_ROOT="${WORK_ROOT:-data/processed/llama_factory/$RUN_ID}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$WORK_ROOT/rollout}"
VERIFIER_DIR="${VERIFIER_DIR:-$WORK_ROOT/api_verifier_kto}"
CONCLUSION_SFT_DIR="${CONCLUSION_SFT_DIR:-$WORK_ROOT/conclusion_policy_sft}"
GROUNDED_ANSWER_SFT_DIR="${GROUNDED_ANSWER_SFT_DIR:-$WORK_ROOT/evidence_grounded_answer_sft}"
GROUNDED_ACTION_SFT_DIR="${GROUNDED_ACTION_SFT_DIR:-$WORK_ROOT/grounded_action_sft}"
REPORT_DIR="${REPORT_DIR:-outputs/soda_flow_reports/$RUN_ID}"
LOG_DIR="${LOG_DIR:-logs/$RUN_ID}"

MAX_ROUNDS="${MAX_ROUNDS:-2}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-20260603}"
VAL_RATIO="${VAL_RATIO:-0.08}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.52}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
STUDENT_LORA_PATH="${STUDENT_LORA_PATH:-model/lora/conclusion_chosen_sft_v1_api_qc_from_schema_sft_qwen35_4b_lr2e6_epoch2_cutoff3072}"
STUDENT_BASE_MODEL="${STUDENT_BASE_MODEL:-model/qwen3.5-4b}"
STUDENT_BACKEND="${STUDENT_BACKEND:-vllm}"
ANSWER_BATCH_SIZE="${ANSWER_BATCH_SIZE:-4}"
ANSWER_MAX_API_EVIDENCE_CHARS="${ANSWER_MAX_API_EVIDENCE_CHARS:-9000}"
ANSWER_MAX_TRAIN_EVIDENCE_CHARS="${ANSWER_MAX_TRAIN_EVIDENCE_CHARS:-12000}"

RUN_ROLLOUT="${RUN_ROLLOUT:-1}"
RUN_VERIFIER="${RUN_VERIFIER:-1}"
RUN_EXTRACT_SFT="${RUN_EXTRACT_SFT:-1}"
RUN_GROUNDED_ANSWER_SFT="${RUN_GROUNDED_ANSWER_SFT:-1}"
RUN_GROUNDED_ACTION_SFT="${RUN_GROUNDED_ACTION_SFT:-1}"
DISABLE_WEB_CONTEXT="${DISABLE_WEB_CONTEXT:-1}"
OVERWRITE_EXTRACT="${OVERWRITE_EXTRACT:-1}"

mkdir -p "$WORK_ROOT" "$REPORT_DIR" "$LOG_DIR"

if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "[error] missing questions file: $QUESTIONS_FILE" >&2
  exit 2
fi
if [[ ! -f "$RUNTIME_CONFIG" ]]; then
  echo "[error] missing runtime config: $RUNTIME_CONFIG" >&2
  exit 2
fi

EFFECTIVE_RUNTIME_CONFIG="$RUNTIME_CONFIG"
if [[ "$DISABLE_WEB_CONTEXT" == "1" ]]; then
  EFFECTIVE_RUNTIME_CONFIG="$REPORT_DIR/runtime_noweb.json"
  "$PYTHON_BIN" - "$RUNTIME_CONFIG" "$EFFECTIVE_RUNTIME_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
payload = json.loads(src.read_text(encoding="utf-8"))
payload.setdefault("inference", {}).setdefault("web_context", {})["enabled"] = False
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
fi

echo "[flow] run_id=$RUN_ID"
echo "[flow] questions=$QUESTIONS_FILE"
echo "[flow] max_rounds=$MAX_ROUNDS"
echo "[flow] runtime=$EFFECTIVE_RUNTIME_CONFIG"
echo "[flow] student_backend=$STUDENT_BACKEND"
echo "[flow] student_base=$STUDENT_BASE_MODEL"
echo "[flow] student_lora=$STUDENT_LORA_PATH"
echo "[flow] rollout_dir=$ROLLOUT_DIR"
echo "[flow] verifier_dir=$VERIFIER_DIR"
echo "[flow] conclusion_sft_dir=$CONCLUSION_SFT_DIR"
echo "[flow] grounded_answer_sft_dir=$GROUNDED_ANSWER_SFT_DIR"
echo "[flow] grounded_action_sft_dir=$GROUNDED_ACTION_SFT_DIR"

rollout_cmd=(
  "$PYTHON_BIN" scripts/generate_soda_blackbox_distillation.py
  --output-dir "$ROLLOUT_DIR"
  --runtime-config "$EFFECTIVE_RUNTIME_CONFIG"
  --teacher-runtime-config "$TEACHER_RUNTIME_CONFIG"
  --questions-file "$QUESTIONS_FILE"
  --seed "$SEED"
  --val-ratio "$VAL_RATIO"
  --max-rounds "$MAX_ROUNDS"
  --device cuda
  --student-backend "$STUDENT_BACKEND"
  --base-model "$STUDENT_BASE_MODEL"
  --lora-path "$STUDENT_LORA_PATH"
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --enforce-eager
  --skip-existing
)
if [[ -n "$LIMIT" ]]; then
  rollout_cmd+=(--limit "$LIMIT")
fi

if [[ "$RUN_ROLLOUT" == "1" ]]; then
  echo "[stage] 1/5 student rollout + teacher replay"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${rollout_cmd[@]}" 2>&1 | tee "$LOG_DIR/rollout.log"
else
  echo "[skip] rollout"
fi

if [[ "$RUN_VERIFIER" == "1" ]]; then
  echo "[stage] 2/5 API evidence-only verifier relabel"
  "$PYTHON_BIN" scripts/build_soda_api_verifier_dataset.py \
    --input-dir "$ROLLOUT_DIR" \
    --output-dir "$VERIFIER_DIR" \
    --dataset-name "${RUN_ID}_api_verifier_kto" \
    --runtime-config "$EFFECTIVE_RUNTIME_CONFIG" \
    --teacher-runtime-config "$TEACHER_RUNTIME_CONFIG" \
    --seed "$SEED" \
    --val-ratio "$VAL_RATIO" \
    --max-rounds "$MAX_ROUNDS" \
    2>&1 | tee "$LOG_DIR/api_verifier.log"
else
  echo "[skip] verifier"
fi

extract_args=()
if [[ "$OVERWRITE_EXTRACT" == "1" ]]; then
  extract_args+=(--overwrite)
fi

if [[ "$RUN_EXTRACT_SFT" == "1" ]]; then
  echo "[stage] 3/5 extract current-schema SFT from verifier chosen records"
  "$PYTHON_BIN" scripts/extract_sft_from_api_verifier_kto.py \
    --input-dir "$VERIFIER_DIR" \
    --output-dir "$CONCLUSION_SFT_DIR" \
    --dataset-name "${RUN_ID}_conclusion_policy_sft" \
    --task-types "user_question_hypothesis_generation,conclusion_generation" \
    --chosen-only \
    "${extract_args[@]}" \
    2>&1 | tee "$LOG_DIR/extract_sft.log"
else
  echo "[skip] extract_sft"
fi

if [[ "$RUN_GROUNDED_ANSWER_SFT" == "1" ]]; then
  echo "[stage] 4/5 build evidence-grounded answer SFT for answer_directly states"
  "$PYTHON_BIN" scripts/build_evidence_grounded_answer_sft_with_api.py \
    --input-dir "$CONCLUSION_SFT_DIR" \
    --output-dir "$GROUNDED_ANSWER_SFT_DIR" \
    --dataset-name "${RUN_ID}_evidence_grounded_answer_sft" \
    --actions answer_directly \
    --no-positive-only \
    --batch-size "$ANSWER_BATCH_SIZE" \
    --max-api-evidence-chars "$ANSWER_MAX_API_EVIDENCE_CHARS" \
    --max-train-evidence-chars "$ANSWER_MAX_TRAIN_EVIDENCE_CHARS" \
    --shuffle \
    2>&1 | tee "$LOG_DIR/grounded_answer_sft.log"
else
  echo "[skip] grounded_answer_sft"
fi

if [[ "$RUN_GROUNDED_ACTION_SFT" == "1" ]]; then
  echo "[stage] 5/5 build no-missing-slots grounded action SFT"
  grounded_action_args=()
  if [[ "$OVERWRITE_EXTRACT" == "1" ]]; then
    grounded_action_args+=(--overwrite)
  fi
  "$PYTHON_BIN" scripts/build_grounded_action_sft_from_policy_and_answer.py \
    --policy-sft-dir "$CONCLUSION_SFT_DIR" \
    --grounded-answer-sft-dir "$GROUNDED_ANSWER_SFT_DIR" \
    --output-dir "$GROUNDED_ACTION_SFT_DIR" \
    --dataset-name "${RUN_ID}_grounded_action_sft" \
    --actions answer_directly,retrieve_more \
    "${grounded_action_args[@]}" \
    2>&1 | tee "$LOG_DIR/grounded_action_sft.log"
else
  echo "[skip] grounded_action_sft"
fi

echo "[done] conclusion policy SFT: $CONCLUSION_SFT_DIR"
echo "[done] evidence-grounded answer SFT: $GROUNDED_ANSWER_SFT_DIR"
echo "[done] final no-missing-slots grounded action SFT: $GROUNDED_ACTION_SFT_DIR"
