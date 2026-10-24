#!/usr/bin/env bash
set -euo pipefail

MOUNT_ROOT=${REMOTE_STORAGE}
PROJECT_ROOT="$MOUNT_ROOT/zhb_asa/rank_v8_coverage"
RUN_ROOT="$PROJECT_ROOT/runs/query_answerability_bge_v1_20260821"
PAIR_ROOT="$RUN_ROOT/pairs"
LOG_ROOT="$RUN_ROOT/logs"
MODEL_ROOT="$RUN_ROOT/model/document_bge_v3_5_pairwise"
MANIFEST_ROOT="$RUN_ROOT/manifests"
LABEL_ROOT="$RUN_ROOT/labels"
BASE_MODEL=${REMOTE_HOME}/ASA-ArknightStoryAgent/model/reranker/bge-reranker-v2-m3
ONLINE_MODEL=${REMOTE_HOME}/ASA-ArknightStoryAgent/model/reranker/bge-reranker-v2-m3-rank-mix-v6-small-patch
TRAINER="$PROJECT_ROOT/scripts/train_evidence_chain_reranker.py"
EXPORTER="$PROJECT_ROOT/scripts/export_query_answerability_bge_pairs.py"
EVALUATOR="$PROJECT_ROOT/scripts/evaluate_query_answerability_reranker.py"
AUDITOR="$PROJECT_ROOT/scripts/audit_query_answerability_labels.py"
MATERIALIZER="$PROJECT_ROOT/scripts/materialize_priority_teacher_shards.py"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

DS_ROOT="$PROJECT_ROOT/data/query_answerability_v3/deepseek_flash_high_v3_5"
GLM_ROOT="$PROJECT_ROOT/data/query_answerability_v3/glm_5_3_high_coding_v3_5_remainder"
INPUT_ROOT="$PROJECT_ROOT/data/query_answerability_v3/input"
TRAIN_INPUT="$INPUT_ROOT/train_safe.jsonl"
VAL_INPUT="$INPUT_ROOT/val_safe.jsonl"
COMPLETION_ROOT="$PROJECT_ROOT/data/query_answerability_v3/completion_v3_5"
TRAIN_COMPLETION="$COMPLETION_ROOT/train_teacher.jsonl"
VAL_COMPLETION="$COMPLETION_ROOT/val_teacher.jsonl"
TRAIN_MATERIALIZED="$LABEL_ROOT/train_teacher_priority.jsonl"
VAL_MATERIALIZED="$LABEL_ROOT/val_teacher_priority.jsonl"

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

require_mounted_output_disk() {
  if ! mountpoint -q "$MOUNT_ROOT"; then
    log "refusing to write: output disk is not mounted at $MOUNT_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$PROJECT_ROOT" || ! -w "$PROJECT_ROOT" ]]; then
    log "refusing to write: mounted project root is absent or not writable: $PROJECT_ROOT" >&2
    exit 1
  fi
}

# This check must precede mkdir.  Otherwise a temporarily missing mount would
# silently create the same path on the system disk and training could fill it.
require_mounted_output_disk
mkdir -p \
  "$PAIR_ROOT" "$LOG_ROOT" "$MODEL_ROOT" "$MANIFEST_ROOT" "$RUN_ROOT/eval" \
  "$LABEL_ROOT" \
  "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/triton" "$RUN_ROOT/tmp"

# Keep framework caches, temporary files and optional tracker state on the
# mounted output disk as well.  The source model may remain read-only in /home.
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$RUN_ROOT/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$RUN_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TMPDIR="$RUN_ROOT/tmp"
export WANDB_DIR="$RUN_ROOT/logs/wandb"
export TOKENIZERS_PARALLELISM=false

exec 9>"$MANIFEST_ROOT/document_bge_v1.lock"
if ! flock -n 9; then
  echo "another document BGE v1 supervisor owns the lock" >&2
  exit 0
fi

for required in \
  "$PYTHON_BIN" \
  "$EXPORTER" \
  "$AUDITOR" \
  "$MATERIALIZER" \
  "$EVALUATOR" \
  "$TRAINER" \
  "$BASE_MODEL/config.json" \
  "$DS_ROOT/train_teacher.jsonl" \
  "$DS_ROOT/val_teacher.jsonl" \
  "$GLM_ROOT/train_teacher.jsonl" \
  "$GLM_ROOT/val_teacher.jsonl" \
  "$TRAIN_INPUT" \
  "$VAL_INPUT"; do
  if [[ ! -e "$required" ]]; then
    log "missing required path: $required"
    exit 1
  fi
done

TRAIN_TEACHERS=(
  "$DS_ROOT/train_teacher.jsonl"
  "$GLM_ROOT/train_teacher.jsonl"
)
VAL_TEACHERS=(
  "$DS_ROOT/val_teacher.jsonl"
  "$GLM_ROOT/val_teacher.jsonl"
)
if [[ -s "$TRAIN_COMPLETION" ]]; then
  TRAIN_TEACHERS+=("$TRAIN_COMPLETION")
fi
if [[ -s "$VAL_COMPLETION" ]]; then
  VAL_TEACHERS+=("$VAL_COMPLETION")
fi

log "waiting for mounted-label I/O and an unoccupied training GPU to become healthy"
while true; do
  read -r load_one _ < /proc/loadavg
  load_integer=${load_one%%.*}
  label_probe=0
  gpu_probe=0
  selected_gpu=""
  selected_gpu_free=0
  selected_gpu_util=100

  if (( load_integer < 160 )); then
    if timeout 20 "$PYTHON_BIN" -c \
      'from pathlib import Path; p=Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/data/query_answerability_v3/deepseek_flash_high_v3_5/train_teacher.jsonl"); print(len(p.open("rb").readline()))' \
      >/dev/null 2>&1; then
      label_probe=1
    fi
  fi

  while IFS=, read -r gpu_index gpu_free gpu_util; do
    gpu_index=${gpu_index//[[:space:]]/}
    gpu_free=${gpu_free//[[:space:]]/}
    gpu_util=${gpu_util//[[:space:]]/}
    # GPU0 is already running latency-sensitive services.  Select the freest
    # currently idle device among physical GPU1/GPU2 without touching any
    # existing process.
    if [[ "$gpu_index" =~ ^[12]$ && "$gpu_free" =~ ^[0-9]+$ \
      && "$gpu_util" =~ ^[0-9]+$ ]] \
      && (( gpu_free >= 18000 && gpu_util <= 15 && gpu_free > selected_gpu_free )); then
      selected_gpu=$gpu_index
      selected_gpu_free=$gpu_free
      selected_gpu_util=$gpu_util
    fi
  done < <(timeout 20 nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "$selected_gpu" ]]; then
    gpu_probe=1
  fi

  if (( label_probe == 1 && gpu_probe == 1 )); then
    log "health gate passed: load=$load_one physical_GPU=$selected_gpu free_MiB=$selected_gpu_free util=$selected_gpu_util"
    break
  fi
  log "health gate waiting: load=$load_one label_io=$label_probe training_gpu_ready=$gpu_probe"
  sleep 60
done

log "materializing complete whole-query teacher shards by provider priority"
require_mounted_output_disk
train_materialize_args=()
for path in "${TRAIN_TEACHERS[@]}"; do
  train_materialize_args+=(--teacher "$path")
done
val_materialize_args=()
for path in "${VAL_TEACHERS[@]}"; do
  val_materialize_args+=(--teacher "$path")
done
"$PYTHON_BIN" "$MATERIALIZER" \
  --input "$TRAIN_INPUT" \
  "${train_materialize_args[@]}" \
  --output "$TRAIN_MATERIALIZED" \
  --manifest "$MANIFEST_ROOT/train_teacher_materialization.json"
"$PYTHON_BIN" "$MATERIALIZER" \
  --input "$VAL_INPUT" \
  "${val_materialize_args[@]}" \
  --output "$VAL_MATERIALIZED" \
  --manifest "$MANIFEST_ROOT/val_teacher_materialization.json"

log "auditing complete materialized shards and split isolation"
require_mounted_output_disk
"$PYTHON_BIN" "$AUDITOR" \
  --train "$TRAIN_MATERIALIZED" \
  --val "$VAL_MATERIALIZED" \
  --train-input "$TRAIN_INPUT" \
  --val-input "$VAL_INPUT" \
  --report "$RUN_ROOT/manifests/label_audit.json"

log "exporting compact pair files"
require_mounted_output_disk
"$PYTHON_BIN" "$EXPORTER" \
  --input "$TRAIN_MATERIALIZED" \
  --output "$PAIR_ROOT/train_pairs.jsonl"
"$PYTHON_BIN" "$EXPORTER" \
  --input "$VAL_MATERIALIZED" \
  --output "$PAIR_ROOT/val_pairs.jsonl"

log "writing immutable input checksums and run manifest"
require_mounted_output_disk
sha256sum \
  "$DS_ROOT/train_teacher.jsonl" \
  "$DS_ROOT/val_teacher.jsonl" \
  "$GLM_ROOT/train_teacher.jsonl" \
  "$GLM_ROOT/val_teacher.jsonl" \
  "$TRAIN_INPUT" \
  "$VAL_INPUT" \
  "$TRAIN_MATERIALIZED" \
  "$VAL_MATERIALIZED" \
  "$PAIR_ROOT/train_pairs.jsonl" \
  "$PAIR_ROOT/val_pairs.jsonl" \
  >"$MANIFEST_ROOT/input_sha256.txt"
{
  echo "run_id=query_answerability_bge_v1_20260821"
  echo "created_at=$(date --iso-8601=seconds)"
  echo "canonical_host=${REMOTE_HOST}"
  echo "teacher_protocol=query_answerability_v3"
  echo "teacher_prompt_version=query_answerability_v3.5_positive_caps"
  echo "training_scope=document_only"
  echo "base_model=$BASE_MODEL"
  echo "online_model=$ONLINE_MODEL"
  echo "output_model=$MODEL_ROOT"
  echo "physical_gpu=$selected_gpu"
} >"$MANIFEST_ROOT/run.env"

COMMON_TRAIN_ARGS=(
  --train-file "$PAIR_ROOT/train_pairs.jsonl"
  --eval-file "$PAIR_ROOT/val_pairs.jsonl"
  --model-name-or-path "$BASE_MODEL"
  --output-dir "$MODEL_ROOT"
  --max-length 1024
  --learning-rate 6e-6
  --num-train-epochs 1
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 1
  --gradient-accumulation-steps 16
  --warmup-ratio 0.05
  --weight-decay 0.01
  --logging-steps 5
  --eval-steps 50
  --save-steps 50
  --save-total-limit 2
  --dataloader-num-workers 0
  --loss-type softplus
  --answerability-query-type-multiplier 1.0
  --fp16
  --gradient-checkpointing
)

log "validating pair files, tokenizer and training arguments"
require_mounted_output_disk
CUDA_VISIBLE_DEVICES="$selected_gpu" "$PYTHON_BIN" "$TRAINER" \
  "${COMMON_TRAIN_ARGS[@]}" --dry-run \
  >"$LOG_ROOT/train_dry_run.log" 2>&1

log "starting one-epoch document-reranker baseline on physical GPU$selected_gpu"
require_mounted_output_disk
CUDA_VISIBLE_DEVICES="$selected_gpu" "$PYTHON_BIN" "$TRAINER" \
  "${COMMON_TRAIN_ARGS[@]}" \
  >"$LOG_ROOT/train_document_bge_v1.log" 2>&1
log "training completed: $MODEL_ROOT"

log "evaluating base and new models on the complete held-out validation pools"
require_mounted_output_disk
for model_spec in "base:$BASE_MODEL" "new:$MODEL_ROOT"; do
  model_name=${model_spec%%:*}
  model_path=${model_spec#*:}
  CUDA_VISIBLE_DEVICES="$selected_gpu" "$PYTHON_BIN" "$EVALUATOR" \
    --input "$VAL_MATERIALIZED" \
    --model "$model_path" \
    --output "$RUN_ROOT/eval/${model_name}_val_query_answerability.json" \
    --device cuda \
    --max-length 1024 \
    --batch-size 4 \
    --duplicate-policy error \
    --cutoff 1 --cutoff 5 --cutoff 10 --cutoff 32 \
    >"$LOG_ROOT/eval_${model_name}.log" 2>&1
done

if [[ -f "$ONLINE_MODEL/config.json" ]]; then
  CUDA_VISIBLE_DEVICES="$selected_gpu" "$PYTHON_BIN" "$EVALUATOR" \
    --input "$VAL_MATERIALIZED" \
    --model "$ONLINE_MODEL" \
    --output "$RUN_ROOT/eval/online_v6_val_query_answerability.json" \
    --device cuda \
    --max-length 1024 \
    --batch-size 4 \
    --duplicate-policy error \
    --cutoff 1 --cutoff 5 --cutoff 10 --cutoff 32 \
    >"$LOG_ROOT/eval_online_v6.log" 2>&1
fi
log "document-reranker baseline and validation evaluation completed"
