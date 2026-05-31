#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:-}/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/train/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

BASE_OUT_DIR="${SODA_BASE_OUT_DIR:-data/processed/llama_factory/soda_blackbox_deepseek_v1}"
MERGED_OUT_DIR="${SODA_MERGED_OUT_DIR:-data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel}"
SHARD_PREFIX="${SODA_SHARD_PREFIX:-data/processed/llama_factory/soda_blackbox_deepseek_v1_shard}"
SEED="${SODA_SEED:-20260529}"
MAX_ROUNDS="${SODA_MAX_ROUNDS:-2}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.50}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"
TRAIN_CONFIG="${SODA_TRAIN_CONFIG:-src/config/llama_factory_soda_blackbox_deepseek_v1_550_parallel_config.yaml}"
SKIP_GENERATION="${SODA_SKIP_GENERATION:-0}"
SKIP_TRAIN="${SODA_SKIP_TRAIN:-0}"

cd "$ROOT_DIR"
mkdir -p logs

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$SKIP_GENERATION" != "1" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[error] Please export DEEPSEEK_API_KEY before running." >&2
  exit 2
fi

for required in "$BASE_OUT_DIR/audit_records.jsonl" "${SHARD_PREFIX}_gpu1/build_summary.json" "${SHARD_PREFIX}_gpu2/build_summary.json"; do
  if [[ ! -f "$required" ]]; then
    echo "[error] Missing required completed shard/input: $required" >&2
    exit 2
  fi
done

run_soda_shard() {
  local gpu="$1"
  local shard="$2"
  local sample="$3"
  local offset="$4"
  local out_dir="${SHARD_PREFIX}_${shard}"
  local log_path="logs/soda_550_${shard}.log"

  echo "[shard-start] gpu=${gpu} shard=${shard} sample=${sample} offset=${offset} out=${out_dir}" | tee "$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/generate_soda_blackbox_distillation.py \
    --output-dir "$out_dir" \
    --runtime-config configs/runtime_inference_gpu.json \
    --teacher-runtime-config api-mode/runtime_deepseek_api.json \
    --sample "$sample" \
    --sample-offset "$offset" \
    --seed "$SEED" \
    --max-rounds "$MAX_ROUNDS" \
    --device cuda \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
    --skip-existing \
    2>&1 | tee -a "$log_path"
  echo "[shard-done] shard=${shard}" | tee -a "$log_path"
}

if [[ "$SKIP_GENERATION" != "1" ]]; then
  run_soda_shard 1 gpu0_split_gpu1 84 50 &
  pid1=$!
  run_soda_shard 2 gpu0_split_gpu2 83 134 &
  pid2=$!

  fail=0
  wait "$pid1" || fail=1
  wait "$pid2" || fail=1
  if [[ "$fail" != "0" ]]; then
    echo "[error] At least one split shard failed. Check logs/soda_550_gpu0_split_gpu*.log" >&2
    exit 1
  fi
else
  echo "[skip] generation"
fi

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

checks = {
    "gpu1": "data/processed/llama_factory/soda_blackbox_deepseek_v1_shard_gpu1/build_summary.json",
    "gpu2": "data/processed/llama_factory/soda_blackbox_deepseek_v1_shard_gpu2/build_summary.json",
    "gpu0_split_gpu1": "data/processed/llama_factory/soda_blackbox_deepseek_v1_shard_gpu0_split_gpu1/build_summary.json",
    "gpu0_split_gpu2": "data/processed/llama_factory/soda_blackbox_deepseek_v1_shard_gpu0_split_gpu2/build_summary.json",
}
failed = False
for name, raw_path in checks.items():
    path = Path(raw_path)
    if not path.exists():
        print(f"[error] missing shard summary: {path}")
        failed = True
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    records = int(summary.get("records_total") or 0)
    student_failed = int((summary.get("stats") or {}).get("student_failed") or 0)
    print(f"[shard-check] {name} records={records} student_failed={student_failed}")
    if records <= 0 or student_failed > 0:
        failed = True
if failed:
    raise SystemExit("[error] one or more SODA shards failed; fix failed shard before merge/train")
PY

"$PYTHON_BIN" scripts/merge_soda_blackbox_shards.py \
  --input-dir "$BASE_OUT_DIR" \
  --input-dir "${SHARD_PREFIX}_gpu1" \
  --input-dir "${SHARD_PREFIX}_gpu2" \
  --input-dir "${SHARD_PREFIX}_gpu0_split_gpu1" \
  --input-dir "${SHARD_PREFIX}_gpu0_split_gpu2" \
  --output-dir "$MERGED_OUT_DIR" \
  --seed "$SEED" \
  --val-ratio 0.08 \
  --overwrite

sed -n '1,160p' "$MERGED_OUT_DIR/build_summary.json"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m llamafactory.cli train "$TRAIN_CONFIG" \
    2>&1 | tee logs/soda_550_parallel_train.log
else
  echo "[skip] train"
fi
