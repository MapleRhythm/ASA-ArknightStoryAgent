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

export PYTHON_BIN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
export ANSWER_ONLY=1

cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/evaluate_action_target_bufan_regression.py \
  --pipeline scripts/run_action_target_kto_full_pipeline.sh \
  --output outputs/eval_action_target_bufan_kto_v1.json \
  --python-bin "$PYTHON_BIN" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "$@"
