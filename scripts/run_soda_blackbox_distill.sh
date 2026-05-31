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

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running SODA distillation.}"

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR="${SODA_OUT_DIR:-data/processed/llama_factory/soda_blackbox_deepseek_v1}"
SAMPLE="${SODA_SAMPLE:-200}"
SAMPLE_OFFSET="${SODA_SAMPLE_OFFSET:-0}"
SEED="${SODA_SEED:-20260529}"
MAX_ROUNDS="${SODA_MAX_ROUNDS:-2}"
GEN_CUDA_VISIBLE_DEVICES="${GEN_CUDA_VISIBLE_DEVICES:-0}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.50}"
CONFIG_PATH="${SODA_TRAIN_CONFIG:-src/config/llama_factory_soda_blackbox_deepseek_v1_config.yaml}"

cd "$ROOT_DIR"

CUDA_VISIBLE_DEVICES="$GEN_CUDA_VISIBLE_DEVICES" \
"$PYTHON_BIN" scripts/generate_soda_blackbox_distillation.py \
  --output-dir "$OUT_DIR" \
  --runtime-config configs/runtime_inference_gpu.json \
  --teacher-runtime-config api-mode/runtime_deepseek_api.json \
  --sample "$SAMPLE" \
  --sample-offset "$SAMPLE_OFFSET" \
  --seed "$SEED" \
  --max-rounds "$MAX_ROUNDS" \
  --device cuda \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
  --skip-existing

CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
"$PYTHON_BIN" -m llamafactory.cli train "$CONFIG_PATH"
