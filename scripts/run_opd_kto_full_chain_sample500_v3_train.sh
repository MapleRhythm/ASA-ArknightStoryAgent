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

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"

cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/build_opd_kto_dataset_from_scores.py \
  --candidates data/processed/opd_candidates/qwen35_4b_full_chain_sample500/candidates.jsonl \
  --scores data/processed/opd_teacher_scores/qwen35_4b_full_chain_sample500_deepseek/scores.jsonl \
  --output-dir data/processed/llama_factory/opd_kto_full_chain_sample500_deepseek_v3 \
  --val-ratio 0.08 \
  --max-evidence-items 3 \
  --max-evidence-chars 360 \
  --negative-ratio-cap 2.0

"$PYTHON_BIN" -m llamafactory.cli train \
  src/config/llama_factory_opd_kto_full_chain_sample500_v3_config.yaml
