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
# Keep this script deterministic. Use TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 if
# you intentionally want a different GPU set.
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"

cd "$ROOT_DIR"

BASE_MODEL="model/qwen3.5-4b"
SFT_LORA="model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_fix1_qwen35_4b_lr3e5_epoch1"
MERGED_SFT_MODEL="model/merged/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_fix1_qwen35_4b_lr3e5_epoch1_merged"

if [[ ! -f "$MERGED_SFT_MODEL/config.json" ]]; then
  echo "[merge] building merged SFT base at $MERGED_SFT_MODEL"
  "$PYTHON_BIN" scripts/merge_lora_to_base.py \
    --base-model "$BASE_MODEL" \
    --lora-path "$SFT_LORA" \
    --output-dir "$MERGED_SFT_MODEL" \
    --dtype float16 \
    --device cpu \
    --max-shard-size 5GB
fi

"$PYTHON_BIN" scripts/build_opd_kto_dataset_from_scores.py \
  --candidates data/processed/opd_candidates/qwen35_4b_full_chain_sample500/candidates.jsonl \
  --scores data/processed/opd_teacher_scores/qwen35_4b_full_chain_sample500_deepseek/scores.jsonl \
  --output-dir data/processed/llama_factory/opd_kto_full_chain_sample500_deepseek_v1 \
  --val-ratio 0.08 \
  --max-evidence-items 3 \
  --max-evidence-chars 360

"$PYTHON_BIN" -m llamafactory.cli train \
  src/config/llama_factory_opd_kto_full_chain_sample500_config.yaml
