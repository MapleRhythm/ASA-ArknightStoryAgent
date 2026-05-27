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

SFT_LORA="${SFT_LORA:-model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1}"
MERGED_SFT_MODEL="${MERGED_SFT_MODEL:-model/merged/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1_merged}"
KTO_CONFIG="src/config/llama_factory_action_target_hard_negative_kto_v1_config.yaml"
KTO_OUTPUT_DIR="$(awk -F': *' '$1 == "output_dir" {print $2; exit}' "$KTO_CONFIG")"
KTO_BASE_MODEL="$(awk -F': *' '$1 == "model_name_or_path" {print $2; exit}' "$KTO_CONFIG")"

if [[ -z "$KTO_OUTPUT_DIR" || -z "$KTO_BASE_MODEL" ]]; then
  echo "Failed to parse model_name_or_path/output_dir from $KTO_CONFIG" >&2
  exit 1
fi

if [[ "$KTO_BASE_MODEL" != "$MERGED_SFT_MODEL" ]]; then
  echo "KTO config base model mismatch:" >&2
  echo "  config: $KTO_BASE_MODEL" >&2
  echo "  script: $MERGED_SFT_MODEL" >&2
  exit 1
fi

if [[ ! -f "$SFT_LORA/adapter_config.json" ]]; then
  echo "SFT LoRA not found: $SFT_LORA" >&2
  exit 1
fi

if [[ ! -f "model/qwen3.5-4b/config.json" ]]; then
  echo "Base model not found: model/qwen3.5-4b" >&2
  exit 1
fi

if [[ -d "$KTO_OUTPUT_DIR" && ! -f "$KTO_OUTPUT_DIR/adapter_config.json" ]]; then
  echo "Found incomplete KTO output dir: $KTO_OUTPUT_DIR" >&2
  echo "Move it away or remove it before training, then rerun this script." >&2
  exit 1
fi
if [[ -f "$KTO_OUTPUT_DIR/adapter_config.json" ]]; then
  echo "KTO adapter already exists: $KTO_OUTPUT_DIR" >&2
  echo "Use scripts/run_action_target_kto_bufan_regression.sh to evaluate it, or move the directory before retraining." >&2
  exit 0
fi

"$PYTHON_BIN" scripts/build_action_target_hard_negative_kto_dataset.py \
  --output-dir data/processed/llama_factory/action_target_hard_negative_kto_v1 \
  --max-auto "${ACTION_TARGET_MAX_AUTO:-0}" \
  --max-strips 3 \
  --max-evidence-chars 260 \
  --val-ratio 0.08

"$PYTHON_BIN" scripts/validate_action_target_kto_dataset.py \
  --dataset-dir data/processed/llama_factory/action_target_hard_negative_kto_v1

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run OK."
  echo "  python: $PYTHON_BIN"
  echo "  cuda: $CUDA_VISIBLE_DEVICES"
  echo "  sft_lora: $SFT_LORA"
  echo "  merged_sft_model: $MERGED_SFT_MODEL"
  echo "  kto_config: $KTO_CONFIG"
  echo "  kto_output_dir: $KTO_OUTPUT_DIR"
  if [[ -f "$MERGED_SFT_MODEL/config.json" ]]; then
    echo "  merge: skip, merged model exists"
  else
    echo "  merge: required before training"
  fi
  exit 0
fi

if [[ ! -f "$MERGED_SFT_MODEL/config.json" ]]; then
  "$PYTHON_BIN" scripts/merge_lora_to_base.py \
    --base-model model/qwen3.5-4b \
    --lora-path "$SFT_LORA" \
    --output-dir "$MERGED_SFT_MODEL" \
    --dtype float16 \
    --device "${MERGE_DEVICE:-cpu}"
fi

"$PYTHON_BIN" -m llamafactory.cli train \
  "$KTO_CONFIG"
