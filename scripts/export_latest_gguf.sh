#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/qwen3.5-4b}"
LORA_PATH="${LORA_PATH:-$ROOT_DIR/model/lora/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1_qwen35_4b_lr3e5_epoch1}"
MERGED_DIR="${MERGED_DIR:-$ROOT_DIR/model/merged/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1_qwen35_4b_lr3e5_epoch1}"
F16_GGUF="${F16_GGUF:-$ROOT_DIR/model/gguf/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1_qwen35_4b_lr3e5_epoch1-merged-f16.gguf}"
Q4_GGUF="${Q4_GGUF:-$ROOT_DIR/model/gguf/teacher_online_chain_short_prompt_v1_ds_flash_800_no_kaltsit_clean_v1_qwen35_4b_lr3e5_epoch1-merged-q4_k_m.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"

mkdir -p "$(dirname "$MERGED_DIR")" "$(dirname "$F16_GGUF")"

if [[ -d "$PYTHON_OVERLAY_DIR" ]]; then
  export PYTHONPATH="$PYTHON_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Step 1/3: merge LoRA into base model"
"$PYTHON_BIN" "$ROOT_DIR/scripts/merge_lora_to_base.py" \
  --base-model "$BASE_MODEL" \
  --lora-path "$LORA_PATH" \
  --output-dir "$MERGED_DIR" \
  --dtype float16 \
  --device cpu

echo "Step 2/3: export merged Hugging Face model to f16 GGUF"
"$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

echo "Step 3/3: quantize f16 GGUF to q4_k_m"
"$QUANTIZE_BIN" "$F16_GGUF" "$Q4_GGUF" Q4_K_M

echo
echo "Export completed."
echo "Merged HF model: $MERGED_DIR"
echo "F16 GGUF:       $F16_GGUF"
echo "Q4_K_M GGUF:    $Q4_GGUF"
