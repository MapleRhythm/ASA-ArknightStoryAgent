#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
MERGE_DEVICE="${MERGE_DEVICE:-cpu}"
BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/qwen3.5-4b}"
INTERMEDIATE_LORA_PATH="${INTERMEDIATE_LORA_PATH:-$ROOT_DIR/model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_latest_kto_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff5632_filtered}"
INTERMEDIATE_MERGED_DIR="${INTERMEDIATE_MERGED_DIR:-$ROOT_DIR/model/merged/soda_targeted_human_20260606_v3_200_current_chain_cutoff5632_kto_merged_text}"
LORA_PATH="${LORA_PATH:-$ROOT_DIR/model/lora/soda_targeted_human_20260606_v3_200_current_chain_from_mergedbase_qwen35_4b_lr8e7_beta001_epoch2_rank8_cutoff6656_filtered}"
MERGED_DIR="${MERGED_DIR:-$ROOT_DIR/model/merged/asa-arknightstoryagent-4b-lora-merged-cutoff6656}"
F16_GGUF="${F16_GGUF:-$ROOT_DIR/model/gguf/qwen3.5-4b-lora-merged-f16.gguf}"
Q4_GGUF="${Q4_GGUF:-$ROOT_DIR/model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/third_party/llama.cpp}"
QUANTIZE_BIN="${QUANTIZE_BIN:-$LLAMA_CPP_DIR/build/bin/llama-quantize}"

prepare_output_dir() {
  local label="$1"
  local dir="$2"

  if [[ ! -d "$dir" ]]; then
    return
  fi

  if [[ "${OVERWRITE_MERGED:-0}" != "1" ]]; then
    echo "$label already exists: $dir" >&2
    echo "Set OVERWRITE_MERGED=1 to rebuild generated merge directories." >&2
    exit 1
  fi

  case "$dir" in
    "$ROOT_DIR"/model/merged/*)
      rm -rf "$dir"
      ;;
    *)
      echo "Refusing to remove nonstandard merge directory: $dir" >&2
      exit 1
      ;;
  esac
}

require_path() {
  local label="$1"
  local path="$2"

  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_path "base model" "$BASE_MODEL"
require_path "intermediate LoRA" "$INTERMEDIATE_LORA_PATH"
require_path "release LoRA" "$LORA_PATH"
require_path "GGUF converter" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py"
require_path "llama.cpp quantizer" "$QUANTIZE_BIN"

mkdir -p "$(dirname "$INTERMEDIATE_MERGED_DIR")" "$(dirname "$MERGED_DIR")" "$(dirname "$F16_GGUF")"

if [[ -d "$PYTHON_OVERLAY_DIR" ]]; then
  export PYTHONPATH="$PYTHON_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Step 1/4: build intermediate merged base"
if [[ "${SKIP_INTERMEDIATE_MERGE:-0}" == "1" && -d "$INTERMEDIATE_MERGED_DIR" ]]; then
  echo "Skipping intermediate merge: $INTERMEDIATE_MERGED_DIR"
else
  prepare_output_dir "Intermediate merged model" "$INTERMEDIATE_MERGED_DIR"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/merge_lora_to_base.py" \
    --base-model "$BASE_MODEL" \
    --lora-path "$INTERMEDIATE_LORA_PATH" \
    --output-dir "$INTERMEDIATE_MERGED_DIR" \
    --dtype float16 \
    --device "$MERGE_DEVICE"
fi

echo "Step 2/4: merge release LoRA into intermediate base"
prepare_output_dir "Merged model" "$MERGED_DIR"
"$PYTHON_BIN" "$ROOT_DIR/scripts/merge_lora_to_base.py" \
  --base-model "$INTERMEDIATE_MERGED_DIR" \
  --lora-path "$LORA_PATH" \
  --output-dir "$MERGED_DIR" \
  --dtype float16 \
  --device "$MERGE_DEVICE"

echo "Step 3/4: export merged Hugging Face model to f16 GGUF"
F16_TMP="${F16_GGUF}.tmp"
rm -f "$F16_TMP"
"$PYTHON_BIN" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_DIR" \
  --outfile "$F16_TMP" \
  --outtype f16
mv "$F16_TMP" "$F16_GGUF"

echo "Step 4/4: quantize f16 GGUF to q4_k_m"
Q4_TMP="${Q4_GGUF}.tmp"
rm -f "$Q4_TMP"
"$QUANTIZE_BIN" "$F16_GGUF" "$Q4_TMP" Q4_K_M
mv "$Q4_TMP" "$Q4_GGUF"

echo
echo "Export completed."
echo "Intermediate HF model: $INTERMEDIATE_MERGED_DIR"
echo "Merged HF model: $MERGED_DIR"
echo "F16 GGUF:       $F16_GGUF"
echo "Q4_K_M GGUF:    $Q4_GGUF"
