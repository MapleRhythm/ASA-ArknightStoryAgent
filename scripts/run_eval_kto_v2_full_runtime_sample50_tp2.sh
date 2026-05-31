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

KTO_LORA="${KTO_LORA:-$ROOT_DIR/model/lora/opd_kto_full_chain_sample500_deepseek_v2_qwen35_4b_lr5e6_beta002_ref_sft_epoch1}"
FULL_RUNTIME="${FULL_RUNTIME:-$ROOT_DIR/model/merged/opd_kto_full_chain_sample500_deepseek_v2_qwen35_4b_full_runtime}"
OUTPUT="${OUTPUT:-$ROOT_DIR/outputs/eval_multiround_retrieval/kto_v2_full_runtime_sample50_tp2.json}"

export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR/.python_packages/train:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1}"

cd "$ROOT_DIR"

if [[ ! -f "$KTO_LORA/adapter_config.json" ]]; then
  echo "KTO LoRA not found: $KTO_LORA" >&2
  exit 1
fi

if [[ ! -f "$FULL_RUNTIME/config.json" || ! -f "$FULL_RUNTIME/export_meta.json" ]]; then
  if [[ -e "$FULL_RUNTIME" && "${EXPORT_OVERWRITE:-0}" != "1" ]]; then
    echo "Full-runtime model path exists but looks incomplete: $FULL_RUNTIME" >&2
    echo "Set EXPORT_OVERWRITE=1 to rebuild it." >&2
    exit 1
  fi

  export_args=()
  if [[ -e "$FULL_RUNTIME" ]]; then
    export_args+=(--overwrite)
  fi

  "$PYTHON_BIN" scripts/export_qwen35_kto_full_runtime_model.py \
    --kto-lora "$KTO_LORA" \
    --output-dir "$FULL_RUNTIME" \
    --dtype float16 \
    --device-map cpu \
    "${export_args[@]}"
fi

"$PYTHON_BIN" scripts/evaluate_multiround_retrieval_recall.py \
  --runtime-config configs/runtime_inference_gpu.json \
  --output "$OUTPUT" \
  --sample 50 \
  --planner-mode conclusion \
  --max-rounds 2 \
  --device cuda:0 \
  --base-model "$FULL_RUNTIME" \
  --no-lora \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.62 \
  --ctx-size 6144 \
  --max-num-batched-tokens 8192 \
  --top-ks 1,5,10,20,50 \
  --progress-every 5 \
  --tag kto_v2_full_runtime_sample50_tp2

echo "Wrote: $OUTPUT"
