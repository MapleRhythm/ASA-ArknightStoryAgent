#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_inference_gpu.json}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"

if [[ -d "$PYTHON_OVERLAY_DIR" ]]; then
  export PYTHONPATH="$PYTHON_OVERLAY_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

DEFAULT_BACKEND="vllm"
if [[ " $* " == *" --gguf-model "* ]] || [[ " $* " == *" --llama-cli "* ]]; then
  DEFAULT_BACKEND="llama.cpp"
fi
if [[ " $* " == *" --backend "* ]]; then
  DEFAULT_BACKEND=""
fi

EXTRA_ARGS=()
if [[ -n "$DEFAULT_BACKEND" ]]; then
  EXTRA_ARGS+=(--backend "$DEFAULT_BACKEND")
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_cpu_inference_merged.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --device cuda \
  "${EXTRA_ARGS[@]}" \
  "$@"
