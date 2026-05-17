#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_gpu_reranker_qwen35_4b.json}"

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

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_cpu_inference.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --device cuda \
  "${EXTRA_ARGS[@]}" \
  "$@"
