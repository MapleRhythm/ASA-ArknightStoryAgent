#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_cpu_api_no_reranker.json}"

exec "$PYTHON_BIN" "$ROOT_DIR/api-mode/run_api_inference.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --device cpu \
  --no-reranker \
  "$@"
