#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_gpu_reranker_qwen35_4b.json}"

SERVICE_ARGS=()
if [[ "${ASA_PERSISTENT_SERVICE:-0}" == "1" ]]; then
  SERVICE_ARGS=(--stdio-jsonl)
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_cpu_inference.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --backend vllm \
  --device cuda \
  "${SERVICE_ARGS[@]}" \
  "$@"
