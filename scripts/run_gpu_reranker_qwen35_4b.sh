#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_gpu_reranker_qwen35_4b.json}"

# Local development convenience: when this release tree is run from the
# training checkout, reuse the pinned dependency overlay if it exists. A normal
# deployed release should still install requirements.txt in its own venv.
DEV_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
EXTRA_PYTHONPATH=()
if [[ -d "$DEV_ROOT/.python_packages/train" ]]; then
  EXTRA_PYTHONPATH+=("$DEV_ROOT/.python_packages/train")
fi
if [[ -d "$DEV_ROOT/.vendor/train_override" ]]; then
  EXTRA_PYTHONPATH+=("$DEV_ROOT/.vendor/train_override")
fi
if [[ ${#EXTRA_PYTHONPATH[@]} -gt 0 ]]; then
  IFS=:
  export PYTHONPATH="${EXTRA_PYTHONPATH[*]}${PYTHONPATH:+:$PYTHONPATH}"
  unset IFS
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_cpu_inference.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --backend vllm \
  --device cuda \
  "$@"
