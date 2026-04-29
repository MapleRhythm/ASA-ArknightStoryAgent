#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_OVERLAY_DIR="${PYTHON_OVERLAY_DIR:-$ROOT_DIR/.python_packages/train}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT_DIR/requirements.train.gpu.inference.txt}"

mkdir -p "$PYTHON_OVERLAY_DIR"

echo "Installing vLLM overlay into: $PYTHON_OVERLAY_DIR"
"$PYTHON_BIN" -m pip install --upgrade --target "$PYTHON_OVERLAY_DIR" -r "$REQUIREMENTS_FILE"

cat <<EOF

vLLM overlay install completed.

Current overlay:
  $PYTHON_OVERLAY_DIR

Run GPU inference with:
  bash scripts/run_gpu_inference.sh --backend vllm --answer-only

If you want this overlay to be visible in ad-hoc Python commands too, export:
  export PYTHONPATH="$PYTHON_OVERLAY_DIR:\$PYTHONPATH"
EOF
