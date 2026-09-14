#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

exec "$PYTHON_BIN" "$ROOT_DIR/api-mode/run_api_inference.py" \
  --runtime-config "$ROOT_DIR/configs/runtime_api_stable.json" \
  --no-save-run "$@"
