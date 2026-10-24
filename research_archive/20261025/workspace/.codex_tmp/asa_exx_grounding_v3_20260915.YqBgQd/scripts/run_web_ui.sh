#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_web_ui.py" "$@"
