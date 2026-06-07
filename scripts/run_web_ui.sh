#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_web_ui.py" "$@"
