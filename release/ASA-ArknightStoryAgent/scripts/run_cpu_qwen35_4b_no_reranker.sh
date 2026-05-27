#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/configs/runtime_cpu_qwen35_4b_no_reranker.json}"
DEFAULT_LLAMA_COMPLETION="$ROOT_DIR/third_party/llama.cpp/build-cpu/bin/llama-completion"
if [[ ! -x "$DEFAULT_LLAMA_COMPLETION" && -x "$ROOT_DIR/third_party/llama.cpp/build/bin/llama-completion" ]]; then
  DEFAULT_LLAMA_COMPLETION="$ROOT_DIR/third_party/llama.cpp/build/bin/llama-completion"
fi
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$DEFAULT_LLAMA_COMPLETION}"
LLAMA_BIN_DIR="$(dirname "$LLAMA_COMPLETION")"

if [[ -f "$LLAMA_COMPLETION" && ! -x "$LLAMA_COMPLETION" ]]; then
  chmod +x "$LLAMA_COMPLETION" 2>/dev/null || {
    cat >&2 <<MSG
[run] llama.cpp binary exists but is not executable:
  $LLAMA_COMPLETION

Please fix permissions:
  chmod +x "$LLAMA_COMPLETION"

If it is owned by root:
  sudo chown -R "\$USER:\$USER" "$ROOT_DIR"
  chmod +x "$LLAMA_COMPLETION"
MSG
    exit 1
  }
fi

if [[ -d "$LLAMA_BIN_DIR" ]]; then
  export LD_LIBRARY_PATH="$LLAMA_BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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

SERVICE_ARGS=()
if [[ "${ASA_PERSISTENT_SERVICE:-0}" == "1" ]]; then
  SERVICE_ARGS=(--stdio-jsonl)
fi

exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_cpu_inference.py" \
  --runtime-config "$RUNTIME_CONFIG" \
  --llama-cli "$LLAMA_COMPLETION" \
  --backend llama.cpp \
  --device cpu \
  --no-reranker \
  "${SERVICE_ARGS[@]}" \
  "$@"
