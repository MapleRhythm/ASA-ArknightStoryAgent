#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"
QUESTION="${1:-真龙为什么要启动不反？}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-$ROOT_DIR/api-mode/runtime_deepseek_api.json}"

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this script.}"

cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/.python_packages/train:${ROOT_DIR}/src:${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" api-mode/run_api_inference.py \
  --runtime-config "$RUNTIME_CONFIG" \
  --answer-only \
  "$QUESTION"
