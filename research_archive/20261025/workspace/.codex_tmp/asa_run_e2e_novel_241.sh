#!/usr/bin/env bash
set -euo pipefail

PROJECT=${REMOTE_HOME}/ASA-ArknightStoryAgent
RUN=${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_novel_questions_20260825
PY=${REMOTE_HOME}/miniconda3/envs/train/bin/python

source ${REMOTE_HOME}/.deepseek_env
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PROJECT/.python_packages/train:$PROJECT/src"

cd "$PROJECT"
exec "$PY" api-mode/run_api_inference.py \
  --runtime-config "$RUN/runtime.json" \
  --no-save-run \
  <"$RUN/questions.txt" \
  >"$RUN/raw_results.stdout" \
  2>"$RUN/inference.log"
