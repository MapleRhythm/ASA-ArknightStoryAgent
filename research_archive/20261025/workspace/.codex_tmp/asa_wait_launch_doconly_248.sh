#!/usr/bin/env bash
set -euo pipefail

STATUS=${REMOTE_STORE}/rank_v8_coverage_241_mirror/logs/doconly_waiter_248.status
TASK=${REMOTE_HOME}/asa_rerank_doconly_1024_b24_248.sh
OUT=${REMOTE_STORE}/rank_v8_coverage_241_mirror/qwen3_embedding_4b_ablation_20260825/rerank_fusion

while true; do
  if [[ -s "$OUT/easy200_tuned_dw125_q40_80_doconly_ml1024_b24.json" ]]; then
    printf '%s already-complete\n' "$(date -Is)" >>"$STATUS"
    exit 0
  fi
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
  compute_count=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)
  printf '%s used_mib=%s compute_count=%s\n' "$(date -Is)" "$used" "$compute_count" >"$STATUS"
  if [[ "$used" -lt 1024 && "$compute_count" -eq 0 ]]; then
    printf '%s launch\n' "$(date -Is)" >>"$STATUS"
    exec bash "$TASK"
  fi
  sleep 30
done
