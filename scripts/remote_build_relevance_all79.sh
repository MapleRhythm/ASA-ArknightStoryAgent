#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/store/zhb/exx_grounding_v1
EVAL="$ROOT/eval/set_audit_v3_all79_20260906"
PRED="$ROOT/eval/exx_binding_gap_mix_v3c_a100_20260905/vllm_outputs"
CODE="$ROOT/code/set_audit_v2_20260905/build_relevance_clean_exx_data.py"

while [[ ! -s "$EVAL/gap_mix.json" ]]; do
  sleep 30
done

for model in clean_sft gap_mix; do
  out="$ROOT/data/relevance_clean_${model}_all79_20260906"
  if [[ ! -s "$out/report.json" ]]; then
    python3 "$CODE" \
      --predictions "$PRED/${model}.predictions.json" \
      --audit "$EVAL/${model}.json" \
      --out-dir "$out"
  fi
done
