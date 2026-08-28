#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 TRAIN_PID TRAIN_LOG GOLD BASE_MODEL SFT_ADAPTER RLVR_ADAPTER EVAL_DIR" >&2
  exit 2
fi

train_pid=$1
train_log=$2
gold=$3
base_model=$4
sft_adapter=$5
rlvr_adapter=$6
eval_dir=$7
root=/mnt/store/zhb/exx_grounding_v1
tools=$root/tools

mkdir -p "$eval_dir"
while kill -0 "$train_pid" 2>/dev/null; do
  sleep 60
done

if ! grep -q "SAVED $rlvr_adapter" "$train_log"; then
  echo "training did not finish successfully; evaluation not started" >&2
  exit 1
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  /mnt/store/zhb/asa_train/env/reasoning/bin/python \
  "$tools/generate_exx_predictions_vllm.py" \
  --gold "$gold" \
  --base-model "$base_model" \
  --adapter "sft=$sft_adapter" \
  --adapter "rlvr=$rlvr_adapter" \
  --output-dir "$eval_dir/predictions" \
  --max-model-len 12000 \
  --max-tokens 512 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.88 \
  --enforce-eager

for name in sft rlvr; do
  python3 "$tools/evaluate_exx_outputs_claimcite_v2_20260829.py" \
    --predictions "$eval_dir/predictions/$name.predictions.json" \
    --gold "$gold" \
    --output "$eval_dir/$name.metrics.json" \
    > "$eval_dir/$name.summary.txt"
done

date -Is > "$eval_dir/COMPLETE"
