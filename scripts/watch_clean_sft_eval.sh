#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 TRAIN_PID [RUN_NAME]" >&2
  exit 2
fi

train_pid=$1
root=/mnt/store/zhb/exx_grounding_v1
run=${2:-exx_binding_clean_sft_v2_a100_r16_lr2e6_e1_20260904}
model=$root/models/$run
log_dir=$root/logs/$run
eval_dir=$root/eval/${run}_val
base=/mnt/store/zhb/ASA-ArknightStoryAgent/model/qwen3.5-4b
baseline=$root/models/exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827
gold=$root/data/exx_binding_clean_sft_v2_20260904/val.json
evaluator=$root/tools/evaluate_exx_outputs.py
vllm_generator=$root/tools/generate_exx_predictions_vllm.py
raw_dir=$eval_dir/vllm_outputs

mkdir -p "$eval_dir"
exec >>"$log_dir/eval_watcher.log" 2>&1
echo "watch_started=$(date -Is) train_pid=$train_pid"
while kill -0 "$train_pid" 2>/dev/null; do
  sleep 30
done
echo "train_process_ended=$(date -Is)"

if [[ ! -f "$model/adapter_model.safetensors" ]]; then
  echo "missing_final_adapter=$model" | tee "$eval_dir/FAILED"
  exit 1
fi
if [[ -e "$eval_dir/COMPLETE" ]]; then
  echo "refusing_existing_complete=$eval_dir"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
python_bin=/mnt/store/zhb/asa_train/env/reasoning/bin/python

if [[ -e "$raw_dir" && -n "$(find "$raw_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing_existing_vllm_output=$raw_dir"
  exit 1
fi

"$python_bin" "$vllm_generator" \
  --gold "$gold" \
  --base-model "$base" \
  --adapter "baseline=$baseline" \
  --adapter "clean_sft=$model" \
  --output-dir "$raw_dir" \
  --max-model-len 12000 \
  --max-tokens 768 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  >"$eval_dir/vllm.generate.log" 2>&1

for name in baseline clean_sft; do
  "$python_bin" "$evaluator" \
    --predictions "$raw_dir/$name.predictions.json" \
    --gold "$gold" \
    --output "$eval_dir/$name.metrics.json" \
    >"$eval_dir/$name.evaluate.log" 2>&1
done
date -Is >"$eval_dir/COMPLETE"
echo "evaluation_complete=$(cat "$eval_dir/COMPLETE")"
