#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 11 ]]; then
  echo "usage: $0 TRAIN_PID TRAIN_LOG MODEL_DIR PYTHON GENERATOR EVALUATOR GOLD BASE_MODEL SFT_ADAPTER OUTPUT_DIR CUDA_DEVICE" >&2
  exit 2
fi

train_pid=$1
train_log=$2
model_dir=$3
python_bin=$4
generator=$5
evaluator=$6
gold=$7
base_model=$8
sft_adapter=$9
output_dir=${10}
cuda_device=${11}

mkdir -p "$output_dir"
watch_log="$output_dir/watcher.log"
failed_marker="$output_dir/FAILED"
complete_marker="$output_dir/COMPLETE"

exec >>"$watch_log" 2>&1
echo "watch_started=$(date --iso-8601=seconds) train_pid=$train_pid"

while kill -0 "$train_pid" 2>/dev/null; do
  sleep 30
done

echo "train_process_ended=$(date --iso-8601=seconds)"
if [[ ! -f "$model_dir/adapter_model.safetensors" ]] || ! grep -Fq "SAVED $model_dir" "$train_log"; then
  echo "training did not produce a verified final adapter" | tee "$failed_marker"
  exit 1
fi

if [[ -e "$output_dir/sft.predictions.json" || -e "$output_dir/rlvr.predictions.json" ]]; then
  echo "refusing to overwrite existing predictions" | tee "$failed_marker"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$cuda_device"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

"$python_bin" "$generator" \
  --gold "$gold" \
  --base-model "$base_model" \
  --adapter "$sft_adapter" \
  --output "$output_dir/sft.predictions.json"
"$python_bin" "$evaluator" \
  --predictions "$output_dir/sft.predictions.json" \
  --gold "$gold" \
  --output "$output_dir/sft.metrics.json"

"$python_bin" "$generator" \
  --gold "$gold" \
  --base-model "$base_model" \
  --adapter "$model_dir" \
  --output "$output_dir/rlvr.predictions.json"
"$python_bin" "$evaluator" \
  --predictions "$output_dir/rlvr.predictions.json" \
  --gold "$gold" \
  --output "$output_dir/rlvr.metrics.json"

date --iso-8601=seconds >"$complete_marker"
echo "evaluation_complete=$(cat "$complete_marker")"
