#!/usr/bin/env bash
set -euo pipefail

root=/mnt/store/zhb/exx_grounding_v1
run=exx_grounding_v3_sft_minimal559_glm53_full_a100_r16_lr3e6_e1_20260903
config=$root/tools/llama_factory_exx_grounding_v3_sft_minimal559_glm53_full_a100_20260903.yaml
data=$root/data/exx_grounding_v3_sft_minimal559_glm53_20260903
base=/mnt/store/zhb/ASA-ArknightStoryAgent/model/qwen3.5-4b
baseline=$root/models/exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827
output=$root/models/$run
log_dir=$root/logs/$run
eval_dir=$root/eval/${run}_val79
tools=$root/tools
python_bin=/mnt/store/zhb/asa_train/env/train/bin/python
python_path=/mnt/store/zhb/asa_train/env/lf_overlay

mkdir -p "$log_dir"
exec >>"$log_dir/launcher.log" 2>&1
echo "launcher_started=$(date -Is)"

for required in \
  "$config" \
  "$data/train.json" \
  "$data/val.json" \
  "$data/dataset_info.json" \
  "$base/config.json" \
  "$baseline/adapter_model.safetensors" \
  "$tools/generate_exx_predictions_transformers.py" \
  "$tools/evaluate_exx_outputs.py"; do
  if [[ ! -e "$required" ]]; then
    echo "missing_required=$required"
    exit 1
  fi
done
if [[ -d "$output" ]] && find "$output" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing_to_overwrite=$output"
  exit 1
fi
free_kib=$(df --output=avail "$root" | tail -n1 | tr -d ' ')
if [[ ! "$free_kib" =~ ^[0-9]+$ || "$free_kib" -lt 4194304 ]]; then
  echo "insufficient_mounted_disk_space free_kib=${free_kib:-unknown}"
  exit 1
fi

echo "waiting_for_exclusive_a100=1"
consecutive_free=0
while (( consecutive_free < 3 )); do
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |
    sed '/^[[:space:]]*$/d' || true)
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
    head -n1 | tr -d ' ')
  if [[ -z "$compute_pids" && "$free_mib" =~ ^[0-9]+$ && "$free_mib" -ge 39000 ]]; then
    consecutive_free=$((consecutive_free + 1))
  else
    consecutive_free=0
  fi
  echo "$(date -Is) free_mib=${free_mib:-unknown} compute_pids=${compute_pids//$'\n'/,} stable=$consecutive_free/3"
  if (( consecutive_free < 3 )); then
    sleep 60
  fi
done

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$python_path"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "training_started=$(date -Is)"
set +e
"$python_bin" -m llamafactory.cli train "$config" >"$log_dir/train.stdout.log" \
  2>"$log_dir/train.stderr.log"
train_status=$?
set -e
printf '%s\n' "$train_status" >"$log_dir/TRAIN_EXIT_STATUS"
if [[ "$train_status" -ne 0 || ! -f "$output/adapter_model.safetensors" ]]; then
  echo "training_failed=$(date -Is) status=$train_status"
  exit "$train_status"
fi
date -Is >"$log_dir/TRAIN_COMPLETE"

if [[ -d "$eval_dir" ]] && find "$eval_dir" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing_to_overwrite_eval=$eval_dir"
  exit 1
fi
mkdir -p "$eval_dir"
gold=$data/val.json
for name in baseline full_v3; do
  adapter=$baseline
  if [[ "$name" == "full_v3" ]]; then
    adapter=$output
  fi
  "$python_bin" "$tools/generate_exx_predictions_transformers.py" \
    --gold "$gold" \
    --base-model "$base" \
    --adapter "$adapter" \
    --output "$eval_dir/$name.predictions.json" \
    --max-new-tokens 768 \
    >"$eval_dir/$name.generate.log" 2>&1
  "$python_bin" "$tools/evaluate_exx_outputs.py" \
    --predictions "$eval_dir/$name.predictions.json" \
    --gold "$gold" \
    --output "$eval_dir/$name.metrics.json" \
    >"$eval_dir/$name.evaluate.log" 2>&1
done
date -Is >"$eval_dir/COMPLETE"
echo "all_complete=$(date -Is)"
