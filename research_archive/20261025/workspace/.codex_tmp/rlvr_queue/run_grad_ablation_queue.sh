#!/usr/bin/env bash
set -euo pipefail

root=${REMOTE_STORE}/exx_grounding_v1
python_bin=${REMOTE_STORE}/asa_train/env/reasoning/bin/python
code=$root/code/rlvr_f3d23a5
base=${REMOTE_STORE}/ASA-ArknightStoryAgent/model/qwen3.5-4b
sft=$root/models/exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827
gold_train=$root/data/exx_grounding_v2_sft_success559_contiguous_20260828/train.json
gold_val=$root/data/exx_grounding_v2_sft_success559_contiguous_20260828/val.json
run1=exx_grounding_v3_semgate_g4_grad1_warm5_cosine_random120_20260829
run2=exx_grounding_v3_semgate_g4_grad2_warm5_cosine_random120_20260829

pid1=$(<"$root/logs/$run1/train.pid")
while kill -0 "$pid1" 2>/dev/null; do
  sleep 30
done
if ! grep -Fq "SAVED $root/runs/$run1" "$root/logs/$run1/train.log"; then
  echo "grad1 run failed" >&2
  exit 1
fi

out2=$root/runs/$run2
log2=$root/logs/$run2
if [[ ! -e "$out2" && ! -e "$log2" ]]; then
  mkdir -p "$log2"
  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "$python_bin" \
    "$code/scripts/train_exx_grpo_rlvr.py" \
    --train-file "$gold_train" --base-model "$base" --sft-adapter "$sft" \
    --output-dir "$out2" --max-rows 120 --selection-order random \
    --max-prompt-tokens 5000 --num-generations 4 --max-completion-length 384 \
    --learning-rate 5e-7 --beta 0.04 --max-grad-norm 2.0 --warmup-ratio 0.05 \
    --lr-scheduler-type cosine --epochs 1 --batch-size 1 \
    --gradient-accumulation-steps 4 --seed 20260828 --temperature 0.9 \
    --logging-steps 5 --save-steps 0 --reward-profile semantic-gated \
    >"$log2/train.log" 2>&1
fi
if ! grep -Fq "SAVED $out2" "$log2/train.log"; then
  echo "grad2 run failed" >&2
  exit 1
fi

eval_dir=$root/eval/semgate_grad_ablation_random120_val79_20260829
if [[ -e "$eval_dir/predictions/config.json" ]]; then
  echo "refusing to overwrite existing evaluation: $eval_dir" >&2
  exit 1
fi
mkdir -p "$eval_dir"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python_bin" \
  "$root/tools/generate_exx_predictions_vllm.py" \
  --gold "$gold_val" --base-model "$base" \
  --adapter "sft=$sft" --adapter "grad1=$root/runs/$run1" --adapter "grad2=$out2" \
  --output-dir "$eval_dir/predictions" --max-model-len 12000 --max-tokens 512 \
  --max-num-batched-tokens 32768 --gpu-memory-utilization 0.88 --enforce-eager \
  >"$eval_dir/generate.log" 2>&1

for name in sft grad1 grad2; do
  python3 "$root/tools/evaluate_exx_outputs_claimcite_v2_20260829.py" \
    --predictions "$eval_dir/predictions/$name.predictions.json" \
    --gold "$gold_val" --output "$eval_dir/$name.metrics.json" \
    >"$eval_dir/$name.summary.txt"
done
date -Is >"$eval_dir/COMPLETE"
