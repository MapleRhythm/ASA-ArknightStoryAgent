#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY before running this script.}"

OUT_CANDIDATES="data/processed/opd_candidates/qwen35_4b_full_chain_smoke10"
OUT_SCORES="data/processed/opd_teacher_scores/qwen35_4b_full_chain_smoke10_deepseek"

rm -rf "$OUT_CANDIDATES" "$OUT_SCORES"

run_train_python() {
  local python_exe
  python_exe="$(command -v python || true)"

  if [[ "${CONDA_DEFAULT_ENV:-}" == "train" && -n "$python_exe" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CONDA_NO_PLUGINS=true \
    PYTHONPATH=.python_packages/train:src \
    "$python_exe" "$@"
    return
  fi

  if [[ -x "$HOME/miniconda3/envs/train/bin/python" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CONDA_NO_PLUGINS=true \
    PYTHONPATH=.python_packages/train:src \
    "$HOME/miniconda3/envs/train/bin/python" "$@"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CONDA_NO_PLUGINS=true \
    PYTHONPATH=.python_packages/train:src \
    conda run --no-capture-output -n train python "$@"
    return
  fi

  echo "Cannot find train Python. Activate env first or install conda in PATH." >&2
  exit 127
}

run_train_python scripts/generate_opd_full_chain_candidates_from_4b.py \
  --output-dir "$OUT_CANDIDATES" \
  --runtime-config configs/runtime_inference_gpu.json \
  --sample 10 \
  --runs-per-question 1 \
  --max-rounds 3 \
  --device cuda:0 \
  --reranker-model model/reranker/bge-reranker-v2-m3-evidence-chain-answerability \
  --minirag-index indexes/arknights_story_minirag_v3/graph.json \
  --dense-top-k 120 \
  --sparse-top-k 120 \
  --fusion-top-k 80 \
  --reranker-candidate-top-k 120 \
  --rerank-top-k 50 \
  --minirag-top-k 120 \
  --minirag-weight 0.35 \
  --minirag-fusion-mode score

run_train_python scripts/score_opd_candidates_with_teacher.py \
  --input "$OUT_CANDIDATES/candidates.jsonl" \
  --output-dir "$OUT_SCORES" \
  --api-type chat_completions \
  --api-base https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --model deepseek-v4-flash \
  --parallel 4 \
  --api-retries 3 \
  --retry-sleep 20 \
  --max-output-tokens 2048

echo
echo "[candidate summary]"
cat "$OUT_CANDIDATES/build_summary.json"
echo
echo "[score summary]"
cat "$OUT_SCORES/score_summary.json"
