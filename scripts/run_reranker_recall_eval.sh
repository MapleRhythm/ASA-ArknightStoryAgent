#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_SH="${CONDA_SH:-/home/zhb/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-train}"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "[eval] missing conda profile: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="${ROOT_DIR}/.python_packages/train:${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DEVICE="${DEVICE:-cuda}"
SAMPLE="${SAMPLE:-}"
TOP_KS="${TOP_KS:-1,5,10,20}"
LISTWISE="${LISTWISE:-data/processed/evidence_chain_reranker/batch_v2_dpo_filtered/reranker_listwise.jsonl}"
INDEX_DIR="${INDEX_DIR:-indexes/arknights_story}"
OLD_RERANKER="${OLD_RERANKER:-model/reranker/bge-reranker-v2-m3-evidence-chain-answerability}"
NEW_RERANKER="${NEW_RERANKER:-}"
OUT_DIR="${OUT_DIR:-outputs/retrieval_eval}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_DIR"

if [[ -z "$NEW_RERANKER" ]]; then
  echo "[eval] NEW_RERANKER is required; the old DPO artifact is no longer kept locally." >&2
  echo "[eval] Example: NEW_RERANKER=model/reranker/<candidate> bash scripts/run_reranker_recall_eval.sh" >&2
  exit 2
fi

if [[ "$DEVICE" == cuda* ]]; then
  python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.stderr.write("[eval] CUDA is not available in this environment. Set DEVICE=cpu to force CPU.\n")
    raise SystemExit(2)
print(f"[eval] cuda devices={torch.cuda.device_count()} current={torch.cuda.current_device()} name={torch.cuda.get_device_name(0)}")
PY
fi

SAMPLE_ARGS=()
if [[ -n "$SAMPLE" ]]; then
  SAMPLE_ARGS=(--sample "$SAMPLE")
fi

COMMON_ARGS=(
  --listwise "$LISTWISE"
  --device "$DEVICE"
  --top-ks "$TOP_KS"
  --index-dir "$INDEX_DIR"
  "${SAMPLE_ARGS[@]}"
)

OLD_OUT="$OUT_DIR/old_answerability_${DEVICE}_${STAMP}.json"
NEW_OUT="$OUT_DIR/new_dpo_${DEVICE}_${STAMP}.json"
REPORT_OUT="$OUT_DIR/comparison_report_${DEVICE}_${STAMP}.md"

echo "[eval] old reranker -> $OLD_OUT"
python scripts/evaluate_retrieval_recall.py \
  "${COMMON_ARGS[@]}" \
  --reranker-model "$OLD_RERANKER" \
  --output "$OLD_OUT" \
  --tag "old_answerability_${DEVICE}_${STAMP}"

echo "[eval] new reranker -> $NEW_OUT"
python scripts/evaluate_retrieval_recall.py \
  "${COMMON_ARGS[@]}" \
  --reranker-model "$NEW_RERANKER" \
  --output "$NEW_OUT" \
  --tag "new_dpo_${DEVICE}_${STAMP}"

python - "$OLD_OUT" "$NEW_OUT" "$REPORT_OUT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

old_path = Path(sys.argv[1])
new_path = Path(sys.argv[2])
report_path = Path(sys.argv[3])
old = json.loads(old_path.read_text(encoding="utf-8"))
new = json.loads(new_path.read_text(encoding="utf-8"))

def recall_line(name: str, payload: dict) -> str:
    overall = payload["overall"]
    recall = overall["recall"]
    return (
        f"| {name} | {overall['count']} | {overall['mrr']:.4f} | "
        f"{recall.get('@1', 0):.4f} | {recall.get('@5', 0):.4f} | "
        f"{recall.get('@10', 0):.4f} | {recall.get('@20', 0):.4f} |"
    )

def delta(key: str) -> float:
    return float(new["overall"]["recall"].get(key, 0)) - float(old["overall"]["recall"].get(key, 0))

text = "\n".join(
    [
        "# Reranker Recall Comparison",
        "",
        f"- old: `{old_path}`",
        f"- new: `{new_path}`",
        f"- listwise: `{new['listwise_path']}`",
        f"- device: `{new['device']}`",
        f"- index: `{new['index_dir']}`",
        "",
        "| model | count | MRR | R@1 | R@5 | R@10 | R@20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        recall_line("old_answerability", old),
        recall_line("new_dpo", new),
        "",
        "## Delta",
        "",
        f"- MRR: {float(new['overall']['mrr']) - float(old['overall']['mrr']):+.4f}",
        f"- R@1: {delta('@1'):+.4f}",
        f"- R@5: {delta('@5'):+.4f}",
        f"- R@10: {delta('@10'):+.4f}",
        f"- R@20: {delta('@20'):+.4f}",
        "",
    ]
)
report_path.write_text(text, encoding="utf-8")
print(f"[eval] report -> {report_path}")
PY

echo "[eval] done"
