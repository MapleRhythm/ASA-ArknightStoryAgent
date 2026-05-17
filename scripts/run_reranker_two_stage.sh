#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zhb/miniconda3/envs/train/bin/python}"

V1_FILE="${V1_FILE:-$ROOT_DIR/data/processed/evidence_chain_reranker/batch_v1_strict/reranker_pairwise.jsonl}"
V2_FILE="${V2_FILE:-$ROOT_DIR/data/processed/evidence_chain_reranker/batch_v2_dpo_filtered/reranker_pairwise.jsonl}"
MERGED_DIR="${MERGED_DIR:-$ROOT_DIR/data/processed/evidence_chain_reranker/reranker_sft_merged_v1}"
MERGED_FILE="${MERGED_FILE:-$MERGED_DIR/reranker_pairwise.jsonl}"

BASE_MODEL="${BASE_MODEL:-$ROOT_DIR/model/reranker/bge-reranker-v2-m3}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-$ROOT_DIR/model/reranker/bge-reranker-v2-m3-evidence-chain-sft-v1}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-$ROOT_DIR/model/reranker/bge-reranker-v2-m3-evidence-chain-dpo-v1}"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"

STAGE1_EPOCHS="${STAGE1_EPOCHS:-2}"
STAGE1_LR="${STAGE1_LR:-2e-5}"
STAGE1_MAX_STEPS="${STAGE1_MAX_STEPS:--1}"
STAGE1_OVERWRITE_OUTPUT_DIR="${STAGE1_OVERWRITE_OUTPUT_DIR:-false}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-1}"
STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:--1}"
STAGE2_DPO_BETA="${STAGE2_DPO_BETA:-0.1}"
STAGE2_OVERWRITE_OUTPUT_DIR="${STAGE2_OVERWRITE_OUTPUT_DIR:-false}"

COMMON_MAX_LENGTH="${MAX_LENGTH:-1024}"
COMMON_EVAL_RATIO="${EVAL_RATIO:-0.05}"
COMMON_TRAIN_BSZ="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
COMMON_EVAL_BSZ="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
COMMON_GRAD_ACCUM="${GRADIENT_ACCUMULATION_STEPS:-8}"
COMMON_SAVE_STEPS="${SAVE_STEPS:-50}"
COMMON_EVAL_STEPS="${EVAL_STEPS:-50}"
COMMON_LOGGING_STEPS="${LOGGING_STEPS:-5}"
COMMON_REPORT_TO="${REPORT_TO:-none}"
COMMON_SEED="${SEED:-20260514}"
COMMON_BF16="${BF16:-auto}"
COMMON_FP16="${FP16:-false}"
COMMON_GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

cd "$ROOT_DIR"

if [[ ! -f "$V1_FILE" ]]; then
  echo "Missing V1_FILE: $V1_FILE" >&2
  exit 1
fi
if [[ ! -f "$V2_FILE" ]]; then
  echo "Missing V2_FILE: $V2_FILE" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Missing BASE_MODEL: $BASE_MODEL" >&2
  exit 1
fi

mkdir -p "$MERGED_DIR"

"$PYTHON_BIN" - "$V1_FILE" "$V2_FILE" "$MERGED_FILE" "$MERGED_DIR/manifest.json" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def issue_flags(record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    positive = str(record.get("positive") or "").strip()
    negative = str(record.get("negative") or "").strip()
    if not record.get("query") or not positive or not negative:
        flags.append("missing_field")
    if positive == negative:
        flags.append("same_text")
    try:
        positive_score = float(record.get("positive_score", 1.0))
        negative_score = float(record.get("negative_score", 0.0))
    except (TypeError, ValueError):
        flags.append("bad_score")
        positive_score = 1.0
        negative_score = 0.0
    if positive_score <= negative_score:
        flags.append("non_positive_margin")
    if negative_score >= 0.9:
        flags.append("negative_score_gte_0.9")
    positive_chain = {str(item) for item in record.get("positive_chain") or [] if str(item)}
    negative_chain = {str(item) for item in record.get("negative_chain") or [] if str(item)}
    if positive_chain and negative_chain and positive_chain == negative_chain:
        flags.append("same_evidence_set")
    answer_evidence = {str(item) for item in record.get("answer_evidence") or [] if str(item)}
    if answer_evidence and answer_evidence.issubset(negative_chain) and negative_score > 0.6:
        flags.append("negative_contains_all_answer_evidence_high_score")
    return flags


v1_path = Path(sys.argv[1])
v2_path = Path(sys.argv[2])
merged_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])

sources = [("batch_v1_strict", v1_path), ("batch_v2_dpo_filtered", v2_path)]
merged: list[dict[str, Any]] = []
seen: set[tuple[str, str, str]] = set()
source_counts: Counter[str] = Counter()
duplicate_counts: Counter[str] = Counter()
flag_counts: Counter[str] = Counter()
negative_types: Counter[str] = Counter()
query_types: Counter[str] = Counter()

for source_name, path in sources:
    for row in read_jsonl(path):
        key = (
            str(row.get("query") or ""),
            str(row.get("positive") or ""),
            str(row.get("negative") or ""),
        )
        if key in seen:
            duplicate_counts[source_name] += 1
            continue
        seen.add(key)
        next_row = dict(row)
        next_row.setdefault("dataset_source", source_name)
        merged.append(next_row)
        source_counts[source_name] += 1
        negative_types[str(row.get("negative_type") or "")] += 1
        query_types[str(row.get("query_type") or "")] += 1
        for flag in issue_flags(row):
            flag_counts[flag] += 1

write_jsonl(merged_path, merged)
manifest = {
    "purpose": "Stage-1 reranker softplus/pairwise SFT dataset. Use batch_v2_dpo_filtered alone for stage-2 DPO.",
    "sources": [{"name": name, "path": str(path)} for name, path in sources],
    "output": str(merged_path),
    "records": len(merged),
    "source_counts": dict(source_counts),
    "duplicates_dropped": dict(duplicate_counts),
    "known_quality_notes": {
        "batch_v1_strict_same_evidence_set": "kept for softplus/pairwise SFT only; do not use this merged file for DPO",
        "batch_v2_dpo_filtered": "filtered to remove high-risk same-evidence-set and high-score negatives",
    },
    "issue_flags": dict(flag_counts),
    "negative_types": dict(negative_types),
    "query_types": dict(query_types),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo
echo "[stage-1] softplus/pairwise SFT"
if [[ "$SKIP_STAGE1" != "true" ]]; then
  TRAIN_FILE="$MERGED_FILE" \
  BASE_MODEL="$BASE_MODEL" \
  OUTPUT_DIR="$STAGE1_OUTPUT_DIR" \
  TRAIN_GPUS="$TRAIN_GPUS" \
  LOSS_TYPE=softplus \
  EPOCHS="$STAGE1_EPOCHS" \
  LEARNING_RATE="$STAGE1_LR" \
  MAX_STEPS="$STAGE1_MAX_STEPS" \
  MAX_LENGTH="$COMMON_MAX_LENGTH" \
  EVAL_RATIO="$COMMON_EVAL_RATIO" \
  PER_DEVICE_TRAIN_BATCH_SIZE="$COMMON_TRAIN_BSZ" \
  PER_DEVICE_EVAL_BATCH_SIZE="$COMMON_EVAL_BSZ" \
  GRADIENT_ACCUMULATION_STEPS="$COMMON_GRAD_ACCUM" \
  SAVE_STEPS="$COMMON_SAVE_STEPS" \
  EVAL_STEPS="$COMMON_EVAL_STEPS" \
  LOGGING_STEPS="$COMMON_LOGGING_STEPS" \
  REPORT_TO="$COMMON_REPORT_TO" \
  SEED="$COMMON_SEED" \
  BF16="$COMMON_BF16" \
  FP16="$COMMON_FP16" \
  GRADIENT_CHECKPOINTING="$COMMON_GRADIENT_CHECKPOINTING" \
  OVERWRITE_OUTPUT_DIR="$STAGE1_OVERWRITE_OUTPUT_DIR" \
  DRY_RUN="$DRY_RUN" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_reranker_sft.sh
else
  echo "Skipped stage-1 because SKIP_STAGE1=true"
fi

echo
echo "[stage-2] DPO preference tuning"
if [[ "$SKIP_STAGE2" != "true" ]]; then
  if [[ "$DRY_RUN" != "true" && ! -d "$STAGE1_OUTPUT_DIR" ]]; then
    echo "Stage-1 output does not exist: $STAGE1_OUTPUT_DIR" >&2
    echo "Run stage-1 first, or set BASE_MODEL/STAGE1_OUTPUT_DIR accordingly." >&2
    exit 1
  fi
  STAGE2_BASE_MODEL="$STAGE1_OUTPUT_DIR"
  if [[ "$DRY_RUN" == "true" ]]; then
    STAGE2_BASE_MODEL="$BASE_MODEL"
  fi
  TRAIN_FILE="$V2_FILE" \
  BASE_MODEL="$STAGE2_BASE_MODEL" \
  OUTPUT_DIR="$STAGE2_OUTPUT_DIR" \
  TRAIN_GPUS="$TRAIN_GPUS" \
  LOSS_TYPE=dpo \
  DPO_BETA="$STAGE2_DPO_BETA" \
  EPOCHS="$STAGE2_EPOCHS" \
  LEARNING_RATE="$STAGE2_LR" \
  MAX_STEPS="$STAGE2_MAX_STEPS" \
  MAX_LENGTH="$COMMON_MAX_LENGTH" \
  EVAL_RATIO="$COMMON_EVAL_RATIO" \
  PER_DEVICE_TRAIN_BATCH_SIZE="$COMMON_TRAIN_BSZ" \
  PER_DEVICE_EVAL_BATCH_SIZE="$COMMON_EVAL_BSZ" \
  GRADIENT_ACCUMULATION_STEPS="$COMMON_GRAD_ACCUM" \
  SAVE_STEPS="$COMMON_SAVE_STEPS" \
  EVAL_STEPS="$COMMON_EVAL_STEPS" \
  LOGGING_STEPS="$COMMON_LOGGING_STEPS" \
  REPORT_TO="$COMMON_REPORT_TO" \
  SEED="$COMMON_SEED" \
  BF16="$COMMON_BF16" \
  FP16="$COMMON_FP16" \
  GRADIENT_CHECKPOINTING="$COMMON_GRADIENT_CHECKPOINTING" \
  OVERWRITE_OUTPUT_DIR="$STAGE2_OVERWRITE_OUTPUT_DIR" \
  DRY_RUN="$DRY_RUN" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash scripts/run_reranker_sft.sh
else
  echo "Skipped stage-2 because SKIP_STAGE2=true"
fi
