#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_RECORDS = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel/audit_records.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/soda_gold_evidence_topk_report.json"
QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
EVIDENCE_RE = re.compile(r"(?s)^evidence_brief:\s*(.*?)(?:\nminirag_hints:|\noutput_schema:|\nfields:|\Z)", re.MULTILINE)
EVIDENCE_LINE_RE = re.compile(r"^(\d+)\.\s+([^:]+):\s*(.*)$")
GOLD_SPLIT_RE = re.compile(r"(?=\[E\d+\])")
KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def normalize(text: str) -> str:
    return "".join(KEEP_RE.findall(text or "")).lower()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) <= n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def overlap_score(gold: str, candidate: str) -> float:
    gold_norm = normalize(gold)
    candidate_norm = normalize(candidate)
    if not gold_norm or not candidate_norm:
        return 0.0
    if gold_norm in candidate_norm or candidate_norm in gold_norm:
        return 1.0
    gold_grams = char_ngrams(gold_norm)
    candidate_grams = char_ngrams(candidate_norm)
    if not gold_grams or not candidate_grams:
        return 0.0
    shared = len(gold_grams & candidate_grams)
    return shared / min(len(gold_grams), len(candidate_grams))


def extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_round(prompt: str) -> str:
    match = ROUND_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_evidence_lines(prompt: str) -> list[dict[str, Any]]:
    match = EVIDENCE_RE.search(prompt or "")
    if not match:
        return []
    lines: list[dict[str, Any]] = []
    for raw in match.group(1).strip().splitlines():
        item = EVIDENCE_LINE_RE.match(raw.strip())
        if not item:
            continue
        lines.append(
            {
                "rank": int(item.group(1)),
                "doc_id": item.group(2).strip(),
                "text": item.group(3).strip(),
            }
        )
    return lines


def extract_gold_units(gold: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for chunk in GOLD_SPLIT_RE.split(gold or ""):
        text = chunk.strip()
        if not text:
            continue
        label_match = re.match(r"\[(E\d+)\]\s*(.*)", text, flags=re.S)
        if label_match:
            label = label_match.group(1)
            body = label_match.group(2).strip()
        else:
            label = f"G{len(units) + 1}"
            body = text
        if body:
            units.append({"label": label, "text": body})
    return units


def first_hit_rank(gold_text: str, evidence_lines: list[dict[str, Any]], *, threshold: float) -> tuple[int | None, float, str]:
    best_rank: int | None = None
    best_score = 0.0
    best_doc_id = ""
    for line in evidence_lines:
        score = overlap_score(gold_text, str(line.get("text") or ""))
        if score > best_score:
            best_score = score
            best_rank = int(line["rank"])
            best_doc_id = str(line.get("doc_id") or "")
    if best_score >= threshold:
        return best_rank, best_score, best_doc_id
    return None, best_score, best_doc_id


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def cumulative_counts(ranks: list[int | None], top_ks: list[int]) -> dict[str, Any]:
    total = len(ranks)
    return {
        f"@{top_k}": {
            "count": sum(rank is not None and rank <= top_k for rank in ranks),
            "ratio": ratio(sum(rank is not None and rank <= top_k for rank in ranks), total),
        }
        for top_k in top_ks
    } | {"missed": {"count": sum(rank is None for rank in ranks), "ratio": ratio(sum(rank is None for rank in ranks), total)}}


def build_report(records: list[dict[str, Any]], *, top_ks: list[int], threshold: float) -> dict[str, Any]:
    prompt_rows: list[dict[str, Any]] = []
    gold_unit_ranks: list[int | None] = []
    rank_counter: Counter = Counter()
    question_best: dict[str, list[int | None]] = defaultdict(list)
    misses: list[dict[str, Any]] = []

    for record in records:
        if record.get("task_type") != "conclusion_generation" or not bool(record.get("kto_tag")):
            continue
        prompt = record.get("conversations", [{}])[0].get("value", "")
        question = extract_question(prompt)
        round_id = extract_round(prompt)
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        source = meta.get("source") if isinstance(meta.get("source"), dict) else {}
        gold_units = extract_gold_units(str(source.get("gold") or ""))
        if not gold_units:
            continue
        evidence_lines = extract_evidence_lines(prompt)
        ranks_for_prompt: list[int | None] = []
        scores_for_prompt: list[float] = []
        for unit in gold_units:
            rank, score, doc_id = first_hit_rank(unit["text"], evidence_lines, threshold=threshold)
            ranks_for_prompt.append(rank)
            scores_for_prompt.append(score)
            gold_unit_ranks.append(rank)
            if rank is not None:
                rank_counter[rank] += 1
            if rank is None and len(misses) < 50:
                misses.append(
                    {
                        "question": question,
                        "round": round_id,
                        "gold_label": unit["label"],
                        "best_score": round(score, 4),
                        "best_doc_id": doc_id,
                        "gold_excerpt": normalize(unit["text"])[:120],
                        "evidence_doc_ids": [line["doc_id"] for line in evidence_lines[:12]],
                    }
                )
        question_best[question].extend(ranks_for_prompt)
        prompt_rows.append(
            {
                "question": question,
                "round": round_id,
                "prompt_key": meta.get("prompt_key"),
                "gold_units": len(gold_units),
                "hit_units": sum(rank is not None for rank in ranks_for_prompt),
                "coverage": ratio(sum(rank is not None for rank in ranks_for_prompt), len(ranks_for_prompt)),
                "min_rank": min((rank for rank in ranks_for_prompt if rank is not None), default=None),
                "max_hit_rank": max((rank for rank in ranks_for_prompt if rank is not None), default=None),
                "best_scores": [round(score, 4) for score in scores_for_prompt],
            }
        )

    prompt_any_ranks = [row["min_rank"] for row in prompt_rows]
    prompt_all_ranks = [
        row["max_hit_rank"] if row["hit_units"] == row["gold_units"] else None
        for row in prompt_rows
    ]
    question_all_ranks: list[int | None] = []
    question_any_ranks: list[int | None] = []
    for ranks in question_best.values():
        hits = [rank for rank in ranks if rank is not None]
        question_any_ranks.append(min(hits) if hits else None)
        question_all_ranks.append(max(hits) if len(hits) == len(ranks) else None)

    coverages = [row["coverage"] for row in prompt_rows]
    return {
        "settings": {
            "match_threshold": threshold,
            "top_ks": top_ks,
            "note": "Ranks are evidence_brief line ranks in existing SODA conclusion prompts; matching uses character trigram overlap.",
        },
        "counts": {
            "conclusion_prompts": len(prompt_rows),
            "questions": len(question_best),
            "gold_units_prompt_level": len(gold_unit_ranks),
        },
        "gold_unit_rank_distribution": dict(sorted(rank_counter.items())),
        "gold_unit_cumulative": cumulative_counts(gold_unit_ranks, top_ks),
        "prompt_any_gold_cumulative": cumulative_counts(prompt_any_ranks, top_ks),
        "prompt_all_gold_cumulative": cumulative_counts(prompt_all_ranks, top_ks),
        "question_any_gold_cumulative": cumulative_counts(question_any_ranks, top_ks),
        "question_all_gold_cumulative": cumulative_counts(question_all_ranks, top_ks),
        "prompt_gold_coverage": {
            "mean": round(statistics.mean(coverages), 4) if coverages else 0.0,
            "p50": round(statistics.median(coverages), 4) if coverages else 0.0,
            "full_coverage_count": sum(value >= 1.0 for value in coverages),
            "full_coverage_ratio": ratio(sum(value >= 1.0 for value in coverages), len(coverages)),
            "zero_coverage_count": sum(value <= 0.0 for value in coverages),
            "zero_coverage_ratio": ratio(sum(value <= 0.0 for value in coverages), len(coverages)),
        },
        "miss_examples": misses,
    }


def parse_top_ks(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gold evidence rank distribution in SODA evidence_brief prompts.")
    parser.add_argument("--audit-records", type=Path, default=DEFAULT_AUDIT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-ks", type=parse_top_ks, default="1,3,5,8,10,12")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_path = args.audit_records if args.audit_records.is_absolute() else PROJECT_ROOT / args.audit_records
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    records = read_records(audit_path)
    report = build_report(records, top_ks=args.top_ks, threshold=args.threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:5000])
    print(f"\n[written] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
