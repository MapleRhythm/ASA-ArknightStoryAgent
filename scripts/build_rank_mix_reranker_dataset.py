#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_rank_clean_reranker_dataset import (  # noqa: E402
    PROJECT_ROOT,
    answer_overlap,
    build_listwise_from_pairs,
    doc_training_text,
    gold_candidate,
    load_documents,
    lexical_grams,
    normalize_text,
    read_jsonl,
    resolve_path,
    should_keep_original_pair,
    text_key,
    write_jsonl,
)

EVIDENCE_SEGMENT_RE = re.compile(r"\[E(\d+)\][\s\S]*?(?=\s*\[E\d+\]|\Z)")


def pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("query") or ""),
        text_key(str(record.get("positive") or "")),
        text_key(str(record.get("negative") or "")),
    )


def chain_set(record: dict[str, Any], field: str) -> set[str]:
    return {str(item) for item in (record.get(field) or []) if str(item)}


def load_eval_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record.get("query") or ""): record
        for record in payload.get("records") or []
        if str(record.get("query") or "")
    }


def numeric_rank(record: dict[str, Any] | None, *, miss_rank: int) -> int:
    if not record:
        return miss_rank
    rank = record.get("first_hit_rank")
    if isinstance(rank, int) and rank > 0:
        return rank
    return miss_rank


def is_regression_case(
    old_record: dict[str, Any],
    new_record: dict[str, Any],
    *,
    min_delta: int,
    miss_rank: int,
) -> bool:
    old_rank = numeric_rank(old_record, miss_rank=miss_rank)
    new_rank = numeric_rank(new_record, miss_rank=miss_rank)
    if old_rank >= miss_rank:
        return False
    if new_rank >= miss_rank and old_rank < miss_rank:
        return True
    if old_rank <= 5 and new_rank > 5:
        return True
    return new_rank - old_rank >= min_delta


def top_doc_ids(eval_record: dict[str, Any], *, max_per_round: int, max_total: int) -> list[str]:
    output: list[str] = []
    for round_info in eval_record.get("rounds") or []:
        for doc_id in (round_info.get("top_doc_ids") or [])[:max_per_round]:
            value = str(doc_id or "").strip()
            if value and value not in output:
                output.append(value)
            if len(output) >= max_total:
                return output
    return output


def add_records(
    target: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    records: list[dict[str, Any]],
    stats: Counter[str],
    *,
    stat_prefix: str,
) -> None:
    for record in records:
        key = pair_key(record)
        if key in seen:
            stats[f"{stat_prefix}_duplicate"] += 1
            continue
        seen.add(key)
        target.append(record)
        stats[f"{stat_prefix}_added"] += 1


def _append_quality_note(record: dict[str, Any], note: str) -> None:
    existing_note = str(record.get("mix_quality_note") or "")
    if note in existing_note.split(";"):
        return
    record["mix_quality_note"] = f"{existing_note};{note}" if existing_note else note


def split_evidence_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for match in EVIDENCE_SEGMENT_RE.finditer(text or ""):
        evidence_id = f"E{match.group(1)}"
        segment = match.group(0).strip()
        if segment:
            segments.append((evidence_id, segment))
    return segments


def compact_text_for_training(record: dict[str, Any], text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    segments = split_evidence_segments(text)
    if len(segments) < 2:
        head_chars = max_chars // 2
        tail_chars = max_chars - head_chars
        return f"{text[:head_chars].rstrip()}\n...\n{text[-tail_chars:].lstrip()}", True

    answer_evidence = {str(item) for item in (record.get("answer_evidence") or [])}
    query_terms = lexical_grams(str(record.get("query") or ""))
    answer_terms = lexical_grams(f"{record.get('answer') or ''} {record.get('answer_focus') or ''}")

    def query_overlap(segment: str) -> float:
        if not query_terms:
            return 0.0
        segment_terms = lexical_grams(segment)
        return len(query_terms & segment_terms) / len(query_terms)

    def answer_term_overlap(segment: str) -> float:
        if not answer_terms:
            return 0.0
        segment_terms = lexical_grams(segment)
        return len(answer_terms & segment_terms) / len(answer_terms)

    scored: list[tuple[float, int, str]] = []
    for index, (evidence_id, segment) in enumerate(segments):
        score = answer_overlap(record, segment)
        score += query_overlap(segment) * 0.6
        if evidence_id in answer_evidence:
            score += 1.0
        scored.append((score, index, segment))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    def clip_segment(segment: str, limit: int) -> str:
        if len(segment) <= limit:
            return segment
        head_chars = limit // 2
        tail_chars = limit - head_chars
        return f"{segment[:head_chars].rstrip()}\n...\n{segment[-tail_chars:].lstrip()}"

    def clip_segment_around_relevant_terms(segment: str, limit: int) -> str:
        if len(segment) <= limit:
            return segment
        if limit <= 40:
            return segment[:limit]
        evidence_prefix_match = re.match(r"^\[E\d+\]\s*", segment)
        prefix = evidence_prefix_match.group(0) if evidence_prefix_match else ""
        body_start = len(prefix)
        body_limit = max(20, limit - len(prefix) - 6)
        relevant_terms = sorted(
            lexical_grams(
                f"{record.get('query') or ''} {record.get('answer') or ''} {record.get('answer_focus') or ''}"
            ),
            key=len,
            reverse=True,
        )[:240]
        positions: list[int] = []
        for term in relevant_terms:
            position = segment.find(term, body_start)
            if position >= 0:
                positions.append(position)
        if not positions:
            return clip_segment(segment, limit)
        best_start = body_start
        best_score = -1
        max_start = max(body_start, len(segment) - body_limit)
        for position in positions:
            start = max(body_start, min(position - body_limit // 2, max_start))
            window = segment[start : start + body_limit]
            score = sum(1 for term in relevant_terms if term in window)
            if score > best_score:
                best_score = score
                best_start = start
        body = segment[best_start : best_start + body_limit].strip()
        head_marker = "..." if best_start > body_start else ""
        tail_marker = "..." if best_start + body_limit < len(segment) else ""
        return f"{prefix}{head_marker}{body}{tail_marker}".strip()

    selected: list[tuple[int, str]] = []
    selected_chars = 0
    selected_indices: set[int] = set()

    def add_selected(index: int, segment: str, *, allow_clip_if_empty: bool = True) -> bool:
        nonlocal selected_chars
        if index in selected_indices:
            return True
        remaining = max_chars - selected_chars - (1 if selected else 0)
        if remaining <= 0:
            return False
        if len(segment) > remaining and not selected and allow_clip_if_empty:
            segment = clip_segment_around_relevant_terms(segment, remaining)
        if not selected:
            selected.append((index, segment))
            selected_chars += len(segment)
            selected_indices.add(index)
            return True
        separator_chars = 1
        if selected_chars + separator_chars + len(segment) > max_chars:
            return False
        selected.append((index, segment))
        selected_chars += separator_chars + len(segment)
        selected_indices.add(index)
        return True

    # Preserve a short query/answer anchor in positives. Otherwise long evidence
    # chains can compact to a later explanatory segment and lose the trigger fact.
    is_positive_text = normalize_text(text) == normalize_text(str(record.get("positive") or ""))
    if is_positive_text:
        required_segments = [
            (index, segment)
            for index, (evidence_id, segment) in enumerate(segments)
            if evidence_id in answer_evidence
        ]
        if len(required_segments) >= 2:
            separator_budget = max(0, len(required_segments) - 1)
            per_segment_limit = max(120, (max_chars - separator_budget) // len(required_segments))
            for index, segment in required_segments:
                add_selected(
                    index,
                    clip_segment_around_relevant_terms(segment, per_segment_limit),
                    allow_clip_if_empty=False,
                )
        anchor_candidates: list[tuple[float, int, str]] = []
        for index, (evidence_id, segment) in enumerate(segments):
            anchor_score = query_overlap(segment) + answer_term_overlap(segment)
            if evidence_id in answer_evidence:
                anchor_score += 0.5
            anchor_candidates.append((anchor_score, index, segment))
        anchor_candidates.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
        for anchor_score, index, segment in anchor_candidates:
            if anchor_score <= 0:
                break
            if len(segment) > max_chars:
                segment = clip_segment_around_relevant_terms(segment, min(max_chars, max(240, max_chars // 2)))
            if add_selected(index, segment):
                break

    for score, index, segment in scored:
        add_selected(index, segment)
    selected.sort(key=lambda item: item[0])
    compacted = "\n".join(segment for _, segment in selected).strip()
    if not compacted:
        return text[:max_chars], True
    return compacted, True


def compact_pairwise_texts(records: list[dict[str, Any]], *, max_chars: int) -> Counter[str]:
    stats: Counter[str] = Counter()
    if max_chars <= 0:
        return stats
    for record in records:
        for field in ("positive", "negative"):
            text = str(record.get(field) or "")
            compacted, changed = compact_text_for_training(record, text, max_chars=max_chars)
            if not changed:
                continue
            record[field] = compacted
            _append_quality_note(record, f"compacted_{field}_text")
            stats[f"compacted_{field}"] += 1
            stats[f"compacted_{field}_{str(record.get('negative_type') or 'unknown')}"] += 1
    return stats


def dedupe_pairwise_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    stats: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        positive = str(record.get("positive") or "")
        negative = str(record.get("negative") or "")
        if not positive.strip() or not negative.strip() or normalize_text(positive) == normalize_text(negative):
            stats[f"drop_equal_or_empty_{str(record.get('negative_type') or 'unknown')}"] += 1
            continue
        key = pair_key(record)
        if key in seen:
            stats[f"drop_duplicate_{str(record.get('negative_type') or 'unknown')}"] += 1
            continue
        seen.add(key)
        output.append(record)
    return output, stats


def downweight_answer_evidence_negatives(
    records: list[dict[str, Any]],
    *,
    negative_score: float,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for record in records:
        negative_type = str(record.get("negative_type") or "")
        if negative_type == "shuffled_order":
            continue
        answer_evidence = chain_set(record, "answer_evidence")
        negative_chain = chain_set(record, "negative_chain")
        if answer_evidence and answer_evidence.issubset(negative_chain):
            old_score = float(record.get("negative_score") or 0.0)
            if old_score < negative_score:
                record["negative_score"] = negative_score
                _append_quality_note(record, "downweighted_answer_evidence_negative")
                stats[f"downweighted_{negative_type}"] += 1
    return stats


def downweight_partial_answer_evidence_negatives(
    records: list[dict[str, Any]],
    *,
    negative_score: float,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for record in records:
        negative_type = str(record.get("negative_type") or "")
        if negative_type == "shuffled_order":
            continue
        answer_evidence = chain_set(record, "answer_evidence")
        negative_chain = chain_set(record, "negative_chain")
        if not answer_evidence:
            continue
        if not (answer_evidence & negative_chain):
            continue
        old_score = float(record.get("negative_score") or 0.0)
        if old_score < negative_score:
            record["negative_score"] = negative_score
            _append_quality_note(record, "downweighted_partial_answer_evidence_negative")
            stats[f"downweighted_{negative_type}"] += 1
    return stats


def downweight_partial_answer_negatives(
    records: list[dict[str, Any]],
    *,
    negative_score: float,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for record in records:
        if str(record.get("negative_type") or "") != "partial_answer":
            continue
        old_score = float(record.get("negative_score") or 0.0)
        if old_score < negative_score:
            record["negative_score"] = negative_score
            _append_quality_note(record, "downweighted_partial_answer_negative")
            stats["downweighted_partial_answer"] += 1
    return stats


def text_similarity(record: dict[str, Any]) -> float:
    positive_terms = lexical_grams(str(record.get("positive") or ""))
    negative_terms = lexical_grams(str(record.get("negative") or ""))
    if not positive_terms and not negative_terms:
        return 0.0
    return len(positive_terms & negative_terms) / max(1, len(positive_terms | negative_terms))


def downweight_text_similar_negatives(
    records: list[dict[str, Any]],
    *,
    negative_score: float,
    min_similarity: float,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for record in records:
        negative_type = str(record.get("negative_type") or "")
        if negative_type == "shuffled_order":
            continue
        old_score = float(record.get("negative_score") or 0.0)
        if old_score >= negative_score:
            continue
        if text_similarity(record) < min_similarity:
            continue
        record["negative_score"] = negative_score
        _append_quality_note(record, "downweighted_text_similar_negative")
        stats[f"downweighted_{negative_type}"] += 1
    return stats


def downweight_answer_like_negatives(
    records: list[dict[str, Any]],
    *,
    negative_score: float,
    min_score_diff: float,
    min_negative_overlap: float,
    overlap_ratio: float,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for record in records:
        negative_type = str(record.get("negative_type") or "")
        if negative_type == "shuffled_order":
            continue
        current_negative_score = float(record.get("negative_score") or 0.0)
        score_diff = float(record.get("positive_score") or 0.0) - current_negative_score
        if score_diff <= min_score_diff:
            continue
        positive = str(record.get("positive") or "")
        negative = str(record.get("negative") or "")
        positive_overlap = answer_overlap(record, positive)
        negative_overlap = answer_overlap(record, negative)
        if negative_overlap < min_negative_overlap:
            continue
        threshold = positive_overlap * overlap_ratio
        if negative_overlap >= threshold:
            record["negative_score"] = max(current_negative_score, negative_score)
            _append_quality_note(record, "downweighted_answer_like_negative")
            stats[f"downweighted_{negative_type}"] += 1
    return stats


def build_shuffled_order_pairs(
    old_pairwise: list[dict[str, Any]],
    *,
    min_positive_answer_overlap: float,
    max_per_query: int,
    negative_score: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_query: Counter[str] = Counter()
    for source in old_pairwise:
        if str(source.get("negative_type") or "") != "shuffled_order":
            continue
        query = str(source.get("query") or "")
        if max_per_query > 0 and per_query[query] >= max_per_query:
            stats["shuffled_cap_per_query"] += 1
            continue
        positive = str(source.get("positive") or "").strip()
        negative = str(source.get("negative") or "").strip()
        if not query or not positive or not negative:
            stats["shuffled_empty"] += 1
            continue
        if normalize_text(positive) == normalize_text(negative):
            stats["shuffled_equal_text"] += 1
            continue
        overlap = answer_overlap(source, positive)
        if overlap < min_positive_answer_overlap:
            stats["shuffled_weak_positive"] += 1
            continue
        record = dict(source)
        record["negative_score"] = negative_score
        record["mix_source"] = "old_shuffled_order_low_weight"
        output.append(record)
        per_query[query] += 1
        stats["shuffled_selected"] += 1
    return output, stats


def build_old_clean_dropped_pairs(
    old_pairwise: list[dict[str, Any]],
    v4_seen: set[tuple[str, str, str]],
    *,
    min_positive_answer_overlap: float,
    max_per_query: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_query: Counter[str] = Counter()
    for source in old_pairwise:
        negative_type = str(source.get("negative_type") or "")
        if negative_type == "shuffled_order":
            continue
        if pair_key(source) in v4_seen:
            continue
        query = str(source.get("query") or "")
        if max_per_query > 0 and per_query[query] >= max_per_query:
            stats["old_clean_cap_per_query"] += 1
            continue
        keep, reason = should_keep_original_pair(source, include_shuffled_order=False)
        if not keep:
            stats[f"old_clean_drop_{reason}"] += 1
            continue
        overlap = answer_overlap(source, str(source.get("positive") or ""))
        if overlap < min_positive_answer_overlap:
            stats["old_clean_weak_positive"] += 1
            continue
        record = dict(source)
        record["mix_source"] = "old_clean_dropped"
        output.append(record)
        per_query[query] += 1
        stats["old_clean_selected"] += 1
    return output, stats


def build_regression_pairs(
    *,
    old_eval: dict[str, dict[str, Any]],
    new_eval: dict[str, dict[str, Any]],
    listwise_by_query: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    min_rank_delta: int,
    min_positive_answer_overlap: float,
    negative_answer_overlap_margin: float,
    max_docs_per_round: int,
    max_docs_per_query: int,
    regression_negative_score: float,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    regression_summary: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for query, new_record in new_eval.items():
        old_record = old_eval.get(query)
        if old_record is None:
            stats["regression_missing_old_eval"] += 1
            continue
        if not is_regression_case(old_record, new_record, min_delta=min_rank_delta, miss_rank=999):
            continue
        source_record = listwise_by_query.get(query)
        if source_record is None:
            stats["regression_missing_listwise"] += 1
            continue
        gold = gold_candidate(source_record)
        if gold is None:
            stats["regression_missing_gold"] += 1
            continue
        positive = str(gold.get("text") or "").strip()
        if not positive:
            stats["regression_empty_gold"] += 1
            continue
        positive_overlap = answer_overlap(source_record, positive)
        if positive_overlap < min_positive_answer_overlap:
            stats["regression_weak_positive"] += 1
            continue
        doc_ids = top_doc_ids(
            new_record,
            max_per_round=max_docs_per_round,
            max_total=max_docs_per_query,
        )
        if not doc_ids:
            stats["regression_no_top_docs"] += 1
            continue
        added_for_query = 0
        skipped_answer_like = 0
        for doc_id in doc_ids:
            doc = documents.get(doc_id)
            if doc is None:
                stats["regression_missing_doc"] += 1
                continue
            negative = doc_training_text(doc)
            if not negative:
                stats["regression_empty_doc"] += 1
                continue
            if text_key(negative) == text_key(positive):
                stats["regression_same_text"] += 1
                continue
            negative_overlap = answer_overlap(source_record, negative)
            if negative_overlap + negative_answer_overlap_margin >= positive_overlap:
                skipped_answer_like += 1
                stats["regression_answer_like_negative"] += 1
                continue
            key = (query, text_key(positive), text_key(negative))
            if key in seen:
                stats["regression_duplicate_local"] += 1
                continue
            seen.add(key)
            output.append(
                {
                    "query": query,
                    "query_type": str(source_record.get("query_type") or "unknown"),
                    "source_name": str(source_record.get("source_name") or "regression_eval"),
                    "positive": positive,
                    "negative": negative,
                    "positive_score": 1.0,
                    "negative_score": regression_negative_score,
                    "negative_type": "online_regression_top_distractor",
                    "answer": str(source_record.get("answer") or ""),
                    "answer_evidence": list(source_record.get("answer_evidence") or []),
                    "answer_focus": str(source_record.get("answer_focus") or ""),
                    "positive_chain": list(gold.get("chain") or []),
                    "negative_chain": [doc_id],
                    "mix_source": "rank_clean_v4_regression_top_doc",
                    "old_first_hit_rank": old_record.get("first_hit_rank"),
                    "new_first_hit_rank": new_record.get("first_hit_rank"),
                }
            )
            added_for_query += 1
        regression_summary.append(
            {
                "query": query,
                "query_type": source_record.get("query_type"),
                "old_first_hit_rank": old_record.get("first_hit_rank"),
                "new_first_hit_rank": new_record.get("first_hit_rank"),
                "added_pairs": added_for_query,
                "skipped_answer_like": skipped_answer_like,
                "top_doc_ids": doc_ids,
            }
        )
        stats["regression_cases"] += 1
        if added_for_query:
            stats["regression_cases_with_pairs"] += 1
    return output, stats, regression_summary


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(values)

    def pct(percent: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        index = round((len(ordered) - 1) * percent)
        return ordered[index]

    return {
        "min": ordered[0],
        "p10": pct(0.10),
        "median": statistics.median(ordered),
        "p90": pct(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def structural_quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_keys: Counter[tuple[str, str, str]] = Counter(pair_key(record) for record in records)
    score_diffs: list[float] = []
    positive_overlaps: list[float] = []
    negative_overlaps: list[float] = []
    positive_lengths: list[int] = []
    negative_lengths: list[int] = []
    issues: Counter[str] = Counter()
    overlaps_by_type: dict[str, list[float]] = defaultdict(list)
    score_diffs_by_type: dict[str, list[float]] = defaultdict(list)
    for record in records:
        query = str(record.get("query") or "").strip()
        positive = str(record.get("positive") or "").strip()
        negative = str(record.get("negative") or "").strip()
        negative_type = str(record.get("negative_type") or "unknown")
        if not query:
            issues["empty_query"] += 1
        if not positive:
            issues["empty_positive"] += 1
        if not negative:
            issues["empty_negative"] += 1
        if normalize_text(positive) == normalize_text(negative):
            issues["positive_equal_negative"] += 1
        if chain_set(record, "answer_evidence") and chain_set(record, "answer_evidence").issubset(chain_set(record, "negative_chain")):
            issues["negative_chain_contains_answer_evidence"] += 1
        if chain_set(record, "positive_chain") and chain_set(record, "positive_chain") == chain_set(record, "negative_chain"):
            issues["same_positive_negative_chain_set"] += 1
        score_diff = float(record.get("positive_score") or 0.0) - float(record.get("negative_score") or 0.0)
        pos_overlap = answer_overlap(record, positive)
        neg_overlap = answer_overlap(record, negative)
        score_diffs.append(score_diff)
        positive_overlaps.append(pos_overlap)
        negative_overlaps.append(neg_overlap)
        overlaps_by_type[negative_type].append(pos_overlap)
        score_diffs_by_type[negative_type].append(score_diff)
        positive_lengths.append(len(positive))
        negative_lengths.append(len(negative))
    duplicate_pairs = sum(count - 1 for count in pair_keys.values() if count > 1)
    if duplicate_pairs:
        issues["duplicate_pair_keys"] = duplicate_pairs
    return {
        "records": len(records),
        "unique_queries": len({str(record.get("query") or "") for record in records}),
        "negative_types": dict(Counter(str(record.get("negative_type") or "unknown") for record in records)),
        "query_types": dict(Counter(str(record.get("query_type") or "unknown") for record in records)),
        "mix_sources": dict(Counter(str(record.get("mix_source") or "rank_clean_v4_base") for record in records)),
        "issues": dict(issues),
        "score_diff": quantiles(score_diffs),
        "positive_answer_overlap": quantiles(positive_overlaps),
        "negative_answer_overlap": quantiles(negative_overlaps),
        "positive_length_chars": quantiles([float(value) for value in positive_lengths]),
        "negative_length_chars": quantiles([float(value) for value in negative_lengths]),
        "positive_overlap_by_negative_type": {
            key: quantiles(values) for key, values in sorted(overlaps_by_type.items())
        },
        "score_diff_by_negative_type": {
            key: quantiles(values) for key, values in sorted(score_diffs_by_type.items())
        },
    }


def excerpt(text: str, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def write_sample_review(
    path: Path,
    records: list[dict[str, Any]],
    *,
    samples_per_type: int,
    seed: int,
    excerpt_chars: int,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("negative_type") or "unknown")].append(record)
    rng = random.Random(seed)
    review_rows: list[dict[str, Any]] = []
    for negative_type, items in sorted(grouped.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        for record in shuffled[:samples_per_type]:
            review_rows.append(
                {
                    "negative_type": negative_type,
                    "mix_source": record.get("mix_source", "rank_clean_v4_base"),
                    "mix_quality_note": record.get("mix_quality_note", ""),
                    "query_type": record.get("query_type"),
                    "source_name": record.get("source_name"),
                    "query": record.get("query"),
                    "answer": record.get("answer"),
                    "positive_overlap": round(answer_overlap(record, str(record.get("positive") or "")), 4),
                    "negative_overlap": round(answer_overlap(record, str(record.get("negative") or "")), 4),
                    "positive_score": record.get("positive_score"),
                    "negative_score": record.get("negative_score"),
                    "positive_chain": record.get("positive_chain"),
                    "negative_chain": record.get("negative_chain"),
                    "positive_excerpt": excerpt(str(record.get("positive") or ""), excerpt_chars),
                    "negative_excerpt": excerpt(str(record.get("negative") or ""), excerpt_chars),
                }
            )
    write_jsonl(path, review_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed low-latency reranker dataset from clean v4 and selected old signals.")
    parser.add_argument("--v4-dir", type=Path, default=Path("data/processed/evidence_chain_reranker/rank_clean_v4"))
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/evidence_chain_reranker/rank_mix_v5"))
    parser.add_argument("--documents", type=Path, default=Path("indexes/arknights_story/documents.jsonl"))
    parser.add_argument(
        "--old-eval-result",
        type=Path,
        default=Path("outputs/eval_multiround_retrieval/api_deepseek_same_offset50_2round.json"),
    )
    parser.add_argument(
        "--new-eval-result",
        type=Path,
        default=Path("outputs/eval_multiround_retrieval/rank_clean_v4_api_deepseek_same_offset50_2round.json"),
    )
    parser.add_argument("--min-shuffled-positive-answer-overlap", type=float, default=0.05)
    parser.add_argument("--shuffled-negative-score", type=float, default=0.95)
    parser.add_argument("--max-shuffled-per-query", type=int, default=1)
    parser.add_argument("--include-old-clean-dropped", action="store_true")
    parser.add_argument("--min-old-clean-positive-answer-overlap", type=float, default=0.05)
    parser.add_argument("--max-old-clean-per-query", type=int, default=1)
    parser.add_argument("--min-regression-rank-delta", type=int, default=5)
    parser.add_argument("--min-regression-positive-answer-overlap", type=float, default=0.05)
    parser.add_argument("--regression-negative-answer-overlap-margin", type=float, default=0.02)
    parser.add_argument("--regression-max-docs-per-round", type=int, default=5)
    parser.add_argument("--regression-max-docs-per-query", type=int, default=8)
    parser.add_argument("--regression-negative-score", type=float, default=0.9)
    parser.add_argument("--max-negatives-per-query", type=int, default=10)
    parser.add_argument("--training-text-max-chars", type=int, default=1200)
    parser.add_argument("--no-downweight-answer-evidence-negatives", action="store_true")
    parser.add_argument("--answer-evidence-negative-score", type=float, default=0.9)
    parser.add_argument("--no-downweight-partial-answer-evidence-negatives", action="store_true")
    parser.add_argument("--partial-answer-evidence-negative-score", type=float, default=0.9)
    parser.add_argument("--no-downweight-partial-answer-negatives", action="store_true")
    parser.add_argument("--partial-answer-negative-score", type=float, default=0.9)
    parser.add_argument("--no-downweight-text-similar-negatives", action="store_true")
    parser.add_argument("--text-similar-negative-score", type=float, default=0.9)
    parser.add_argument("--text-similar-min-similarity", type=float, default=0.3)
    parser.add_argument("--no-downweight-answer-like-negatives", action="store_true")
    parser.add_argument("--answer-like-negative-score", type=float, default=0.9)
    parser.add_argument("--answer-like-min-score-diff", type=float, default=0.15)
    parser.add_argument("--answer-like-min-negative-overlap", type=float, default=0.05)
    parser.add_argument("--answer-like-overlap-ratio", type=float, default=0.8)
    parser.add_argument("--sample-review-per-type", type=int, default=30)
    parser.add_argument("--sample-review-excerpt-chars", type=int, default=380)
    parser.add_argument("--seed", type=int, default=20260528)
    return parser.parse_args()


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value
    return output


def main() -> int:
    args = parse_args()
    v4_dir = resolve_path(args.v4_dir)
    old_dir = resolve_path(args.old_dir)
    output_dir = resolve_path(args.output_dir)
    documents_path = resolve_path(args.documents)
    old_eval_path = resolve_path(args.old_eval_result)
    new_eval_path = resolve_path(args.new_eval_result)

    v4_pairwise = read_jsonl(v4_dir / "reranker_pairwise.jsonl")
    old_pairwise = read_jsonl(old_dir / "reranker_pairwise.jsonl")
    old_listwise = read_jsonl(old_dir / "reranker_listwise.jsonl")
    listwise_by_query = {str(record.get("query") or ""): record for record in old_listwise}
    documents = load_documents(documents_path)

    output_pairwise = [dict(record) for record in v4_pairwise]
    seen = {pair_key(record) for record in output_pairwise}
    stats: Counter[str] = Counter({"rank_clean_v4_base": len(output_pairwise)})

    shuffled_pairs, shuffled_stats = build_shuffled_order_pairs(
        old_pairwise,
        min_positive_answer_overlap=args.min_shuffled_positive_answer_overlap,
        max_per_query=args.max_shuffled_per_query,
        negative_score=args.shuffled_negative_score,
    )
    add_records(output_pairwise, seen, shuffled_pairs, stats, stat_prefix="shuffled_order")

    old_clean_stats: Counter[str] = Counter()
    if args.include_old_clean_dropped:
        old_clean_pairs, old_clean_stats = build_old_clean_dropped_pairs(
            old_pairwise,
            {pair_key(record) for record in v4_pairwise},
            min_positive_answer_overlap=args.min_old_clean_positive_answer_overlap,
            max_per_query=args.max_old_clean_per_query,
        )
        add_records(output_pairwise, seen, old_clean_pairs, stats, stat_prefix="old_clean_dropped")

    regression_pairs, regression_stats, regression_summary = build_regression_pairs(
        old_eval=load_eval_records(old_eval_path),
        new_eval=load_eval_records(new_eval_path),
        listwise_by_query=listwise_by_query,
        documents=documents,
        min_rank_delta=args.min_regression_rank_delta,
        min_positive_answer_overlap=args.min_regression_positive_answer_overlap,
        negative_answer_overlap_margin=args.regression_negative_answer_overlap_margin,
        max_docs_per_round=args.regression_max_docs_per_round,
        max_docs_per_query=args.regression_max_docs_per_query,
        regression_negative_score=args.regression_negative_score,
    )
    add_records(output_pairwise, seen, regression_pairs, stats, stat_prefix="regression")

    compact_stats = compact_pairwise_texts(
        output_pairwise,
        max_chars=args.training_text_max_chars,
    )
    output_pairwise, post_compact_dedupe_stats = dedupe_pairwise_records(output_pairwise)

    downweight_stats: Counter[str] = Counter()
    if not args.no_downweight_answer_evidence_negatives:
        downweight_stats = downweight_answer_evidence_negatives(
            output_pairwise,
            negative_score=args.answer_evidence_negative_score,
        )
    partial_answer_evidence_downweight_stats: Counter[str] = Counter()
    if not args.no_downweight_partial_answer_evidence_negatives:
        partial_answer_evidence_downweight_stats = downweight_partial_answer_evidence_negatives(
            output_pairwise,
            negative_score=args.partial_answer_evidence_negative_score,
        )
    partial_answer_downweight_stats: Counter[str] = Counter()
    if not args.no_downweight_partial_answer_negatives:
        partial_answer_downweight_stats = downweight_partial_answer_negatives(
            output_pairwise,
            negative_score=args.partial_answer_negative_score,
        )
    text_similar_downweight_stats: Counter[str] = Counter()
    if not args.no_downweight_text_similar_negatives:
        text_similar_downweight_stats = downweight_text_similar_negatives(
            output_pairwise,
            negative_score=args.text_similar_negative_score,
            min_similarity=args.text_similar_min_similarity,
        )
    answer_like_downweight_stats: Counter[str] = Counter()
    if not args.no_downweight_answer_like_negatives:
        answer_like_downweight_stats = downweight_answer_like_negatives(
            output_pairwise,
            negative_score=args.answer_like_negative_score,
            min_score_diff=args.answer_like_min_score_diff,
            min_negative_overlap=args.answer_like_min_negative_overlap,
            overlap_ratio=args.answer_like_overlap_ratio,
        )

    output_listwise = build_listwise_from_pairs(
        source_records=old_listwise,
        pairwise_records=output_pairwise,
        max_negatives_per_query=args.max_negatives_per_query,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reranker_pairwise.jsonl", output_pairwise)
    write_jsonl(output_dir / "reranker_listwise.jsonl", output_listwise)
    write_jsonl(output_dir / "regression_cases.jsonl", regression_summary)
    write_sample_review(
        output_dir / "sample_review.jsonl",
        output_pairwise,
        samples_per_type=args.sample_review_per_type,
        seed=args.seed,
        excerpt_chars=args.sample_review_excerpt_chars,
    )

    quality_report = structural_quality_report(output_pairwise)
    manifest = {
        "v4_dir": str(v4_dir),
        "old_dir": str(old_dir),
        "output_dir": str(output_dir),
        "documents": str(documents_path),
        "old_eval_result": str(old_eval_path),
        "new_eval_result": str(new_eval_path),
        "v4_pairwise_records": len(v4_pairwise),
        "old_pairwise_records": len(old_pairwise),
        "output_pairwise_records": len(output_pairwise),
        "output_listwise_records": len(output_listwise),
        "build_stats": dict(stats),
        "shuffled_selection_stats": dict(shuffled_stats),
        "old_clean_selection_stats": dict(old_clean_stats),
        "regression_selection_stats": dict(regression_stats),
        "compact_text_stats": dict(compact_stats),
        "post_compact_dedupe_stats": dict(post_compact_dedupe_stats),
        "downweight_answer_evidence_negative_stats": dict(downweight_stats),
        "downweight_partial_answer_evidence_negative_stats": dict(partial_answer_evidence_downweight_stats),
        "downweight_partial_answer_negative_stats": dict(partial_answer_downweight_stats),
        "downweight_text_similar_negative_stats": dict(text_similar_downweight_stats),
        "downweight_answer_like_negative_stats": dict(answer_like_downweight_stats),
        "quality_report": quality_report,
        "args": jsonable_args(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
