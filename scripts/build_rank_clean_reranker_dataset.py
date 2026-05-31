#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHITESPACE_RE = re.compile(r"\s+")
LEXICAL_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.-]+")
STOP_CHARS = set("的是了和与及或在为被把对中上下一一个这那其已以而但也都并之时后前就能会要可让给从向于将更很最等因由用所")
GENERIC_TERMS = {
    "为什么",
    "什么",
    "如何",
    "具体",
    "原因",
    "现有",
    "证据",
    "显示",
    "说明",
    "表示",
    "认为",
    "通过",
    "因为",
    "所以",
    "最终",
    "这个",
    "那个",
    "他们",
    "自己",
    "问题",
    "直接",
    "核心",
    "关键",
    "剧情",
    "一事",
    "一种",
    "进行",
    "时候",
    "没有",
    "不是",
    "就是",
    "可以",
    "需要",
    "选择",
    "事情",
    "发生",
    "答案",
    "相关",
    "指出",
    "提到",
    "意味着",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub("", text or "")


def text_key(text: str) -> str:
    return normalize_text(text)[:600]


def lexical_grams(text: str) -> set[str]:
    terms: set[str] = set()
    for token in LEXICAL_TOKEN_RE.findall(text or ""):
        if re.fullmatch(r"[A-Za-z0-9_.-]+", token):
            normalized = token.lower()
            if re.fullmatch(r"e\d+", normalized):
                continue
            if len(normalized) >= 2 and normalized not in GENERIC_TERMS:
                terms.add(normalized)
            continue
        for size in (2, 3, 4):
            for index in range(len(token) - size + 1):
                gram = token[index : index + size]
                if gram in GENERIC_TERMS:
                    continue
                if all(char in STOP_CHARS for char in gram):
                    continue
                terms.add(gram)
    return terms


def answer_overlap(record: dict[str, Any], text: str) -> float:
    answer_terms = lexical_grams(
        f"{record.get('answer') or ''} {record.get('answer_focus') or ''}"
    )
    if not answer_terms:
        return 0.0
    text_terms = lexical_grams(text)
    return len(answer_terms & text_terms) / len(answer_terms)


def load_documents(path: Path) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = str(doc.get("id") or "").strip()
            if doc_id:
                docs[doc_id] = doc
    return docs


def infer_activity_id_from_source_name(source_name: str) -> str:
    value = str(source_name or "").strip()
    if "_part" in value:
        value = value.split("_part", 1)[0]
    if value.endswith(".json"):
        value = value[:-5]
    return value


def group_documents_by_activity(documents: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for doc in documents.values():
        activity_id = str(doc.get("activity_id") or "").strip()
        if activity_id:
            grouped.setdefault(activity_id, []).append(doc)
    return grouped


def gold_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in record.get("candidates") or []:
        if candidate.get("label") == "positive" or candidate.get("type") == "gold":
            return candidate
    return None


def should_keep_original_pair(record: dict[str, Any], *, include_shuffled_order: bool) -> tuple[bool, str]:
    negative_type = str(record.get("negative_type") or "").strip()
    positive_chain = set(str(item) for item in (record.get("positive_chain") or []))
    negative_chain = set(str(item) for item in (record.get("negative_chain") or []))
    answer_evidence = set(str(item) for item in (record.get("answer_evidence") or []))
    negative_score = float(record.get("negative_score") or 0.0)
    positive = normalize_text(str(record.get("positive") or ""))
    negative = normalize_text(str(record.get("negative") or ""))

    if not positive or not negative or positive == negative:
        return False, "empty_or_equal_text"
    if not include_shuffled_order and negative_type == "shuffled_order":
        return False, "drop_shuffled_order"
    if answer_evidence and answer_evidence.issubset(negative_chain) and negative_score >= 0.6:
        return False, "negative_contains_answer_evidence"
    if positive_chain and negative_chain == positive_chain:
        return False, "negative_same_chain_set"
    return True, "kept"


def has_usable_positive(record: dict[str, Any], *, min_answer_overlap: float) -> bool:
    if min_answer_overlap <= 0:
        return True
    return answer_overlap(record, str(record.get("positive") or "")) >= min_answer_overlap


def make_pair(
    *,
    query: str,
    query_type: str,
    source_name: str,
    positive: str,
    negative: str,
    negative_type: str,
    answer: str = "",
    answer_evidence: list[str] | None = None,
    answer_focus: str = "",
    positive_score: float = 1.0,
    negative_score: float = 0.25,
    positive_chain: list[str] | None = None,
    negative_chain: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "query": query,
        "query_type": query_type,
        "source_name": source_name,
        "positive": positive,
        "negative": negative,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "negative_type": negative_type,
        "answer": answer,
        "answer_evidence": answer_evidence or [],
        "answer_focus": answer_focus,
        "positive_chain": positive_chain or [],
        "negative_chain": negative_chain or [],
    }


def record_metadata(source_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": str(source_record.get("query") or ""),
        "query_type": str(source_record.get("query_type") or "unknown"),
        "source_name": str(source_record.get("source_name") or ""),
        "answer": str(source_record.get("answer") or ""),
        "answer_evidence": list(source_record.get("answer_evidence") or []),
        "answer_focus": str(source_record.get("answer_focus") or ""),
    }


def doc_training_text(doc: dict[str, Any]) -> str:
    title = " / ".join(
        str(doc.get(key) or "").strip()
        for key in ("activity_name", "story_name", "stage_code", "avg_tag")
        if str(doc.get(key) or "").strip()
    )
    text = str(doc.get("clean_text") or doc.get("search_text") or "").strip()
    if title:
        return f"{title}\n{text}"
    return text


def best_recovered_positive(
    source_record: dict[str, Any],
    *,
    current_positive: str,
    documents_by_activity: dict[str, list[dict[str, Any]]],
    min_overlap: float,
) -> tuple[str, str, float] | None:
    activity_id = infer_activity_id_from_source_name(str(source_record.get("source_name") or ""))
    if not activity_id:
        return None
    best_doc_id = ""
    best_text = ""
    best_overlap = 0.0
    current_key = text_key(current_positive)
    for doc in documents_by_activity.get(activity_id, []):
        candidate_text = doc_training_text(doc)
        if not candidate_text or text_key(candidate_text) == current_key:
            continue
        overlap = answer_overlap(source_record, candidate_text)
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc_id = str(doc.get("id") or "")
            best_text = candidate_text
    if best_overlap < min_overlap or not best_text:
        return None
    return best_doc_id, best_text, best_overlap


def build_eval_hard_negative_pairs(
    *,
    eval_paths: list[Path],
    listwise_records: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    max_top_docs: int,
    min_positive_answer_overlap: float,
    skip_negative_answer_overlap_margin: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for path in eval_paths:
        if not path.exists():
            stats[f"missing_eval:{path}"] += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for eval_record in payload.get("records") or []:
            query = str(eval_record.get("query") or "").strip()
            source_record = listwise_records.get(query)
            if source_record is None:
                stats["eval_query_not_in_listwise"] += 1
                continue
            gold = gold_candidate(source_record)
            if gold is None:
                stats["eval_missing_gold"] += 1
                continue
            positive = str(gold.get("text") or "").strip()
            if not positive:
                stats["eval_empty_gold"] += 1
                continue
            positive_overlap = answer_overlap({"answer": source_record.get("answer"), "answer_focus": source_record.get("answer_focus")}, positive)
            if positive_overlap < min_positive_answer_overlap:
                stats["eval_weak_positive"] += 1
                continue
            top_doc_ids: list[str] = []
            for round_info in eval_record.get("rounds") or []:
                top_doc_ids.extend(str(item or "").strip() for item in (round_info.get("top_doc_ids") or [])[:max_top_docs])
            top_doc_ids = [item for item in dict.fromkeys(top_doc_ids) if item]
            if not top_doc_ids:
                stats["eval_no_top_docs"] += 1
                continue
            hit_rank = eval_record.get("first_hit_rank")
            is_low_rank_hit = isinstance(hit_rank, int) and hit_rank > 5
            is_miss = not bool((eval_record.get("cumulative_hit") or {}).get("50"))
            if not is_low_rank_hit and not is_miss:
                continue
            for doc_id in top_doc_ids:
                doc = documents.get(doc_id)
                if doc is None:
                    stats["eval_top_doc_missing"] += 1
                    continue
                negative = doc_training_text(doc)
                if not negative:
                    stats["eval_top_doc_empty"] += 1
                    continue
                if text_key(negative) == text_key(positive):
                    continue
                negative_overlap = answer_overlap(
                    {"answer": source_record.get("answer"), "answer_focus": source_record.get("answer_focus")},
                    negative,
                )
                if negative_overlap + skip_negative_answer_overlap_margin >= positive_overlap:
                    stats["eval_top_doc_answer_like"] += 1
                    continue
                key = (query, text_key(positive), text_key(negative))
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    make_pair(
                        query=query,
                        query_type=str(source_record.get("query_type") or "unknown"),
                        source_name=str(source_record.get("source_name") or "eval_hard_negative"),
                        positive=positive,
                        negative=negative,
                        negative_type="online_top_distractor_miss" if is_miss else "online_top_distractor_low_rank",
                        answer=str(source_record.get("answer") or ""),
                        answer_evidence=list(source_record.get("answer_evidence") or []),
                        answer_focus=str(source_record.get("answer_focus") or ""),
                        negative_score=0.35 if is_miss else 0.45,
                        positive_chain=list(gold.get("chain") or []),
                        negative_chain=[doc_id],
                    )
                )
                stats["eval_hard_negative_pairs"] += 1
    return output, stats


def build_eval_answer_like_positive_pairs(
    *,
    eval_paths: list[Path],
    listwise_records: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    max_top_docs: int,
    min_positive_answer_overlap: float,
    answer_like_margin: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()

    for path in eval_paths:
        if not path.exists():
            stats[f"missing_eval:{path}"] += 1
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for eval_record in payload.get("records") or []:
            query = str(eval_record.get("query") or "").strip()
            source_record = listwise_records.get(query)
            if source_record is None:
                continue
            gold = gold_candidate(source_record)
            if gold is None:
                continue
            gold_text = str(gold.get("text") or "").strip()
            if not gold_text:
                continue
            gold_overlap = answer_overlap(source_record, gold_text)
            top_doc_ids: list[str] = []
            for round_info in eval_record.get("rounds") or []:
                top_doc_ids.extend(str(item or "").strip() for item in (round_info.get("top_doc_ids") or [])[:max_top_docs])
            for doc_id in dict.fromkeys(item for item in top_doc_ids if item):
                doc = documents.get(doc_id)
                if doc is None:
                    continue
                positive = doc_training_text(doc)
                if not positive:
                    continue
                positive_overlap = answer_overlap(source_record, positive)
                if positive_overlap < min_positive_answer_overlap:
                    continue
                if positive_overlap + answer_like_margin < gold_overlap:
                    continue
                key = (query, text_key(positive), text_key(gold_text))
                if key in seen or text_key(positive) == text_key(gold_text):
                    continue
                seen.add(key)
                meta = record_metadata(source_record)
                output.append(
                    make_pair(
                        **meta,
                        positive=positive,
                        negative=gold_text,
                        negative_type="weaker_original_gold",
                        positive_score=1.0,
                        negative_score=max(0.2, min(0.75, gold_overlap)),
                        positive_chain=[doc_id],
                        negative_chain=list(gold.get("chain") or []),
                    )
                )
                stats["eval_answer_like_positive_pairs"] += 1
    return output, stats


def build_recovered_positive_pairs(
    *,
    source_records: list[dict[str, Any]],
    documents_by_activity: dict[str, list[dict[str, Any]]],
    min_original_positive_overlap: float,
    min_recovered_positive_overlap: float,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    output: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    for source_record in source_records:
        gold = gold_candidate(source_record)
        if gold is None:
            continue
        gold_text = str(gold.get("text") or "").strip()
        if not gold_text:
            continue
        gold_overlap = answer_overlap(source_record, gold_text)
        if gold_overlap >= min_original_positive_overlap:
            continue
        recovered = best_recovered_positive(
            source_record,
            current_positive=gold_text,
            documents_by_activity=documents_by_activity,
            min_overlap=min_recovered_positive_overlap,
        )
        if recovered is None:
            stats["recovered_positive_not_found"] += 1
            continue
        doc_id, positive, positive_overlap = recovered
        key = (str(source_record.get("query") or ""), text_key(positive), text_key(gold_text))
        if key in seen:
            continue
        seen.add(key)
        meta = record_metadata(source_record)
        output.append(
            make_pair(
                **meta,
                positive=positive,
                negative=gold_text,
                negative_type="weak_original_gold",
                positive_score=1.0,
                negative_score=max(0.1, min(0.5, gold_overlap)),
                positive_chain=[doc_id],
                negative_chain=list(gold.get("chain") or []),
            )
        )
        stats["recovered_positive_pairs"] += 1
        stats[f"recovered_overlap_ge_{int(positive_overlap * 100):02d}"] += 1
    return output, stats


def build_listwise_from_pairs(
    *,
    source_records: list[dict[str, Any]],
    pairwise_records: list[dict[str, Any]],
    max_negatives_per_query: int,
) -> list[dict[str, Any]]:
    negatives_by_query: dict[str, list[dict[str, Any]]] = {}
    for pair in pairwise_records:
        negatives_by_query.setdefault(str(pair.get("query") or ""), []).append(pair)

    output: list[dict[str, Any]] = []
    for source in source_records:
        query = str(source.get("query") or "").strip()
        gold = gold_candidate(source)
        if not query or gold is None:
            continue
        candidates = [
            {
                "text": str(gold.get("text") or "").strip(),
                "score": 1.0,
                "label": "positive",
                "type": "gold",
                "chain": list(gold.get("chain") or []),
            }
        ]
        seen_texts = {text_key(candidates[0]["text"])}
        for pair in sorted(
            negatives_by_query.get(query, []),
            key=lambda item: float(item.get("negative_score") or 0.0),
            reverse=True,
        ):
            negative = str(pair.get("negative") or "").strip()
            key = text_key(negative)
            if not negative or key in seen_texts:
                continue
            seen_texts.add(key)
            candidates.append(
                {
                    "text": negative,
                    "score": float(pair.get("negative_score") or 0.0),
                    "label": "negative",
                    "type": str(pair.get("negative_type") or "negative"),
                    "chain": list(pair.get("negative_chain") or []),
                }
            )
            if len(candidates) >= max_negatives_per_query + 1:
                break
        if len(candidates) < 2:
            continue
        record = dict(source)
        record["candidates"] = candidates
        output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rank-oriented cleaned reranker dataset.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/rank_clean_v1"),
    )
    parser.add_argument("--documents", type=Path, default=Path("indexes/arknights_story/documents.jsonl"))
    parser.add_argument(
        "--eval-result",
        type=Path,
        action="append",
        default=[],
        help="Recent eval JSON used to add online top distractor hard negatives.",
    )
    parser.add_argument("--max-top-docs", type=int, default=5)
    parser.add_argument("--max-negatives-per-query", type=int, default=8)
    parser.add_argument("--include-shuffled-order", action="store_true")
    parser.add_argument(
        "--min-positive-answer-overlap",
        type=float,
        default=0.0,
        help="Drop pairs whose positive evidence has too little lexical overlap with answer/answer_focus.",
    )
    parser.add_argument(
        "--online-negative-answer-overlap-margin",
        type=float,
        default=0.02,
        help="Skip online hard negatives that look as answer-bearing as the positive evidence.",
    )
    parser.add_argument(
        "--min-recovered-positive-answer-overlap",
        type=float,
        default=0.0,
        help="Add recovered positives from same-activity documents when original gold is weak.",
    )
    parser.add_argument(
        "--add-online-answer-like-positives",
        action="store_true",
        help="Turn answer-like top retrieved documents into positive-vs-original-gold pairs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = resolve_path(args.source_dir)
    output_dir = resolve_path(args.output_dir)
    documents_path = resolve_path(args.documents)

    source_pairwise = read_jsonl(source_dir / "reranker_pairwise.jsonl")
    source_listwise = read_jsonl(source_dir / "reranker_listwise.jsonl")
    documents = load_documents(documents_path)
    documents_by_activity = group_documents_by_activity(documents)

    cleaned_pairwise: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_pair_keys: set[tuple[str, str, str]] = set()
    for record in source_pairwise:
        keep, reason = should_keep_original_pair(record, include_shuffled_order=args.include_shuffled_order)
        stats[reason] += 1
        if not keep:
            continue
        if not has_usable_positive(record, min_answer_overlap=args.min_positive_answer_overlap):
            stats["weak_positive_answer_overlap"] += 1
            continue
        query = str(record.get("query") or "")
        key = (query, text_key(str(record.get("positive") or "")), text_key(str(record.get("negative") or "")))
        if key in seen_pair_keys:
            stats["duplicate_pair"] += 1
            continue
        seen_pair_keys.add(key)
        cleaned_pairwise.append(record)

    listwise_by_query = {str(record.get("query") or ""): record for record in source_listwise}
    eval_paths = [resolve_path(path) for path in args.eval_result]
    eval_pairs: list[dict[str, Any]] = []
    eval_stats: Counter[str] = Counter()
    if eval_paths:
        eval_pairs, eval_stats = build_eval_hard_negative_pairs(
            eval_paths=eval_paths,
            listwise_records=listwise_by_query,
            documents=documents,
            max_top_docs=args.max_top_docs,
            min_positive_answer_overlap=args.min_positive_answer_overlap,
            skip_negative_answer_overlap_margin=args.online_negative_answer_overlap_margin,
        )
        for pair in eval_pairs:
            key = (str(pair.get("query") or ""), text_key(str(pair.get("positive") or "")), text_key(str(pair.get("negative") or "")))
            if key in seen_pair_keys:
                stats["duplicate_eval_pair"] += 1
                continue
            seen_pair_keys.add(key)
            cleaned_pairwise.append(pair)

    recovered_pairs: list[dict[str, Any]] = []
    recovered_stats: Counter[str] = Counter()
    if args.min_recovered_positive_answer_overlap > 0 and args.min_positive_answer_overlap > 0:
        recovered_pairs, recovered_stats = build_recovered_positive_pairs(
            source_records=source_listwise,
            documents_by_activity=documents_by_activity,
            min_original_positive_overlap=args.min_positive_answer_overlap,
            min_recovered_positive_overlap=args.min_recovered_positive_answer_overlap,
        )
        for pair in recovered_pairs:
            key = (str(pair.get("query") or ""), text_key(str(pair.get("positive") or "")), text_key(str(pair.get("negative") or "")))
            if key in seen_pair_keys:
                stats["duplicate_recovered_pair"] += 1
                continue
            seen_pair_keys.add(key)
            cleaned_pairwise.append(pair)

    online_positive_pairs: list[dict[str, Any]] = []
    online_positive_stats: Counter[str] = Counter()
    if args.add_online_answer_like_positives and eval_paths:
        online_positive_pairs, online_positive_stats = build_eval_answer_like_positive_pairs(
            eval_paths=eval_paths,
            listwise_records=listwise_by_query,
            documents=documents,
            max_top_docs=args.max_top_docs,
            min_positive_answer_overlap=args.min_positive_answer_overlap,
            answer_like_margin=args.online_negative_answer_overlap_margin,
        )
        for pair in online_positive_pairs:
            key = (str(pair.get("query") or ""), text_key(str(pair.get("positive") or "")), text_key(str(pair.get("negative") or "")))
            if key in seen_pair_keys:
                stats["duplicate_online_positive_pair"] += 1
                continue
            seen_pair_keys.add(key)
            cleaned_pairwise.append(pair)

    cleaned_listwise = build_listwise_from_pairs(
        source_records=source_listwise,
        pairwise_records=cleaned_pairwise,
        max_negatives_per_query=args.max_negatives_per_query,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reranker_pairwise.jsonl", cleaned_pairwise)
    write_jsonl(output_dir / "reranker_listwise.jsonl", cleaned_listwise)

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "documents": str(documents_path),
        "eval_results": [str(path) for path in eval_paths],
        "source_pairwise_records": len(source_pairwise),
        "source_listwise_records": len(source_listwise),
        "cleaned_pairwise_records": len(cleaned_pairwise),
        "cleaned_listwise_records": len(cleaned_listwise),
        "filter_stats": dict(stats),
        "eval_hard_negative_stats": dict(eval_stats),
        "recovered_positive_stats": dict(recovered_stats),
        "online_answer_like_positive_stats": dict(online_positive_stats),
        "min_positive_answer_overlap": args.min_positive_answer_overlap,
        "online_negative_answer_overlap_margin": args.online_negative_answer_overlap_margin,
        "min_recovered_positive_answer_overlap": args.min_recovered_positive_answer_overlap,
        "add_online_answer_like_positives": args.add_online_answer_like_positives,
        "negative_types": dict(Counter(str(record.get("negative_type") or "") for record in cleaned_pairwise)),
        "query_types": dict(Counter(str(record.get("query_type") or "") for record in cleaned_pairwise)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
