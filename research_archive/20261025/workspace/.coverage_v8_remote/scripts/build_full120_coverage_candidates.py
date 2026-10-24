#!/usr/bin/env python3
"""Export strict 120-item ASA candidate pools for reranker teachers.

The existing rank_v8 raw files may contain fewer than 120 candidate passages
after text-level de-duplication.  This script preserves the existing candidate
order, then queries the current hybrid retrieval stack with a larger pre-rerank
pool and appends additional unique candidate texts until every record has
exactly ``--target-candidates`` candidates.

Candidate retrieval remains the production BM25 + FAISS + MiniRAG/scoped
retrieval stack.  This script does not ask a teacher to generate candidates or
change retrieval.  It only exports that fixed pool (supplementing legacy short
exports through the same pre-rerank stack when explicitly requested).

Outputs are independent JSONL files intended for GLM/other teacher labelling;
the original DeepSeek/Flash checkpoints and raw files are not modified.  Gold
removal is deliberately an explicit migration step; use
``sanitize_reranker_teacher_inputs.py`` before sending an export to a teacher.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

CHAIN_TAG_RE = re.compile(r"\[E\d+\]\s*")
WHITESPACE_RE = re.compile(r"\s+")
FORBIDDEN_RECORD_FIELDS = {
    "gold_evidence",
    "gold_evidence_ids",
    "coverable_evidence_ids",
    "answer_evidence",
}
FORBIDDEN_CANDIDATE_FIELDS = {"covered_evidence_ids", "relevance", "label"}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub("", CHAIN_TAG_RE.sub("", text))


def resolve_project_path(project_root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def configure_imports(project_root: Path) -> None:
    train_overlay = project_root / ".python_packages" / "train"
    train_override = project_root / ".vendor" / "train_override"
    if train_overlay.exists():
        sys.path.insert(0, str(train_overlay))
    if train_override.exists():
        sys.path.insert(0, str(train_override))
    sys.path.insert(0, str(project_root / "src"))


def compact_candidate(candidate: dict[str, Any], *, candidate_id: str) -> dict[str, Any]:
    copied = dict(candidate)
    copied["candidate_id"] = candidate_id
    copied["text"] = str(copied.get("text") or "")
    copied.setdefault("kind", "document")
    copied.setdefault("doc_id", "")
    copied.setdefault("story_id", "")
    copied.setdefault("stage_code", "")
    copied.setdefault("retrieval_rank", 0)
    return copied


def strip_supervision(record: dict[str, Any]) -> dict[str, Any]:
    allowed_record_fields = {
        "query",
        "query_type",
        "source_name",
        "story_family",
        "answer",
    }
    copied = {key: value for key, value in record.items() if key in allowed_record_fields}
    copied["candidates"] = [
        {
            "candidate_id": str(candidate.get("candidate_id") or ""),
            "text": str(candidate.get("text") or ""),
            "kind": str(candidate.get("kind") or "document"),
        }
        for candidate in record.get("candidates") or []
    ]
    return copied


def candidate_representations(
    hit: dict[str, Any],
    *,
    rank: int,
    max_chain_chars: int,
) -> list[dict[str, Any]]:
    document = hit.get("document") or {}
    values = [
        (
            "document",
            str(document.get("search_text") or document.get("clean_text") or "").strip(),
        ),
        ("chain", str(hit.get("evidence_chain_text") or "").strip()),
    ]
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, text in values:
        if kind == "chain" and len(text) > max_chain_chars:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(
            {
                "candidate_id": "",
                "text": text,
                "kind": kind,
                "retrieval_rank": rank,
                "doc_id": str(document.get("id") or ""),
                "story_id": str(document.get("story_id") or ""),
                "stage_code": str(document.get("stage_code") or ""),
                "rerank_score": hit.get("rerank_score"),
                "fusion_score": hit.get("fusion_score"),
            }
        )
    return output


def build_query_config(args: argparse.Namespace):
    from goldenglow.config import QueryConfig

    return QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        minirag_top_k=args.minirag_top_k,
        fusion_top_k=args.candidate_pool_top_k,
        rerank_top_k=args.target_candidates,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights={
            "fact": 0.5,
            "relation": 1.0,
            "causality": 1.0,
            "reveal": 1.0,
            "reasoning": 0.75,
            "mystery": 0.75,
            "answerability": 0.75,
        },
        minirag_fusion_mode=args.minirag_fusion_mode,
        reranker_candidate_top_k=args.candidate_pool_top_k,
        enable_neighbor_expansion=args.enable_neighbor_expansion,
        neighbor_max_seed_docs=args.neighbor_max_seed_docs,
        neighbor_story_window=args.neighbor_story_window,
        neighbor_activity_story_sort_window=args.neighbor_activity_story_sort_window,
        rerank_batch_size=args.rerank_batch_size,
    )


def load_retriever(args: argparse.Namespace):
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever

    project_root = args.project_root
    return ArknightsHybridRetriever.from_paths(
        embedding_model_path=resolve_project_path(project_root, args.embedding_model),
        reranker_model_path=None,
        documents_path=resolve_project_path(project_root, args.index_dir) / "documents.jsonl",
        faiss_index_path=resolve_project_path(project_root, args.index_dir) / "faiss.index",
        bm25_tokens_path=resolve_project_path(project_root, args.index_dir) / "bm25_tokens.pkl",
        minirag_index_path=resolve_project_path(project_root, args.minirag_index),
        device=args.device,
    )


def normalize_candidate_ids(record: dict[str, Any], target_candidates: int) -> dict[str, Any]:
    copied = dict(record)
    copied["candidates"] = [
        compact_candidate(candidate, candidate_id=f"C{index:03d}")
        for index, candidate in enumerate(copied.get("candidates") or [], start=1)
    ][:target_candidates]
    return copied


def supplement_record(
    record: dict[str, Any],
    *,
    retriever: Any,
    query_config: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = args.target_candidates
    query = str(record.get("query") or "").strip()
    output = dict(record)
    selected: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    seen_doc_kind: set[tuple[str, str]] = set()

    for raw_candidate in record.get("candidates") or []:
        text = str(raw_candidate.get("text") or "")
        normalized = normalize_text(text)
        doc_kind = (str(raw_candidate.get("doc_id") or ""), str(raw_candidate.get("kind") or "document"))
        if not normalized or normalized in seen_texts or doc_kind in seen_doc_kind:
            continue
        selected.append(dict(raw_candidate))
        seen_texts.add(normalized)
        seen_doc_kind.add(doc_kind)
        if len(selected) >= target:
            break

    original_unique = len(selected)
    retrieval_hits = 0
    appended = 0
    if len(selected) < target:
        hits = retriever.search_pre_rerank(query, config=query_config)
        retrieval_hits = len(hits)
        for rank, hit in enumerate(hits, start=1):
            for candidate in candidate_representations(
                hit, rank=rank, max_chain_chars=args.max_chain_chars
            ):
                normalized = normalize_text(candidate["text"])
                doc_kind = (str(candidate.get("doc_id") or ""), str(candidate.get("kind") or "document"))
                if not normalized or normalized in seen_texts or doc_kind in seen_doc_kind:
                    continue
                selected.append(candidate)
                seen_texts.add(normalized)
                seen_doc_kind.add(doc_kind)
                appended += 1
                if len(selected) >= target:
                    break
            if len(selected) >= target:
                break

    output["candidates"] = [
        compact_candidate(candidate, candidate_id=f"C{index:03d}")
        for index, candidate in enumerate(selected[:target], start=1)
    ]
    output["candidate_generation"] = {
        "source": "rank_v8_coverage_full120_supplement",
        "target_candidates": target,
        "original_candidate_count": len(record.get("candidates") or []),
        "original_unique_candidates": original_unique,
        "retrieval_hits": retrieval_hits,
        "supplemented_candidates": appended,
        "candidate_pool_top_k": args.candidate_pool_top_k,
        "dense_top_k": args.dense_top_k,
        "sparse_top_k": args.sparse_top_k,
        "minirag_top_k": args.minirag_top_k,
        "enable_neighbor_expansion": args.enable_neighbor_expansion,
    }
    diagnostic = {
        "query": query,
        "source_name": record.get("source_name"),
        "before": len(record.get("candidates") or []),
        "original_unique": original_unique,
        "after": len(output["candidates"]),
        "retrieval_hits": retrieval_hits,
        "supplemented": appended,
    }
    return output, diagnostic


def validate_records(records: list[dict[str, Any]], target: int) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    bad: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        candidates = record.get("candidates") or []
        count = len(candidates)
        counts[count] += 1
        ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
        texts = [normalize_text(str(candidate.get("text") or "")) for candidate in candidates]
        reason = ""
        if count != target:
            reason = f"candidate_count={count}"
        elif len(set(ids)) != target or ids != [f"C{i:03d}" for i in range(1, target + 1)]:
            reason = "candidate_id_set_invalid"
        elif any(not text for text in texts):
            reason = "empty_candidate_text"
        elif len(set(texts)) != target:
            reason = "duplicate_candidate_text"
        elif any(field in record for field in FORBIDDEN_RECORD_FIELDS):
            reason = "record_supervision_leak"
        elif any(
            field in candidate
            for candidate in candidates
            for field in FORBIDDEN_CANDIDATE_FIELDS
        ):
            reason = "candidate_supervision_leak"
        if reason:
            bad.append({"line": index, "query": str(record.get("query") or "")[:120], "reason": reason})
    return {"candidate_counts": dict(sorted(counts.items())), "bad": bad[:50], "bad_count": len(bad)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("${REMOTE_STORE}/ASA-ArknightStoryAgent"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/arknights_story"))
    parser.add_argument("--embedding-model", type=Path, default=Path("model/embeddings/bge-small-zh-v1.5"))
    parser.add_argument("--minirag-index", type=Path, default=Path("indexes/arknights_story_minirag_v3/graph.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-candidates", type=int, default=120)
    parser.add_argument("--candidate-pool-top-k", type=int, default=320)
    parser.add_argument("--dense-top-k", type=int, default=240)
    parser.add_argument("--sparse-top-k", type=int, default=240)
    parser.add_argument("--minirag-top-k", type=int, default=240)
    parser.add_argument("--minirag-weight", type=float, default=0.35)
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default="score")
    parser.add_argument("--enable-neighbor-expansion", action="store_true")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=24)
    parser.add_argument("--neighbor-story-window", type=int, default=2)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=1)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--max-chain-chars", type=int, default=2600)
    parser.add_argument(
        "--teacher-safe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Strip gold evidence and legacy labels while exporting. This is an "
            "explicit compatibility option; prefer the standalone sanitizer."
        ),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    input_path = args.input if args.input.is_absolute() else Path.cwd() / args.input
    output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    diagnostics_path = (
        args.diagnostics_output
        if args.diagnostics_output and args.diagnostics_output.is_absolute()
        else (Path.cwd() / args.diagnostics_output if args.diagnostics_output else output_path.with_suffix(".diagnostics.jsonl"))
    )

    configure_imports(args.project_root)
    records = read_jsonl(input_path)
    if args.limit:
        records = records[: args.limit]

    needs_retriever = any(len(record.get("candidates") or []) < args.target_candidates for record in records)
    retriever = None
    query_config = None
    if needs_retriever:
        started = time.time()
        print(f"Loading retriever for supplementing {input_path}...", file=sys.stderr, flush=True)
        retriever = load_retriever(args)
        query_config = build_query_config(args)
        print(f"Retriever loaded in {time.time() - started:.1f}s", file=sys.stderr, flush=True)

    output_records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        before = len(record.get("candidates") or [])
        if before >= args.target_candidates:
            output = normalize_candidate_ids(record, args.target_candidates)
            diagnostic = {
                "query": str(record.get("query") or ""),
                "source_name": record.get("source_name"),
                "before": before,
                "after": len(output.get("candidates") or []),
                "supplemented": 0,
                "retrieval_hits": 0,
            }
        else:
            if retriever is None or query_config is None:
                raise RuntimeError("retriever not loaded for short candidate record")
            output, diagnostic = supplement_record(
                record, retriever=retriever, query_config=query_config, args=args
            )
        after = len(output.get("candidates") or [])
        if args.teacher_safe:
            output = strip_supervision(output)
        output_records.append(output)
        diagnostics.append(diagnostic)
        stats[f"before_{before}"] += 1
        stats[f"after_{after}"] += 1
        if after != args.target_candidates:
            stats["failed_full120"] += 1
        if diagnostic.get("supplemented", 0):
            stats["supplemented_records"] += 1
        if index % 25 == 0:
            print(f"[{index}/{len(records)}] after={after} query={str(record.get('query') or '')[:40]}", file=sys.stderr, flush=True)

    validation = validate_records(output_records, args.target_candidates)
    write_jsonl(output_path, output_records)
    write_jsonl(diagnostics_path, diagnostics)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "diagnostics_output": str(diagnostics_path),
        "records": len(output_records),
        "target_candidates": args.target_candidates,
        "stats": dict(stats),
        "validation": validation,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if validation["bad_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
