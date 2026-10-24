#!/usr/bin/env python3
"""Build frozen query-only teacher pools from ASA's production retriever.

Only query metadata is inherited from the source JSONL.  Old candidates and
all answer/gold fields are discarded before retrieval.  The written records
contain exactly the fields accepted by the query-answerability teacher
protocol, so retrieval ranks, scores, and document identifiers cannot leak to
the labelling model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
SAFE_RECORD_FIELDS = {"query", "query_type", "source_name", "story_family", "candidates"}
SAFE_CANDIDATE_FIELDS = {"candidate_id", "text", "kind"}


def configure_import_path() -> None:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    use_overrides = (
        override_flag.lower() in {"1", "true", "yes", "on"}
        if override_flag is not None
        else "train" in Path(sys.executable).as_posix().lower()
    )
    if use_overrides and TRAIN_PYTHON_OVERLAY_DIR.exists():
        sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
    if use_overrides and TRAIN_OVERRIDE_DIR.exists():
        sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


configure_import_path()

from goldenglow.config import QueryConfig  # noqa: E402
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_records(path: Path, *, offset: int, limit: int | None) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in iter_jsonl(path):
        record = {
            "query": str(raw.get("query") or "").strip(),
            "query_type": str(raw.get("query_type") or "unknown"),
            "source_name": str(raw.get("source_name") or ""),
            "story_family": str(raw.get("story_family") or ""),
        }
        key = (record["source_name"], record["query"])
        if not all(key) or not record["story_family"]:
            raise ValueError(f"source contains empty query metadata: {record}")
        if key in seen:
            raise ValueError(f"source contains duplicate query key: {key}")
        seen.add(key)
        records.append(record)
    records = records[offset:]
    return records[:limit] if limit is not None else records


def candidate_text(document: dict[str, Any]) -> str:
    """Match the online cross-encoder's document payload."""

    return str(document.get("search_text") or document.get("clean_text") or "").strip()


def safe_record(
    metadata: dict[str, str], hits: list[dict[str, Any]], *, expected_candidates: int
) -> dict[str, Any]:
    candidates: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    for hit in hits:
        text = candidate_text(hit.get("document") or {})
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1:03d}",
                "text": text,
                "kind": "document",
            }
        )
        if len(candidates) >= expected_candidates:
            break
    if len(candidates) != expected_candidates:
        raise ValueError(
            f"query {metadata['query']!r} yielded {len(candidates)} unique candidates, "
            f"expected {expected_candidates}"
        )
    record: dict[str, Any] = {**metadata, "candidates": candidates}
    validate_safe_record(record, expected_candidates=expected_candidates)
    return record


def fused_hits_for_query(
    retriever: ArknightsHybridRetriever,
    query: str,
    *,
    dense_hits: list[dict[str, Any]],
    config: QueryConfig,
) -> list[dict[str, Any]]:
    sparse_hits = retriever.sparse_search(query, top_k=config.sparse_top_k)
    primary = retriever.reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=[],
        top_k=max(config.reranker_candidate_top_k, config.fusion_top_k),
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
        minirag_weight=0.0,
        dense_min_quota=config.dense_min_quota,
        sparse_min_quota=config.sparse_min_quota,
    )
    primary_texts = {
        candidate_text(item.get("document") or {}) for item in primary
    }
    if len(primary_texts) >= config.reranker_candidate_top_k:
        return primary

    # Exact duplicate chunks occasionally occupy two document IDs.  Preserve
    # the configured 120-document pool and source cutoffs, then append only the
    # next RRF-ranked source-union documents so safe_record can replace the
    # duplicate teacher-visible text.  This never adds another retrieval lane.
    source_union = retriever.reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=[],
        top_k=len(dense_hits) + len(sparse_hits),
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight,
        minirag_weight=0.0,
        dense_min_quota=0,
        sparse_min_quota=0,
    )
    seen_doc_indices = {int(item["doc_index"]) for item in primary}
    return primary + [
        item for item in source_union if int(item["doc_index"]) not in seen_doc_indices
    ]


def validate_safe_record(record: dict[str, Any], *, expected_candidates: int) -> None:
    if set(record) != SAFE_RECORD_FIELDS:
        raise ValueError(f"unsafe record fields: {sorted(set(record) - SAFE_RECORD_FIELDS)}")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != expected_candidates:
        raise ValueError(f"candidate count is not {expected_candidates}")
    ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != SAFE_CANDIDATE_FIELDS:
            raise ValueError("unsafe candidate fields")
        ids.append(str(candidate.get("candidate_id") or ""))
        if not str(candidate.get("text") or "").strip():
            raise ValueError("empty candidate text")
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError("empty or duplicate candidate IDs")


def build_retriever(args: argparse.Namespace) -> ArknightsHybridRetriever:
    return ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=None,
        documents_path=args.index_dir / "documents.jsonl",
        faiss_index_path=args.index_dir / "faiss.index",
        bm25_tokens_path=args.index_dir / "bm25_tokens.pkl",
        sparse_index_path=args.index_dir / "sparse_index.pkl",
        index_metadata_path=args.index_dir / "index_meta.json",
        device=args.device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--dense-top-k", type=int, default=120)
    parser.add_argument("--sparse-top-k", type=int, default=120)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=1.0)
    parser.add_argument("--dense-min-quota", type=int, default=20)
    parser.add_argument(
        "--sparse-rank-cutoff",
        type=int,
        default=100,
        help=(
            "Retain at least the sparse lane's first N ranks when filling the "
            "fixed pool. This is a source rank cutoff, not a disjoint quota."
        ),
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--dense-query-batch-size",
        type=int,
        default=32,
        help="Batch Qwen3 query encoding; sparse scoring remains per query.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.source = resolve(args.source)
    args.output = resolve(args.output)
    args.index_dir = resolve(args.index_dir)
    args.embedding_model = resolve(args.embedding_model)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    records = source_records(args.source, offset=args.offset, limit=args.limit)
    retriever = build_retriever(args)
    config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        minirag_top_k=0,
        fusion_top_k=args.expected_candidates,
        rerank_top_k=args.expected_candidates,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        dense_min_quota=args.dense_min_quota,
        sparse_min_quota=args.sparse_rank_cutoff,
        minirag_weight=0.0,
        reranker_candidate_top_k=args.expected_candidates,
        enable_neighbor_expansion=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as sink:
        batch_size = max(args.dense_query_batch_size, 1)
        written = 0
        for batch_start in range(0, len(records), batch_size):
            metadata_batch = records[batch_start : batch_start + batch_size]
            dense_batches = retriever.dense_search_many(
                [metadata["query"] for metadata in metadata_batch],
                top_k=config.dense_top_k,
            )
            for metadata, dense_hits in zip(metadata_batch, dense_batches):
                hits = fused_hits_for_query(
                    retriever,
                    metadata["query"],
                    dense_hits=dense_hits,
                    config=config,
                )
                record = safe_record(metadata, hits, expected_candidates=args.expected_candidates)
                encoded = (
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                sink.write(encoded.decode("utf-8"))
                digest.update(encoded)
                written += 1
                print(
                    f"[{written}/{len(records)}] candidates={len(record['candidates'])} "
                    f"query={record['query'][:50]}",
                    flush=True,
                )
            sink.flush()
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "records": len(records),
        "candidates_per_record": args.expected_candidates,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "retrieval": {
            "index_dir": str(args.index_dir),
            "embedding_model": str(args.embedding_model),
            "dense_top_k": args.dense_top_k,
            "sparse_top_k": args.sparse_top_k,
            "dense_weight": args.dense_weight,
            "sparse_weight": args.sparse_weight,
            "dense_min_quota": args.dense_min_quota,
            "sparse_rank_cutoff": args.sparse_rank_cutoff,
            "rrf_k": args.rrf_k,
            "dense_query_batch_size": args.dense_query_batch_size,
            "minirag_top_k": 0,
            "enable_neighbor_expansion": False,
            "duplicate_text_backfill": "next_rrf_rank_from_dense_sparse_source_union",
        },
        "teacher_visible_fields": [
            "query",
            "query_type",
            "source_name",
            "story_family",
            "candidates.*.candidate_id/text/kind",
        ],
        "contains_reference_answer": False,
        "contains_gold_or_retrieval_metadata": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
