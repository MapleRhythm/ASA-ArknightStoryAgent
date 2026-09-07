#!/usr/bin/env python3
"""Build auditable first-round retrieval snapshots for offline evaluation.

This script is read-only with respect to indexes and runtime configuration.  It
does not call the answer generator and does not change production defaults.
Each input row must contain a question and explicit gold document IDs:

{
  "id": "q1",
  "question": "...",
  "question_family": "family-a",
  "gold_evidence_ids": ["activities/...#chunk-0001"],
  "hypothesis": { ... optional production hypothesis fields ... },
  "queries": ["... optional exact retrieval queries ..."]
}

The JSONL output can be consumed by ``scripts/evaluate_retrieval_trace.py`` and
``scripts/evaluate_prompt_selection_ablation.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    MINIRAG_GRAPH_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)
from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline  # noqa: E402
from asa_arknight_story_agent.inference.evidence.prompt_preparation import (  # noqa: E402
    prepare_prompt_evidence,
)
from asa_arknight_story_agent.inference.pipeline.query_flow import build_initial_queries  # noqa: E402
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument  # noqa: E402
from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis  # noqa: E402
from asa_arknight_story_agent.inference.retrieval.finalization import (  # noqa: E402
    build_final_rerank_query,
    fuse_retrieval_hits,
    rerank_and_rescue_hits,
)
from asa_arknight_story_agent.inference.retrieval.minirag_scope_context import (  # noqa: E402
    build_minirag_scope_context,
    minirag_scope_values,
)
from asa_arknight_story_agent.inference.retrieval.minirag_second_pass import (  # noqa: E402
    run_minirag_second_pass,
)
from asa_arknight_story_agent.inference.retrieval.neighbors import (  # noqa: E402
    expand_fused_hits_with_neighbors,
)
from asa_arknight_story_agent.inference.retrieval.search import (  # noqa: E402
    search_queries,
)
from asa_arknight_story_agent.inference.common.text_utils import (  # noqa: E402
    expand_queries_with_main_chapter_terms,
)
from asa_arknight_story_agent.runtime_config import (  # noqa: E402
    load_runtime_config,
    resolve_path_value,
)


SNAPSHOT_VERSION = "asa_retrieval_stage_snapshot_v2"
HYPOTHESIS_FIELDS = (
    "question",
    "intent",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)


class SnapshotBuildError(ValueError):
    """Raised when the manifest cannot produce a reproducible snapshot."""


class _NoGenerationRunner:
    """Placeholder satisfying the retrieval/prompt pipeline constructor."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            for key in ("rows", "records", "items"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
            else:
                values = [value]
        else:
            raise SnapshotBuildError(f"expected JSON object/array: {path}")
    if not all(isinstance(row, dict) for row in values):
        raise SnapshotBuildError("all manifest rows must be JSON objects")
    return [row for row in values if isinstance(row, dict)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_config_from_runtime(retrieval_cfg: dict[str, Any]) -> QueryConfig:
    values = {
        field.name: retrieval_cfg[field.name]
        for field in fields(QueryConfig)
        if field.name in retrieval_cfg
    }
    return QueryConfig(**values)


def hypothesis_from_row(row: dict[str, Any]) -> HypothesisDocument:
    question = str(row.get("question") or "").strip()
    if not question:
        raise SnapshotBuildError("missing question")
    value = row.get("hypothesis")
    if value is None:
        return build_hypothesis(question, str(row.get("dialogue_context") or ""))
    if not isinstance(value, dict):
        raise SnapshotBuildError("hypothesis must be an object")
    unknown = set(value) - set(HYPOTHESIS_FIELDS)
    if unknown:
        raise SnapshotBuildError("unknown hypothesis fields: " + ",".join(sorted(unknown)))
    return HypothesisDocument(
        question=str(value.get("question") or question),
        intent=str(value.get("intent") or "plot_reasoning"),
        query_type=str(value.get("query_type") or "reasoning"),
        entities=[str(item) for item in value.get("entities") or []],
        keywords=[str(item) for item in value.get("keywords") or []],
        expected_answer_type=str(value.get("expected_answer_type") or ""),
        dialogue_context=str(value.get("dialogue_context") or ""),
    )


def queries_from_row(
    row: dict[str, Any], question: str, hypothesis: HypothesisDocument
) -> list[str]:
    value = row.get("queries")
    if value is None:
        return build_initial_queries(question, hypothesis)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SnapshotBuildError("queries must be a list of strings")
    queries = [item.strip() for item in value if item.strip()]
    if not queries:
        raise SnapshotBuildError("queries must not be empty")
    return queries


def gold_ids_from_row(row: dict[str, Any]) -> list[str]:
    value = row.get("gold_evidence_ids")
    if not isinstance(value, list):
        raise SnapshotBuildError("missing gold_evidence_ids list")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        item_id = str(item).strip()
        if item_id and item_id not in seen:
            seen.add(item_id)
            result.append(item_id)
    if not result:
        raise SnapshotBuildError("gold_evidence_ids must not be empty")
    return result


def gold_supported_facts_from_row(
    row: dict[str, Any], gold_ids: list[str]
) -> list[dict[str, Any]]:
    value = row.get("gold_supported_facts")
    if value is None:
        return []
    if not isinstance(value, list):
        raise SnapshotBuildError("gold_supported_facts must be a list")
    gold_set = set(gold_ids)
    result: list[dict[str, Any]] = []
    for fact in value:
        if not isinstance(fact, dict):
            raise SnapshotBuildError(
                "gold_supported_facts items must be objects"
            )
        fact_text = str(fact.get("fact") or "").strip()
        evidence_ids = fact.get("gold_evidence_ids")
        if not fact_text:
            raise SnapshotBuildError("gold supported fact missing fact text")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise SnapshotBuildError(
                "gold supported fact missing gold_evidence_ids"
            )
        normalized_ids = list(
            dict.fromkeys(
                str(item).strip()
                for item in evidence_ids
                if str(item).strip()
            )
        )
        if not normalized_ids:
            raise SnapshotBuildError(
                "gold supported fact has no valid evidence IDs"
            )
        unknown = sorted(set(normalized_ids) - gold_set)
        if unknown:
            raise SnapshotBuildError(
                "gold supported fact references unknown evidence IDs: "
                + ",".join(unknown)
            )
        result.append(
            {
                "fact": fact_text,
                "gold_evidence_ids": normalized_ids,
            }
        )
    return result


def hit_document_id(item: dict[str, Any]) -> str:
    document = item.get("document") or {}
    return str(document.get("id") or item.get("doc_index") or "").strip()


def simplify_hit(item: dict[str, Any]) -> dict[str, Any]:
    document = item.get("document") or {}
    return {
        "id": hit_document_id(item),
        "doc_index": item.get("doc_index"),
        "activity_name": document.get("activity_name"),
        "story_name": document.get("story_name"),
        "stage_code": document.get("stage_code"),
        "source_path": document.get("source_path"),
        "clean_text": document.get("clean_text"),
        "search_text": document.get("search_text"),
        "evidence_chain_text": item.get("evidence_chain_text"),
        "score": item.get("score"),
        "dense_score": item.get("dense_score"),
        "sparse_score": item.get("sparse_score"),
        "minirag_score": item.get("minirag_score"),
        "fusion_score": item.get("fusion_score"),
        "rerank_score": item.get("rerank_score"),
        "evidence_chain_score": item.get("evidence_chain_score"),
        "evidence_chain_model_score": item.get("evidence_chain_model_score"),
        "evidence_chain_roles": item.get("evidence_chain_roles"),
        "supplemental_source": item.get("supplemental_source"),
    }


def _finalize_stage(
    *,
    pipeline: CPUInferencePipeline,
    question: str,
    hypothesis: HypothesisDocument,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    minirag_hits: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_question, rerank_query, safe_related_terms = build_final_rerank_query(
        question, hypothesis
    )
    fused_hits = fuse_retrieval_hits(
        pipeline.retriever,
        rerank_query=rerank_query,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=minirag_hits,
        query_config=pipeline.query_config,
    )
    if pipeline.query_config.enable_neighbor_expansion:
        fused_hits = expand_fused_hits_with_neighbors(
            pipeline=pipeline, fused_hits=fused_hits
        )
    reranked_hits = rerank_and_rescue_hits(
        pipeline.retriever,
        question=question,
        resolved_question=resolved_question,
        rerank_query=rerank_query,
        hypothesis=hypothesis,
        fused_hits=fused_hits,
        safe_related_terms=safe_related_terms,
        query_config=pipeline.query_config,
    )
    return fused_hits, reranked_hits


def _first_round_stages(
    *,
    pipeline: CPUInferencePipeline,
    question: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    query_config = pipeline.query_config
    expansion: dict[str, Any] = {"enabled": False}
    enable_minirag = bool(getattr(pipeline, "snapshot_enable_minirag", False))
    if not (
        enable_minirag
        and
        query_config.minirag_chapter_isolation
        and query_config.minirag_auto_second_retrieval
    ):
        dense_hits, sparse_hits, minirag_hits = search_queries(
            pipeline=pipeline,
            queries=queries,
            enable_minirag=enable_minirag,
        )
        fused_hits, reranked_hits = _finalize_stage(
            pipeline=pipeline,
            question=question,
            hypothesis=hypothesis,
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=minirag_hits,
        )
        return (
            dense_hits,
            sparse_hits,
            minirag_hits,
            fused_hits,
            reranked_hits,
            expansion,
        )

    expanded_queries = expand_queries_with_main_chapter_terms(queries)
    dense_hits, sparse_hits, _ = search_queries(
        pipeline=pipeline,
        queries=expanded_queries,
        enable_minirag=False,
    )
    first_pass_fused, first_pass_evidence = _finalize_stage(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=[],
    )
    scope_context = build_minirag_scope_context(
        pipeline=pipeline,
        first_pass_evidence=first_pass_evidence,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
    )
    if not scope_context.scope_info:
        return (
            dense_hits,
            sparse_hits,
            [],
            first_pass_fused,
            first_pass_evidence,
            expansion,
        )

    (
        chapter_scope,
        scope_ratio,
        graph_scope_enabled,
        second_pass_scope_enabled,
        graph_scope,
    ) = minirag_scope_values(pipeline, scope_context.scope_info)
    graph_hits = pipeline._search_minirag_queries(
        expanded_queries,
        minirag_chapter_scope=graph_scope,
    )
    if not graph_hits:
        expansion = {
            "enabled": True,
            "chapter_scope": chapter_scope,
            "scope_ratio": scope_ratio,
            "graph_scope_enabled": graph_scope_enabled,
            "second_pass_scope_enabled": second_pass_scope_enabled,
            "graph_hit_count": 0,
        }
        return (
            dense_hits,
            sparse_hits,
            [],
            first_pass_fused,
            first_pass_evidence,
            expansion,
        )

    second_pass = run_minirag_second_pass(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        expanded_queries=expanded_queries,
        graph_hits=graph_hits,
        chapter_scope=chapter_scope,
        graph_scope=graph_scope,
        graph_scope_enabled=graph_scope_enabled,
        second_pass_scope_enabled=second_pass_scope_enabled,
        sparse_storyline_scope=scope_context.sparse_storyline_scope,
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        scope_info=scope_context.scope_info,
    )
    final_dense = second_pass.combined_dense_hits
    final_sparse = second_pass.combined_sparse_hits
    final_minirag = second_pass.combined_minirag_hits
    fused_hits, reranked_hits = _finalize_stage(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        dense_hits=final_dense,
        sparse_hits=final_sparse,
        minirag_hits=final_minirag,
    )
    expansion = {
        "enabled": True,
        "chapter_scope": chapter_scope,
        "scope_ratio": scope_ratio,
        "graph_scope_enabled": graph_scope_enabled,
        "second_pass_scope_enabled": second_pass_scope_enabled,
        "graph_hit_count": len(graph_hits),
        "second_pass_queries": second_pass.second_pass_queries,
        "use_scoped_candidates": second_pass.use_scoped_candidates,
    }
    return (
        final_dense,
        final_sparse,
        final_minirag,
        fused_hits,
        reranked_hits,
        expansion,
    )


def build_snapshot_row(
    *,
    pipeline: CPUInferencePipeline,
    row: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    hypothesis = hypothesis_from_row(row)
    queries = queries_from_row(row, question, hypothesis)
    gold_ids = gold_ids_from_row(row)
    gold_supported_facts = gold_supported_facts_from_row(row, gold_ids)
    (
        dense_hits,
        sparse_hits,
        minirag_hits,
        fused_hits,
        reranked_hits,
        expansion,
    ) = _first_round_stages(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        queries=queries,
    )
    prompt_hits = prepare_prompt_evidence(
        pipeline=pipeline,
        question=question,
        hypothesis=hypothesis,
        evidence=reranked_hits,
    )
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "id": str(row.get("id") or row.get("task_id") or row_index),
        "question": question,
        "question_family": str(
            row.get("question_family")
            or row.get("family")
            or row.get("family_id")
            or question
        ),
        "query_type": hypothesis.query_type,
        "gold_evidence_ids": gold_ids,
        "gold_supported_facts": gold_supported_facts,
        "hypothesis": {
            field: getattr(hypothesis, field) for field in HYPOTHESIS_FIELDS
        },
        "queries": queries,
        "stages": {
            "dense": [simplify_hit(item) for item in dense_hits],
            "sparse": [simplify_hit(item) for item in sparse_hits],
            "minirag": [simplify_hit(item) for item in minirag_hits],
            "fusion": [simplify_hit(item) for item in fused_hits],
            "rerank": [simplify_hit(item) for item in reranked_hits],
            "prompt": [simplify_hit(item) for item in prompt_hits],
        },
        "rounds": [
            {
                "round": 1,
                "query": "\n\n".join(queries),
                "doc_ids": [hit_document_id(item) for item in reranked_hits],
            }
        ],
        "minirag_expansion": expansion,
    }


def build_pipeline(
    *,
    retriever: Any,
    query_config: QueryConfig,
    inference_cfg: dict[str, Any],
    enable_minirag: bool = False,
) -> CPUInferencePipeline:
    web_context = inference_cfg.get("web_context")
    if isinstance(web_context, dict) and web_context.get("enabled"):
        raise SnapshotBuildError(
            "web context must be disabled for deterministic retrieval snapshots"
        )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=_NoGenerationRunner(),
        query_config=query_config,
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 10)),
        enable_mmr=bool(inference_cfg.get("enable_mmr", False)),
        mmr_lambda=float(inference_cfg.get("mmr_lambda", 0.72)),
        enable_set_coverage_selection=bool(
            inference_cfg.get("enable_set_coverage_selection", False)
        ),
        enable_pyramid_order=bool(inference_cfg.get("enable_pyramid_order", False)),
        enable_evidence_pinning=bool(
            inference_cfg.get("enable_evidence_pinning", False)
        ),
        enable_crag_refinement=bool(
            inference_cfg.get("enable_crag_refinement", False)
        ),
        crag_refine_top_sentences=int(
            inference_cfg.get("crag_refine_top_sentences", 4)
        ),
        crag_refine_max_sentences=int(
            inference_cfg.get("crag_refine_max_sentences", 24)
        ),
        use_model_hypothesis=True,
        use_model_conclusion_generation=True,
        web_context_config={"enabled": False},
    )
    pipeline.snapshot_enable_minirag = bool(enable_minirag)
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-model", type=Path, default=None)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--faiss-index", type=Path, default=FAISS_INDEX_PATH)
    parser.add_argument("--bm25-tokens", type=Path, default=BM25_TOKENS_PATH)
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace it: {args.output}")
    runtime = load_runtime_config(args.runtime_config)
    retrieval_cfg = (
        runtime.get("retrieval", {})
        if isinstance(runtime.get("retrieval"), dict)
        else {}
    )
    inference_cfg = (
        runtime.get("inference", {})
        if isinstance(runtime.get("inference"), dict)
        else {}
    )
    query_config = query_config_from_runtime(retrieval_cfg)
    device = str(args.device or retrieval_cfg.get("device") or "cpu")
    embedding_model = resolve_path_value(
        args.embedding_model,
        retrieval_cfg,
        "embedding_model_path",
        EMBEDDING_MODEL_DIR,
    )
    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True))
    reranker_model = (
        resolve_path_value(
            args.reranker_model,
            retrieval_cfg,
            "reranker_model_path",
            RERANKER_MODEL_DIR,
        )
        if enable_reranker
        else None
    )
    enable_minirag = bool(retrieval_cfg.get("enable_minirag", False))
    minirag_index = (
        resolve_path_value(
            args.minirag_index,
            retrieval_cfg,
            "minirag_index_path",
            MINIRAG_GRAPH_PATH,
        )
        if enable_minirag
        else None
    )
    required_paths = [
        ("documents", args.documents),
        ("faiss index", args.faiss_index),
        ("BM25 tokens", args.bm25_tokens),
        ("embedding model", embedding_model),
    ]
    if reranker_model is not None:
        required_paths.append(("reranker model", reranker_model))
    if minirag_index is not None:
        required_paths.append(("MiniRAG index", minirag_index))
    missing = [f"{label}: {path}" for label, path in required_paths if path is None or not path.exists()]
    if missing:
        raise SystemExit("missing snapshot assets:\n" + "\n".join(missing))

    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=embedding_model or EMBEDDING_MODEL_DIR,
        reranker_model_path=reranker_model,
        reranker_max_length=int(retrieval_cfg.get("reranker_max_length", 1024)),
        documents_path=args.documents,
        faiss_index_path=args.faiss_index,
        bm25_tokens_path=args.bm25_tokens,
        minirag_index_path=minirag_index,
        device=device,
    )
    pipeline = build_pipeline(
        retriever=retriever,
        query_config=query_config,
        inference_cfg=inference_cfg,
        enable_minirag=enable_minirag,
    )
    rows = read_rows(args.manifest)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise SystemExit("manifest produced no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            try:
                snapshot = build_snapshot_row(
                    pipeline=pipeline,
                    row=row,
                    row_index=index,
                )
            except Exception as exc:
                failures.append(
                    {
                        "id": str(row.get("id") or row.get("task_id") or index),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
    meta = {
        "snapshot_version": SNAPSHOT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_sha256": sha256_file(args.runtime_config),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows_requested": len(rows),
        "rows_written": written,
        "rows_failed": len(failures),
        "failure_examples": failures[:20],
        "device": device,
        "assets": {
            "documents": str(args.documents.resolve()),
            "faiss_index": str(args.faiss_index.resolve()),
            "bm25_tokens": str(args.bm25_tokens.resolve()),
            "embedding_model": str((embedding_model or EMBEDDING_MODEL_DIR).resolve()),
            "reranker_model": str(reranker_model.resolve()) if reranker_model else None,
            "minirag_index": str(minirag_index.resolve()) if minirag_index else None,
            "minirag_enabled": enable_minirag,
        },
        "query_config": {
            field.name: getattr(query_config, field.name) for field in fields(QueryConfig)
        },
        "prompt_selection": {
            "top_k": pipeline.prompt_evidence_top_k,
            "enable_mmr": pipeline.enable_mmr,
            "mmr_lambda": pipeline.mmr_lambda,
            "enable_set_coverage_selection": pipeline.enable_set_coverage_selection,
            "enable_pyramid_order": pipeline.enable_pyramid_order,
            "enable_evidence_pinning": pipeline.enable_evidence_pinning,
            "enable_crag_refinement": pipeline.enable_crag_refinement,
        },
    }
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
