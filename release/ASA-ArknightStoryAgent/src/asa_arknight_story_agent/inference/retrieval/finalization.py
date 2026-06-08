from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.bundle_evidence import best_anchor_bundle_evidence
from asa_arknight_story_agent.inference.anchors.terms import extract_action_targets
from asa_arknight_story_agent.inference.evidence.texts import evidence_identity
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import (
    expand_related_retrieval_terms,
    extract_content_tokens,
    resolve_referential_question,
)
from asa_arknight_story_agent.inference.retrieval.reranking import classify_retrieval_query_mode, rerank_hits
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def build_final_rerank_query(question: str, hypothesis: HypothesisDocument) -> tuple[str, str, list[str]]:
    resolved_question = resolve_referential_question(question, hypothesis.entities)
    # Keep generation-time expansion out of the final reranker query. The
    # reranker was trained to compare evidence chains against the user
    # question; model-generated keywords are useful for candidate recall but can
    # drag final ranking toward noisy aliases or question fragments.
    rerank_query = resolved_question
    safe_related_terms = expand_related_retrieval_terms(
        extract_action_targets(resolved_question + "\n" + question)
        + extract_content_tokens(resolved_question)
        + hypothesis.entities[:4]
    )
    if safe_related_terms:
        rerank_query = rerank_query + "\n核心相关线索: " + " ".join(safe_related_terms[:10])
    return resolved_question, rerank_query, safe_related_terms


def fuse_retrieval_hits(
    retriever: Any,
    *,
    rerank_query: str,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    minirag_hits: list[dict[str, Any]],
    query_config: Any,
) -> list[dict[str, Any]]:
    minirag_weight = retriever.effective_minirag_weight(rerank_query, config=query_config)
    if query_config.minirag_fusion_mode == "append":
        primary_hits = retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=[],
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
            minirag_weight=0.0,
        )
        return retriever.append_supplemental_hits(
            primary_hits,
            minirag_hits if minirag_weight > 0 else [],
            top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
            source_name="minirag",
        )
    return retriever.reciprocal_rank_fusion(
        dense_hits=dense_hits,
        sparse_hits=sparse_hits,
        minirag_hits=minirag_hits if minirag_weight > 0 else [],
        top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
        rrf_k=query_config.rrf_k,
        dense_weight=query_config.dense_weight,
        sparse_weight=query_config.sparse_weight,
        minirag_weight=minirag_weight,
    )


def rerank_and_rescue_hits(
    retriever: Any,
    *,
    question: str,
    resolved_question: str,
    rerank_query: str,
    hypothesis: HypothesisDocument,
    fused_hits: list[dict[str, Any]],
    safe_related_terms: list[str],
    query_config: Any,
) -> list[dict[str, Any]]:
    reranked_hits = rerank_hits(
        retriever,
        rerank_query,
        fused_hits,
        top_k=query_config.rerank_top_k,
        batch_size=query_config.rerank_batch_size,
        query_mode=classify_retrieval_query_mode(hypothesis),
    )
    rescue_core_terms = dedupe_keep_order(
        extract_action_targets(resolved_question + "\n" + question)
        + extract_content_tokens(resolved_question)
    )[:6]
    # Rescue candidates should favor the closest deterministic expansion terms;
    # broader related terms are useful for recall, but can otherwise crowd out
    # the direct bridge evidence.
    rescue_bundle_terms = dedupe_keep_order([*rescue_core_terms, *safe_related_terms[:3]])
    rescue_hits = best_anchor_bundle_evidence(
        fused_hits,
        core_terms=rescue_core_terms,
        bundle_terms=rescue_bundle_terms,
        limit=max(1, min(4, query_config.rerank_top_k // 4 or 1)),
    )
    if not rescue_hits:
        return reranked_hits

    merged_hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*rescue_hits, *reranked_hits]:
        identity = evidence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged_hits.append(item)
        if len(merged_hits) >= query_config.rerank_top_k:
            break
    return merged_hits
