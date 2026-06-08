from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
from asa_arknight_story_agent.inference.retrieval.reranking import classify_retrieval_query_mode, rerank_hits
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.web_context.config import WebContextConfig
from asa_arknight_story_agent.inference.web_context.terms import web_context_question_terms

if TYPE_CHECKING:
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever


def web_context_has_scope_hit(item: dict[str, Any], *, story_name: str, question: str) -> bool:
    doc = item.get("document") or {}
    text = re.sub(r"\s+", "", str(doc.get("clean_text") or doc.get("search_text") or ""))
    if story_name and story_name in text:
        return True
    question_terms = extract_content_tokens(question)
    anchor_terms = [term for term in [*web_context_question_terms(question, [], hypothesis=None), *question_terms] if len(term) >= 2]
    return any(term in text for term in dedupe_keep_order(anchor_terms)[:12])


def filter_web_context_candidates(
    *,
    retriever: ArknightsHybridRetriever,
    question: str,
    story_name: str,
    candidates: list[dict[str, Any]],
    config: WebContextConfig,
    hypothesis: HypothesisDocument | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates or config.rerank_top_k <= 0:
        return [], {"status": "disabled_or_no_candidates", "candidate_count": len(candidates)}
    scoped_candidates = []
    rejected: list[dict[str, Any]] = []
    for item in candidates:
        if config.require_story_or_question_hit and not web_context_has_scope_hit(
            item,
            story_name=story_name,
            question=question,
        ):
            doc = item.get("document") or {}
            rejected.append(
                {
                    "id": doc.get("id"),
                    "title": str(doc.get("clean_text") or "").splitlines()[0][:120],
                    "reason": "missing_story_or_question_hit",
                }
            )
            continue
        scoped_candidates.append(item)
    if not scoped_candidates:
        return [], {
            "status": "all_candidates_rejected_by_scope",
            "candidate_count": len(candidates),
            "rejected": rejected[:8],
        }
    rerank_query = question
    if hypothesis is not None:
        terms = dedupe_keep_order(hypothesis.entities + hypothesis.keywords + extract_content_tokens(question))
        if terms:
            rerank_query = rerank_query + "\n联网资料相关线索: " + " ".join(terms[:12])
    reranked = rerank_hits(
        retriever,
        rerank_query,
        scoped_candidates,
        top_k=min(config.rerank_top_k, len(scoped_candidates)),
        batch_size=4,
        query_mode=classify_retrieval_query_mode(hypothesis) if hypothesis else None,
    )
    accepted = [
        item
        for item in reranked
        if float(item.get("rerank_score") or 0.0) >= config.rerank_min_score
    ]
    return accepted, {
        "status": "filtered",
        "candidate_count": len(candidates),
        "scoped_candidate_count": len(scoped_candidates),
        "accepted_count": len(accepted),
        "rerank_min_score": config.rerank_min_score,
        "top_scores": [
            {
                "id": (item.get("document") or {}).get("id"),
                "score": item.get("rerank_score"),
                "title": str((item.get("document") or {}).get("clean_text") or "").splitlines()[0][:120],
            }
            for item in reranked[:5]
        ],
        "rejected": rejected[:8],
    }
