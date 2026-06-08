from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.web_context.cache import (
    cache_key_for_web_context,
    read_web_context_cache,
    write_web_context_cache,
)
from asa_arknight_story_agent.inference.web_context.cache_items import web_context_items_from_cache
from asa_arknight_story_agent.inference.web_context.collection import (
    build_web_context_queries,
    collect_web_search_results,
    fetch_web_context_pages,
    make_web_page_candidates,
)
from asa_arknight_story_agent.inference.web_context.config import WebContextConfig
from asa_arknight_story_agent.inference.web_context.filtering import filter_web_context_candidates
from asa_arknight_story_agent.inference.web_context.pages import select_web_context_lines
from asa_arknight_story_agent.inference.web_context.rendering import build_web_context_text
from asa_arknight_story_agent.inference.web_context.terms import (
    dominant_story_name_from_evidence,
    story_name_candidate,
    web_context_question_terms,
)

if TYPE_CHECKING:
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever


def build_web_context_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    config: WebContextConfig,
    *,
    retriever: ArknightsHybridRetriever | None = None,
    hypothesis: HypothesisDocument | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config.enabled or not evidence:
        return [], {"enabled": config.enabled, "status": "disabled_or_no_evidence"}
    story_name, story_scores = dominant_story_name_from_evidence(
        evidence,
        max_items=config.max_first_round_evidence,
        min_hits=config.min_story_hits,
    )
    if not story_name:
        return [], {"enabled": True, "status": "no_dominant_story", "story_scores": story_scores}
    question_terms = web_context_question_terms(question, evidence, hypothesis=hypothesis)
    queries = build_web_context_queries(
        story_name=story_name,
        question=question,
        question_terms=question_terms,
        config=config,
    )
    cache_key = cache_key_for_web_context(story_name, queries)
    cached = read_web_context_cache(config.cache_dir, cache_key, config.cache_ttl_seconds)
    if cached and cached.get("text"):
        return web_context_items_from_cache(
            story_name=story_name,
            cached=cached,
            question=question,
            queries=queries,
            question_terms=question_terms,
            config=config,
            retriever=retriever,
            hypothesis=hypothesis,
        )

    deadline = time.time() + config.max_elapsed_seconds
    candidate_results, candidate_urls = collect_web_search_results(config, queries, deadline)
    pages, rejected_pages = fetch_web_context_pages(
        story_name=story_name,
        question=question,
        candidate_results=candidate_results,
        config=config,
        deadline=deadline,
    )

    if not pages:
        return [], {
            "enabled": True,
            "status": "no_pages",
            "story_name": story_name,
            "queries": queries,
            "question_terms": question_terms,
            "candidate_urls": candidate_urls[: config.max_search_results],
            "candidate_results": candidate_results[: config.max_search_results],
            "rejected_pages": rejected_pages[:8],
        }

    text = build_web_context_text(
        story_name=story_name,
        queries=queries,
        pages=pages,
        max_total_chars=config.max_total_chars,
    )
    urls = [page["url"] for page in pages]
    write_web_context_cache(config.cache_dir, cache_key, {"text": text, "urls": urls, "story_name": story_name})
    candidates = make_web_page_candidates(story_name, pages)
    filter_record: dict[str, Any] = {"status": "not_filtered_no_retriever"}
    if retriever is not None:
        candidates, filter_record = filter_web_context_candidates(
            retriever=retriever,
            question=question,
            story_name=story_name,
            candidates=candidates,
            config=config,
            hypothesis=hypothesis,
        )
    return candidates, {
        "enabled": True,
        "status": "fetched",
        "story_name": story_name,
        "queries": queries,
        "question_terms": question_terms,
        "urls": urls,
        "story_scores": story_scores,
        "filter": filter_record,
    }


__all__ = [
    "story_name_candidate",
    "dominant_story_name_from_evidence",
    "web_context_question_terms",
    "select_web_context_lines",
    "filter_web_context_candidates",
    "build_web_context_evidence",
]
