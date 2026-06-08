from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.web_context.config import WebContextConfig
from asa_arknight_story_agent.inference.web_context.filtering import filter_web_context_candidates
from asa_arknight_story_agent.inference.web_context.rendering import make_web_context_evidence_item

if TYPE_CHECKING:
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever


def web_context_items_from_cache(
    *,
    story_name: str,
    cached: dict[str, Any],
    question: str,
    queries: list[str],
    question_terms: list[str],
    config: WebContextConfig,
    retriever: ArknightsHybridRetriever | None,
    hypothesis: HypothesisDocument | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    urls = [str(url) for url in cached.get("urls") or []]
    cached_items = [make_web_context_evidence_item(story_name, str(cached["text"]), urls)]
    filter_record: dict[str, Any] = {"status": "not_filtered_no_retriever"}
    if retriever is not None:
        cached_items, filter_record = filter_web_context_candidates(
            retriever=retriever,
            question=question,
            story_name=story_name,
            candidates=cached_items,
            config=config,
            hypothesis=hypothesis,
        )
    return cached_items, {
        "enabled": True,
        "status": "cache_hit",
        "story_name": story_name,
        "queries": queries,
        "question_terms": question_terms,
        "urls": urls,
        "filter": filter_record,
    }
