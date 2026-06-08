from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_text
from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    WEB_CONTEXT_EXCLUDED_ACTIVITY_NAMES,
    WEB_CONTEXT_GENERIC_QUERY_TERMS,
    WEB_CONTEXT_QUERY_ANCHOR_TERMS,
)
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens, extract_entities
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def story_name_candidate(document: dict[str, Any]) -> str:
    for key in ("activity_name", "story_name"):
        value = re.sub(r"\s+", " ", str(document.get(key) or "")).strip()
        if not value or value in WEB_CONTEXT_EXCLUDED_ACTIVITY_NAMES:
            continue
        if value.startswith("档案资料") or value in {"晋升记录", "模组故事", "语音记录"}:
            continue
        return value
    return ""


def dominant_story_name_from_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    min_hits: int,
) -> tuple[str, dict[str, int]]:
    scores: dict[str, int] = {}
    for rank, item in enumerate(evidence[:max_items], start=1):
        document = item.get("document") or {}
        story_name = story_name_candidate(document)
        if not story_name:
            continue
        weight = max(1, max_items - rank + 1)
        scores[story_name] = scores.get(story_name, 0) + weight
    if not scores:
        return "", {}
    winner, score = max(scores.items(), key=lambda pair: (pair[1], len(pair[0])))
    if score < min_hits:
        return "", scores
    return winner, scores


def web_context_question_terms(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    hypothesis: HypothesisDocument | None = None,
    limit: int = 12,
) -> list[str]:
    terms: list[str] = []
    terms.extend(extract_entities(question))
    terms.extend(extract_content_tokens(question))

    seed_text = question + "\n" + "\n".join(evidence_text(item)[:1200] for item in evidence[:8])
    if "岁陵" in question and "危机" in question:
        terms.extend(["岁陵", "危机", "岁陵危机", "岁兽之患", "岁兽", "苏醒", "平息"])
    for anchor in sorted(WEB_CONTEXT_QUERY_ANCHOR_TERMS, key=len, reverse=True):
        if anchor and anchor in seed_text:
            terms.append(anchor)
    if "危机" in question:
        terms.append("危机")
    if hypothesis is not None:
        terms.extend(hypothesis.entities)
        terms.extend(
            term
            for term in hypothesis.keywords[:16]
            if term not in WEB_CONTEXT_GENERIC_QUERY_TERMS
        )
    for item in evidence[:8]:
        text = evidence_text(item)
        if "不反" in text or "不反" in question:
            terms.extend(["不反", "岁陵", "真龙"])
        for term in WEB_CONTEXT_QUERY_ANCHOR_TERMS:
            if term in question or term in text:
                terms.append(term)
    return dedupe_keep_order(
        [
            term
            for term in terms
            if term
            and term not in COMMON_NON_ENTITY_WORDS
            and term not in WEB_CONTEXT_GENERIC_QUERY_TERMS
            and (term in WEB_CONTEXT_QUERY_ANCHOR_TERMS or term not in NOISY_RETRIEVAL_TOKENS)
            and len(term) <= 12
        ]
    )[:limit]
