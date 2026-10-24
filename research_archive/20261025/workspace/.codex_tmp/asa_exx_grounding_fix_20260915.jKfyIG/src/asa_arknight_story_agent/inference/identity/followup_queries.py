from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.identity.bridge_terms import extract_bridge_terms
from asa_arknight_story_agent.inference.common.lexicon import IDENTITY_HINT_WORDS, RELATION_TERMS, TITLE_TERMS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def build_follow_up_queries(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if not any(token in question for token in IDENTITY_HINT_WORDS):
        return [], []

    bridge_terms = extract_bridge_terms(question, hypothesis, evidence)
    anchor = hypothesis.entities[0] if hypothesis.entities else ""

    queries: list[str] = []
    if anchor:
        queries.extend(
            [
                f"{anchor} 身世 真相",
                f"{anchor} 身份 来历",
            ]
        )

    for term in bridge_terms:
        if anchor:
            queries.append(f"{anchor} {term}")
        if term in TITLE_TERMS:
            queries.append(f"{term} 是谁")
            if anchor:
                queries.append(f"{anchor} {term} 什么关系")
        if term in RELATION_TERMS and anchor:
            queries.append(f"{anchor} {term} 是谁")

    if anchor and any("真相" in item["document"]["clean_text"] for item in evidence[:4]):
        queries.append(f"{anchor} 身世 全部真相")

    deduped_queries = []
    seen: set[str] = {question.strip()}
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:6], bridge_terms
