from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.common.lexicon import (
    BRIDGE_STOP_WORDS,
    RELATION_TERMS,
    TITLE_TERMS,
)
from asa_arknight_story_agent.inference.common.patterns import INHERITANCE_RE, KINSHIP_RE, QUESTION_TOKEN_RE
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import extract_entities


def extract_bridge_terms(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> list[str]:
    counts: dict[str, int] = {}
    known_terms = set(hypothesis.entities) | set(hypothesis.keywords) | set(extract_entities(question))

    for item in evidence[:6]:
        text = item["document"]["clean_text"]

        for title in TITLE_TERMS:
            if title in text:
                counts[title] = counts.get(title, 0) + 3

        for match in INHERITANCE_RE.finditer(text):
            phrase = match.group(1)
            if phrase not in known_terms:
                counts[phrase] = counts.get(phrase) + 3 if phrase in counts else 3

        for match in KINSHIP_RE.finditer(text):
            phrase = match.group(1)
            counts[phrase] = counts.get(phrase, 0) + 2

        for token in QUESTION_TOKEN_RE.findall(text):
            normalized = token.strip()
            if (
                not normalized
                or normalized in known_terms
                or normalized in BRIDGE_STOP_WORDS
                or (len(normalized) == 1 and not normalized.isascii())
            ):
                continue
            score = 1
            if normalized in TITLE_TERMS:
                score += 2
            if normalized in RELATION_TERMS:
                score += 2
            counts[normalized] = counts.get(normalized, 0) + score

    filtered_counts = {
        term: score
        for term, score in counts.items()
        if term in TITLE_TERMS or term in RELATION_TERMS or score >= 2
    }

    ranked = sorted(
        filtered_counts.items(),
        key=lambda item: (
            item[0] not in TITLE_TERMS,
            item[0] not in RELATION_TERMS,
            -item[1],
            len(item[0]),
        ),
    )
    return [term for term, _ in ranked[:6]]
