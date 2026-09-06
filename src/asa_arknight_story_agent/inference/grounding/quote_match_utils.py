from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.common.lexicon import COMMON_NON_ENTITY_WORDS
from asa_arknight_story_agent.inference.pipeline.constants import NOISY_RETRIEVAL_TOKENS, PRONOUN_REFERENCES
from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens, is_entity_candidate
from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta


GROUNDING_LONG_TOKEN_MIN_LEN = 3
GROUNDING_EVIDENCE_POOL_TOP_K = 12


def grounding_extract_answer_tokens(answer: str, question: str) -> list[str]:
    answer_tokens = [
        token
        for token in extract_content_tokens(answer)
        if is_entity_candidate(token)
        and token not in COMMON_NON_ENTITY_WORDS
        and token not in NOISY_RETRIEVAL_TOKENS
        and token not in PRONOUN_REFERENCES
    ]
    question_tokens = set(extract_content_tokens(question))
    return [token for token in answer_tokens if token not in question_tokens]


def grounding_evidence_pool(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence[:GROUNDING_EVIDENCE_POOL_TOP_K]:
        document = item.get("document") or {}
        for value in (
            item.get("evidence_chain_text"),
            document.get("clean_text"),
            document.get("search_text"),
            document.get("activity_name"),
            document.get("story_name"),
            document.get("stage_code"),
        ):
            text = strip_internal_evidence_meta(str(value or "")).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def normalize_for_evidence_match(text: str) -> str:
    cleaned = strip_internal_evidence_meta(str(text or ""))
    # Evidence-chain renders prefix members with [E1]/[E2]... markers; models
    # often copy the marker along with the quote, so strip it from both sides
    # of the comparison. Truncation ellipses at render boundaries are likewise
    # not part of the underlying text.
    cleaned = re.sub(r"\[E\d+\]", "", cleaned)
    cleaned = cleaned.replace("...", "").replace("…", "")
    return re.sub(r"\s+", "", cleaned)


def quote_matches_evidence(quote: str, evidence_text: str) -> bool:
    """Return whether a model quote is supported by one evidence item.

    Models frequently abbreviate a copied quote with ``...``/``…``.  Removing
    the ellipsis before substring matching is incorrect because the omitted
    characters still exist in the source text (``A...B`` would become ``AB``).
    Treat ellipses as ordered wildcards instead, while retaining exact
    substring matching for ordinary quotes.
    """

    raw_quote = strip_internal_evidence_meta(str(quote or ""))
    raw_quote = re.sub(r"\[E\d+\]", "", raw_quote)
    target = normalize_for_evidence_match(evidence_text)
    if not target:
        return False
    if not re.search(r"(?:\.{3}|…)", raw_quote):
        return normalize_for_evidence_match(raw_quote) in target

    parts = [
        normalize_for_evidence_match(part)
        for part in re.split(r"(?:\.{3}|…)+", raw_quote)
    ]
    parts = [part for part in parts if part]
    if len(parts) < 2:
        return False
    cursor = 0
    for part in parts:
        position = target.find(part, cursor)
        if position < 0:
            return False
        cursor = position + len(part)
    return True
