from __future__ import annotations

import re
from collections import defaultdict
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


def ordered_fuzzy_term_match(
    term: str,
    evidence_text: str,
    *,
    max_intervening_chars: int | None = None,
) -> bool:
    """Match a short Chinese term when evidence inserts modifiers.

    ``extract_content_tokens`` intentionally emits compact retrieval terms.
    For grounding, however, a fixed-width token can split a phrase at an
    arbitrary boundary (for example ``强化防卫体系`` becoming
    ``化防卫体系``) or the evidence can insert a normal qualifier
    (``强化罗德岛防卫体系``).  Exact substring matching therefore creates
    false rejections.  This helper keeps the safety direction conservative:

    * ASCII terms still require an exact normalized substring;
    * Chinese terms must preserve character order;
    * only a short bounded gap is allowed between adjacent characters.

    It is deliberately not a general semantic matcher.  Relation/entity
    terms listed in ``claim_has_unsupported_quote_required_terms`` remain
    strict and can still reject unsupported identity or causal claims.
    """

    normalized_term = normalize_for_evidence_match(term)
    normalized_evidence = normalize_for_evidence_match(evidence_text)
    if not normalized_term or not normalized_evidence:
        return False
    if normalized_term in normalized_evidence:
        return True
    if normalized_term.isascii():
        return False
    # Two-character fragments are too ambiguous to fuzzy-match safely.
    if len(normalized_term) < 3:
        return False
    if max_intervening_chars is None:
        # Allow a small amount of inserted context without turning this into
        # an unrestricted subsequence test.
        max_intervening_chars = max(2, min(8, len(normalized_term)))

    def ordered_subsequence(candidate: str) -> bool:
        # Keep all viable positions instead of greedily taking the first one:
        # a common source pattern is ``凯尔希医生……考虑……强化`` where an earlier
        # unrelated ``考`` would otherwise make the greedy path fail.
        positions: dict[str, list[int]] = defaultdict(list)
        for index, char in enumerate(normalized_evidence):
            positions[char].append(index)
        states = {position: position for position in positions.get(candidate[0], [])}
        if not states:
            return False
        for char in candidate[1:]:
            next_states: dict[int, int] = {}
            for position in positions.get(char, []):
                # The evidence is short enough that scanning the current
                # frontier is cheaper and clearer than introducing an index.
                starts = [
                    start
                    for previous, start in states.items()
                    if 0 <= position - previous - 1 <= max_intervening_chars
                ]
                if starts:
                    # A later start minimizes the final span and avoids an
                    # early distractor occurrence.
                    next_states[position] = max(starts)
            if not next_states:
                return False
            states = next_states
        limit = max_intervening_chars * (len(candidate) - 1)
        return any(
            position - start - (len(candidate) - 1) <= limit
            for position, start in states.items()
        )

    if ordered_subsequence(normalized_term):
        return True
    # Chinese morphology often inserts/removes a single aspect particle
    # (例如“同意了娜塔莉娅” vs “同意娜塔莉娅”).  Permit at most one omitted
    # character, while retaining order and gap limits.  More than one missing
    # character is rejected to avoid turning this into a loose bag-of-chars
    # matcher.
    if len(normalized_term) >= 5:
        return any(
            ordered_subsequence(normalized_term[:index] + normalized_term[index + 1 :])
            for index in range(len(normalized_term))
        )
    return False
