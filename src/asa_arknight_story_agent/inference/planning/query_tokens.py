from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.lexicon import (
    COMMON_NON_ENTITY_WORDS,
    DOMAIN_RELATED_RETRIEVAL_TERMS,
    STORY_HINT_WORDS,
)
from asa_arknight_story_agent.inference.common.patterns import (
    CHAPTER_TOKEN_RE,
    CHINESE_TOKEN_SPLIT_RE,
    QUESTION_TOKEN_RE,
)
from asa_arknight_story_agent.inference.pipeline.constants import (
    ENTITY_EXCLUDE_MARKERS,
    NOISY_RETRIEVAL_TOKENS,
    NOISY_TOKEN_MARKERS,
    PRONOUN_REFERENCES,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def extract_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(match.group(0) for match in CHAPTER_TOKEN_RE.finditer(text))
    for raw_token in QUESTION_TOKEN_RE.findall(text):
        parts = [raw_token] if raw_token.isascii() else [part for part in CHINESE_TOKEN_SPLIT_RE.split(raw_token) if part]
        for part in parts:
            normalized = part.strip()
            normalized = re.sub(r"(城)(?:识|发|曝|揭|撞|送|遭|被|去|到)$", r"\1", normalized)
            if (
                not normalized
                or normalized in COMMON_NON_ENTITY_WORDS
                or normalized in NOISY_RETRIEVAL_TOKENS
                or normalized in PRONOUN_REFERENCES
                or len(normalized) == 1 and not normalized.isascii()
                or any(marker in normalized for marker in NOISY_TOKEN_MARKERS)
                or normalized.endswith("吗")
            ):
                continue
            tokens.append(normalized)
    return dedupe_keep_order(tokens)


def expand_related_retrieval_terms(terms: list[str], *, limit: int = 16) -> list[str]:
    related: list[str] = []
    for term in terms:
        compact = re.sub(r"\s+", "", term or "")
        if not compact:
            continue
        for key, values in DOMAIN_RELATED_RETRIEVAL_TERMS.items():
            if key in compact or compact in key:
                related.extend(values)
    return dedupe_keep_order(
        item
        for item in related
        if item and item not in COMMON_NON_ENTITY_WORDS and item not in NOISY_RETRIEVAL_TOKENS
    )[:limit]


def is_entity_candidate(token: str) -> bool:
    if (
        not token
        or token in STORY_HINT_WORDS
        or token in NOISY_RETRIEVAL_TOKENS
        or token in PRONOUN_REFERENCES
        or any(marker in token for marker in ENTITY_EXCLUDE_MARKERS)
    ):
        return False
    return True
