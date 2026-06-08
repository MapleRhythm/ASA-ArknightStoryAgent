from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text, evidence_identity


def text_similarity_tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    cjk_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = {
        cjk_chars[index] + cjk_chars[index + 1]
        for index in range(len(cjk_chars) - 1)
    }
    ascii_tokens = set(re.findall(r"[a-z0-9_]{2,}", normalized, flags=re.IGNORECASE))
    return cjk_bigrams | ascii_tokens


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def dedupe_prompt_evidence_candidates(
    evidence: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    seen_token_sets: list[set[str]] = []
    for item in evidence:
        identity = evidence_identity(item)
        if identity in seen_identities:
            continue
        text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        token_set = text_similarity_tokens(text)
        if token_set and any(jaccard_similarity(token_set, seen) >= similarity_threshold for seen in seen_token_sets):
            continue
        seen_identities.add(identity)
        if token_set:
            seen_token_sets.append(token_set)
        output.append(item)
    return output
