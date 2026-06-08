from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import (
    best_prompt_text,
    evidence_identity,
    is_web_context_item,
    prefer_direct_prompt_text,
    prompt_evidence_score,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_similarity import (
    dedupe_prompt_evidence_candidates,
    jaccard_similarity,
    text_similarity_tokens,
)


def merge_forced_prompt_evidence(
    forced: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_token_sets: list[set[str]] = []
    for item in forced + selected:
        identity = evidence_identity(item)
        if identity in seen:
            continue
        text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        token_set = text_similarity_tokens(text)
        if token_set and any(jaccard_similarity(token_set, seen_tokens) >= 0.82 for seen_tokens in seen_token_sets):
            continue
        seen.add(identity)
        if token_set:
            seen_token_sets.append(token_set)
        output.append(prefer_direct_prompt_text(item) if is_web_context_item(item) else item)
        if len(output) >= limit:
            break
    return output


def apply_pyramid_evidence_order(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(evidence) <= 2:
        return evidence
    return [evidence[0], *evidence[2:], evidence[1]]


def select_prompt_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
) -> list[dict[str, Any]]:
    del question, hypothesis
    if prompt_evidence_top_k <= 0 or not evidence:
        return []
    candidates = dedupe_prompt_evidence_candidates(evidence)
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (prompt_evidence_score(pair[1]), -pair[0]),
        reverse=True,
    )
    return [item for _, item in ranked[:prompt_evidence_top_k]]
