from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import (
    best_prompt_text,
    evidence_text,
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


def select_prompt_evidence_coverage(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
) -> list[dict[str, Any]]:
    """Select a diverse evidence *set* with query-term coverage.

    This is deliberately an opt-in ablation.  It does not invent answer slots
    or inspect gold labels: query bigrams are weighted by inverse document
    frequency, and a candidate is rewarded only for covering terms not already
    covered by the selected set.  The normal score remains a tie-breaker, so a
    high-scoring duplicate cannot crowd out complementary evidence.
    """
    if prompt_evidence_top_k <= 0 or not evidence:
        return []
    candidates = dedupe_prompt_evidence_candidates(evidence)
    if len(candidates) <= prompt_evidence_top_k:
        return candidates[:prompt_evidence_top_k]
    query_tokens = text_similarity_tokens(
        "\n".join(
            [
                str(question or ""),
                str(getattr(hypothesis, "question", "") or ""),
                " ".join(getattr(hypothesis, "entities", []) or []),
                " ".join(getattr(hypothesis, "keywords", []) or []),
                str(getattr(hypothesis, "expected_answer_type", "") or ""),
            ]
        )
    )
    candidate_tokens = [text_similarity_tokens(evidence_text(item)) for item in candidates]
    document_frequency = {
        token: sum(token in tokens for tokens in candidate_tokens)
        for token in query_tokens
    }
    weights = {
        token: 1.0 / max(1, frequency)
        for token, frequency in document_frequency.items()
    }
    total_weight = sum(weights.values()) or 1.0
    base_scores = [prompt_evidence_score(item) for item in candidates]
    score_min, score_max = min(base_scores), max(base_scores)
    score_span = score_max - score_min
    normalized_scores = [
        1.0 if score_span <= 1e-9 else (score - score_min) / score_span
        for score in base_scores
    ]
    selected: list[int] = []
    covered: set[str] = set()
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < prompt_evidence_top_k:
        best_index = max(
            remaining,
            key=lambda index: (
                sum(
                    weights[token]
                    for token in (candidate_tokens[index] & query_tokens) - covered
                )
                / total_weight
                + 0.15 * normalized_scores[index],
                normalized_scores[index],
                -index,
            ),
        )
        selected.append(best_index)
        covered.update(candidate_tokens[best_index] & query_tokens)
        remaining.remove(best_index)
    return [candidates[index] for index in selected]
