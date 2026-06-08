from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_text, prompt_evidence_score
from asa_arknight_story_agent.inference.evidence.prompt_similarity import (
    dedupe_prompt_evidence_candidates,
    jaccard_similarity,
    text_similarity_tokens,
)


def select_prompt_evidence_mmr(
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
    lambda_mult: float,
) -> list[dict[str, Any]]:
    if prompt_evidence_top_k <= 0 or not evidence:
        return []
    candidates = dedupe_prompt_evidence_candidates(evidence)
    if len(candidates) <= prompt_evidence_top_k:
        return candidates[:prompt_evidence_top_k]
    scores = [prompt_evidence_score(item) for item in candidates]
    score_min = min(scores)
    score_max = max(scores)
    score_span = score_max - score_min
    normalized_scores = [
        1.0 if score_span <= 1e-9 else (score - score_min) / score_span
        for score in scores
    ]
    token_sets = [
        text_similarity_tokens(evidence_text(item))
        for item in candidates
    ]

    selected_indices: list[int] = []
    remaining_indices = set(range(len(candidates)))
    while remaining_indices and len(selected_indices) < prompt_evidence_top_k:
        best_index = None
        best_score = float("-inf")
        for index in remaining_indices:
            diversity_penalty = 0.0
            if selected_indices:
                diversity_penalty = max(
                    jaccard_similarity(token_sets[index], token_sets[selected_index])
                    for selected_index in selected_indices
                )
            mmr_score = lambda_mult * normalized_scores[index] - (1.0 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        if best_index is None:
            break
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [candidates[index] for index in selected_indices]
