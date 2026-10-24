from __future__ import annotations

from collections import Counter
from typing import Any

from asa_arknight_story_agent.retrieval.minirag import document_chapter_scope_key, document_chapter_scope_label
from asa_arknight_story_agent.retrieval.storyline import document_storyline_scopes, storyline_scope_label


def infer_dominant_minirag_chapter_scope(
    *ranked_lists: list[dict[str, Any]],
    max_items: int = 40,
) -> dict[str, Any] | None:
    scope_scores: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    scope_labels: dict[str, str] = {}
    for source_index, ranked in enumerate(ranked_lists):
        source_weight = 1.25 if source_index == 0 else 1.0
        for rank, item in enumerate(ranked[:max_items]):
            doc = item.get("document") or {}
            if not isinstance(doc, dict):
                continue
            scope = document_chapter_scope_key(doc)
            if not scope:
                continue
            scope_scores[scope] += source_weight / ((rank + 1) ** 0.5)
            scope_counts[scope] += 1
            scope_labels.setdefault(scope, document_chapter_scope_label(doc) or scope)
    if not scope_scores:
        return None
    ranked_scopes = sorted(
        scope_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    scope, score = ranked_scopes[0]
    if scope_counts[scope] < 3:
        return None
    runner_up_score = float(ranked_scopes[1][1]) if len(ranked_scopes) > 1 else 0.0
    dominance_ratio = float(score) / max(runner_up_score, 1e-6)
    if runner_up_score > 0 and dominance_ratio < 1.15 and scope_counts[scope] < 6:
        return None
    return {
        "scope": scope,
        "label": scope_labels.get(scope) or scope,
        "score": float(score),
        "count": int(scope_counts[scope]),
        "runner_up_score": runner_up_score,
        "dominance_ratio": dominance_ratio,
        "candidates": [
            {
                "scope": candidate_scope,
                "label": scope_labels.get(candidate_scope) or candidate_scope,
                "score": float(candidate_score),
                "count": int(scope_counts[candidate_scope]),
            }
            for candidate_scope, candidate_score in ranked_scopes[:5]
        ],
    }


def infer_dominant_storyline_scope(
    *ranked_lists: list[dict[str, Any]],
    max_items: int = 40,
) -> dict[str, Any] | None:
    scope_scores: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    for source_index, ranked in enumerate(ranked_lists):
        source_weight = 1.25 if source_index == 0 else 1.0
        for rank, item in enumerate(ranked[:max_items]):
            doc = item.get("document") or {}
            if not isinstance(doc, dict):
                continue
            scopes = document_storyline_scopes(doc)
            if not scopes:
                continue
            for scope in scopes:
                scope_scores[scope] += source_weight / ((rank + 1) ** 0.5)
                scope_counts[scope] += 1
    if not scope_scores:
        return None
    ranked_scopes = sorted(scope_scores.items(), key=lambda item: item[1], reverse=True)
    scope, score = ranked_scopes[0]
    if scope_counts[scope] < 3:
        return None
    runner_up_score = float(ranked_scopes[1][1]) if len(ranked_scopes) > 1 else 0.0
    dominance_ratio = float(score) / max(runner_up_score, 1e-6)
    if runner_up_score > 0 and dominance_ratio < 1.1 and scope_counts[scope] < 6:
        return None
    return {
        "scope": scope,
        "label": storyline_scope_label(scope),
        "score": float(score),
        "count": int(scope_counts[scope]),
        "runner_up_score": runner_up_score,
        "dominance_ratio": dominance_ratio,
        "candidates": [
            {
                "scope": candidate_scope,
                "label": storyline_scope_label(candidate_scope),
                "score": float(candidate_score),
                "count": int(scope_counts[candidate_scope]),
            }
            for candidate_scope, candidate_score in ranked_scopes[:5]
        ],
    }


def filter_hits_by_chapter_scope(
    hits: list[dict[str, Any]],
    chapter_scope: str,
) -> list[dict[str, Any]]:
    if not chapter_scope:
        return hits
    scoped_hits: list[dict[str, Any]] = []
    for item in hits:
        doc = item.get("document") or {}
        if isinstance(doc, dict) and document_chapter_scope_key(doc) == chapter_scope:
            scoped_hits.append(item)
    return scoped_hits
