from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.anchors.action_evidence import (
    action_target_score,
    best_action_target_evidence,
)
from asa_arknight_story_agent.inference.anchors.bundle_evidence import best_anchor_bundle_evidence
from asa_arknight_story_agent.inference.anchors.terms import (
    anchor_hit_count,
    extract_action_targets,
    extract_question_anchor_terms,
)
from asa_arknight_story_agent.inference.definitions.definition_evidence import (
    best_definition_evidence,
    is_definition_or_identity_question,
)
from asa_arknight_story_agent.inference.evidence.texts import (
    evidence_identity,
    evidence_score,
    evidence_text,
    prefer_direct_prompt_text,
)
from asa_arknight_story_agent.inference.common.lexicon import ACTION_COST_MARKERS, ACTION_PURPOSE_MARKERS
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import expand_related_retrieval_terms
from asa_arknight_story_agent.inference.reveal.detection import is_reveal_question
from asa_arknight_story_agent.inference.reveal.scoring import best_reveal_evidence


def pin_anchor_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    anchors = extract_question_anchor_terms(question, hypothesis)
    is_reveal = is_reveal_question(question, hypothesis)
    if not anchors and not is_reveal:
        return selected[:limit]
    action_targets = extract_action_targets(question + "\n" + hypothesis.question)
    related_anchors = expand_related_retrieval_terms(action_targets + anchors)

    pinned: list[dict[str, Any]] = []
    max_pinned = max(1, min(3, limit // 3 or 1))
    bundle_core_terms = action_targets or anchors[:3]
    bundle_terms = [*bundle_core_terms, *anchors[:8], *related_anchors[:12]]
    seen_bundle_terms: list[str] = []
    for term in bundle_terms:
        if term not in seen_bundle_terms:
            seen_bundle_terms.append(term)
    if is_definition_or_identity_question(question, hypothesis):
        pinned.extend(
            best_definition_evidence(
                evidence,
                anchors=anchors,
                limit=max(2, min(4, limit // 2 or 2)),
            )
        )
    pinned.extend(
        best_anchor_bundle_evidence(
            evidence,
            core_terms=bundle_core_terms,
            bundle_terms=seen_bundle_terms,
            limit=max_pinned,
        )
    )
    if is_reveal:
        reveal_pinned = best_reveal_evidence(
            question,
            hypothesis,
            evidence,
            limit=max(2, min(5, limit // 2 or 2)),
        )
        pinned.extend(prefer_direct_prompt_text(item) for item in reveal_pinned)
    if action_targets:
        purpose_evidence = best_action_target_evidence(evidence, action_targets, ACTION_PURPOSE_MARKERS)
        cost_evidence = best_action_target_evidence(evidence, action_targets, ACTION_COST_MARKERS)
        for item in (purpose_evidence, cost_evidence):
            if item is not None:
                pinned.append(item)
        action_pinned = [
            item
            for item in evidence
            if action_target_score(evidence_text(item), action_targets) >= 2
        ]
        pinned.extend(
            sorted(
                action_pinned,
                key=lambda item: (action_target_score(evidence_text(item), action_targets), evidence_score(item)),
                reverse=True,
            )[:max_pinned]
        )
    for item in evidence:
        text = evidence_text(item)
        if anchor_hit_count(text, anchors) < 2:
            continue
        pinned.append(item)
        if len(pinned) >= max_pinned:
            break

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in pinned + selected:
        identity = evidence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(prefer_direct_prompt_text(item) if is_reveal else item)
        if len(output) >= limit:
            break
    return output
