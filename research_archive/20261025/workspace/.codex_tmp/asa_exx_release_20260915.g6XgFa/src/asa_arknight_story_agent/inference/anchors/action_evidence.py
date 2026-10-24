from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_score, evidence_text
from asa_arknight_story_agent.inference.common.lexicon import ACTION_ANSWER_MARKERS, ACTION_WORDS
from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta


def action_target_score(text: str, targets: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not targets:
        return 0
    target_hit = any(target and target in compact_text for target in targets)
    if not target_hit:
        return 0
    action_hit = any(action in compact_text for action in ACTION_WORDS)
    marker_hit = any(marker in compact_text for marker in ACTION_ANSWER_MARKERS)
    return int(target_hit) + int(action_hit) + int(marker_hit)


def action_target_marker_score(text: str, targets: list[str], markers: tuple[str, ...]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not targets:
        return 0
    if not any(target and target in compact_text for target in targets):
        return 0
    marker_hits = sum(1 for marker in markers if marker in compact_text)
    if marker_hits <= 0:
        return 0
    action_hit = any(action in compact_text for action in ACTION_WORDS)
    return marker_hits + int(action_hit)


def best_action_target_evidence(
    evidence: list[dict[str, Any]],
    targets: list[str],
    markers: tuple[str, ...],
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in evidence
        if action_target_marker_score(evidence_text(item), targets, markers) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            action_target_marker_score(evidence_text(item), targets, markers),
            action_target_score(evidence_text(item), targets),
            evidence_score(item),
        ),
    )
