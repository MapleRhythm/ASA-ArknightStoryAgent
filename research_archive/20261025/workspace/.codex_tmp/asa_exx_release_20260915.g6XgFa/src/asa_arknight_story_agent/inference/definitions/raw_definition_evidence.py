from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.definitions.definition_scoring import is_definition_or_identity_question
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.definitions.raw_definition_candidates import (
    allow_rogue_raw_story_search,
    raw_exact_candidates_from_file,
)
from asa_arknight_story_agent.inference.definitions.raw_definition_terms import raw_exact_anchor_terms
from asa_arknight_story_agent.inference.definitions.raw_story_cleaning import (
    clean_raw_story_context,
    raw_line_context,
)
from asa_arknight_story_agent.inference.definitions.raw_story_files import raw_story_text_files

__all__ = [
    "raw_story_text_files",
    "raw_exact_anchor_terms",
    "raw_line_context",
    "clean_raw_story_context",
    "raw_exact_definition_evidence",
]


def raw_exact_definition_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not is_definition_or_identity_question(question, hypothesis):
        return []
    if hypothesis.query_type in {"reveal", "mystery"} or any(term in question for term in ("阴谋", "识破", "曝光")):
        return []
    anchors = raw_exact_anchor_terms(question, hypothesis)
    if not anchors:
        return []
    compact_anchors = [re.sub(r"\s+", "", anchor) for anchor in anchors if anchor]
    allow_rogue = allow_rogue_raw_story_search(question)
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for path in raw_story_text_files():
        normalized_path = path.as_posix().lower()
        if not allow_rogue and "/obt/rogue/" in normalized_path:
            continue
        item = raw_exact_candidates_from_file(
            path=path,
            anchors=anchors,
            compact_anchors=compact_anchors,
        )
        if item is None:
            continue
        raw_exact = item.get("raw_exact") or {}
        clean_text = str((item.get("document") or {}).get("clean_text") or "")
        candidates.append((int(raw_exact.get("score") or 0), -len(clean_text), str(path), item))
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return [item for _, _, _, item in candidates[:limit]]
