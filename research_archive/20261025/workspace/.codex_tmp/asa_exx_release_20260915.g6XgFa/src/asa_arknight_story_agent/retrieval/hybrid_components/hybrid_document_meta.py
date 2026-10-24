from __future__ import annotations

import re

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    MAIN_CHAPTER_SOURCE_RE,
    MOEGIRL_SOURCE_MARKERS,
    PROFILE_SOURCE_MARKERS,
    STAGE_NUMBER_RE,
    STORY_SOURCE_MARKERS,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_utils import extract_main_chapter_numbers


class HybridDocumentMetaMixin:
    def _document_source_type(self, document: dict) -> str:
        fields = " ".join(
            str(document.get(key) or "")
            for key in ("id", "source_path", "activity_name", "story_id", "activity_id", "avg_tag")
        )
        if any(marker in fields for marker in MOEGIRL_SOURCE_MARKERS):
            return "moegirl_background"
        if any(marker in fields for marker in PROFILE_SOURCE_MARKERS):
            return "profile"
        if "charword/" in fields:
            return "voice"
        if any(marker in fields for marker in STORY_SOURCE_MARKERS):
            return "story_text"
        return "other"

    def _document_text(self, document: dict) -> str:
        return str(document.get("search_text") or document.get("clean_text") or "")

    def _document_main_chapter_number(self, document: dict) -> int | None:
        fields = " ".join(
            str(document.get(key) or "")
            for key in ("activity_id", "story_id", "story_key", "source_path", "id", "search_text")
        )
        match = MAIN_CHAPTER_SOURCE_RE.search(fields)
        if match:
            return int(match.group(1))
        numbers = extract_main_chapter_numbers(fields)
        return numbers[0] if numbers else None

    def _document_stage_number(self, document: dict) -> int | None:
        stage_code = str(document.get("stage_code") or "")
        stage_match = re.search(r"(\d+)", stage_code)
        if stage_match:
            return int(stage_match.group(1))

        source = " ".join(
            str(document.get(key) or "")
            for key in ("story_id", "story_key", "source_path", "id")
        )
        source_match = STAGE_NUMBER_RE.search(source)
        if source_match:
            return int(source_match.group(1))
        return None
