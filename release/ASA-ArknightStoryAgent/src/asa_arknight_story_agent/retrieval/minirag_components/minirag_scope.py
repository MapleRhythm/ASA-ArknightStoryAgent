from __future__ import annotations

import re
from typing import Any


CHAPTER_SCOPE_EXCLUDED_ACTIVITY_IDS = frozenset(
    {
        "operator_voice",
        "operator_handbook",
        "moegirl_lore",
        "obt",
    }
)


def document_chapter_scope_key(document: dict[str, Any]) -> str:
    """Return the story/event scope used to isolate MiniRAG graph traversal."""
    activity_id = str(document.get("activity_id") or "").strip()
    if activity_id and activity_id not in CHAPTER_SCOPE_EXCLUDED_ACTIVITY_IDS:
        return f"activity:{activity_id}"

    story_id = str(document.get("story_id") or document.get("story_key") or "").strip()
    match = re.search(r"(?:^|/)activities/([^/]+)/", story_id)
    if match:
        return f"activity:{match.group(1)}"
    match = re.search(r"(?:^|/)(?:level_)?main[_-](\d{1,2})(?:[-_/]|$)", story_id, flags=re.IGNORECASE)
    if match:
        return f"activity:main_{int(match.group(1))}"

    zone_id = str(document.get("zone_id") or "").strip()
    if zone_id.startswith("main_"):
        return f"activity:{zone_id}"
    return ""


def document_chapter_scope_label(document: dict[str, Any]) -> str:
    scope = document_chapter_scope_key(document)
    if not scope:
        return ""
    activity_name = str(document.get("activity_name") or "").strip()
    chapter_name = str(document.get("chapter_name") or "").strip()
    zone_name = str(document.get("zone_name") or "").strip()
    label_parts = [part for part in (activity_name, chapter_name, zone_name) if part]
    if label_parts:
        return f"{scope} ({' / '.join(label_parts[:2])})"
    return scope
