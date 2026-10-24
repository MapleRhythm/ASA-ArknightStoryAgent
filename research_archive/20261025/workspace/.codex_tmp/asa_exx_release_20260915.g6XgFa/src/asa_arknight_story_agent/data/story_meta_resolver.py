from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asa_arknight_story_agent.data.story_metadata import (
    build_activity_lookup,
    build_chapter_lookup,
    build_roguelike_activity_lookup,
    build_stage_lookup,
    build_story_lookup,
    build_story_review_lookup,
    build_zone_lookup,
    build_zone_to_activity_lookup,
    extract_activity_hint,
    extract_stage_id_from_story_key,
    normalize_stage_id,
    normalize_story_key,
    render_zone_name,
)
from asa_arknight_story_agent.data.story_text import normalize_text


@dataclass(slots=True)
class StoryMetadataResolver:
    review_lookup: dict[str, dict]
    review_stage_lookup: dict[str, dict]
    review_activity_lookup: dict[str, str]
    stage_lookup: dict[str, dict]
    story_lookup: dict[str, dict]
    zone_lookup: dict[str, dict]
    chapter_lookup: dict[str, dict]
    activity_lookup: dict[str, dict]
    zone_to_activity_lookup: dict[str, str]
    roguelike_activity_lookup: dict[str, str]

    @classmethod
    def from_excel_root(cls, excel_root: Path) -> "StoryMetadataResolver":
        review_lookup, review_stage_lookup, review_activity_lookup = build_story_review_lookup(excel_root)
        return cls(
            review_lookup=review_lookup,
            review_stage_lookup=review_stage_lookup,
            review_activity_lookup=review_activity_lookup,
            stage_lookup=build_stage_lookup(excel_root),
            story_lookup=build_story_lookup(excel_root),
            zone_lookup=build_zone_lookup(excel_root),
            chapter_lookup=build_chapter_lookup(excel_root),
            activity_lookup=build_activity_lookup(excel_root),
            zone_to_activity_lookup=build_zone_to_activity_lookup(excel_root),
            roguelike_activity_lookup=build_roguelike_activity_lookup(excel_root),
        )

    def resolve(self, relative_story_key: str) -> dict:
        normalized_story_key = normalize_story_key(relative_story_key)
        review_meta = self.review_lookup.get(normalized_story_key, {})
        story_meta = self.story_lookup.get(normalized_story_key, {})
        trigger = story_meta.get("trigger") or {}
        condition = story_meta.get("condition") or {}
        required_stages = condition.get("requiredStages") or []
        stage_id = normalize_stage_id(
            review_meta.get("stage_id")
            or trigger.get("key")
            or (required_stages[0].get("stageId") if required_stages else None)
            or extract_stage_id_from_story_key(normalized_story_key)
        )
        if not review_meta and stage_id:
            review_meta = self.review_stage_lookup.get(stage_id, {})
        stage_meta = self.stage_lookup.get(stage_id or "", {})
        zone_id = stage_meta.get("zoneId")
        zone_meta = self.zone_lookup.get(zone_id or "", {})
        chapter_meta = self.chapter_lookup.get(zone_id or "", {})

        activity_hint = extract_activity_hint(normalized_story_key)
        activity_id = (
            review_meta.get("story_group")
            or self.zone_to_activity_lookup.get(zone_id or "")
            or activity_hint
        )
        activity_meta = self.activity_lookup.get(str(activity_id or ""), {})
        activity_name = (
            review_meta.get("story_group_name")
            or normalize_text(str(activity_meta.get("name") or ""))
            or self.review_activity_lookup.get(str(activity_id or "").lower())
            or self.roguelike_activity_lookup.get(str(activity_id or "").lower())
            or None
        )
        zone_name = render_zone_name(zone_meta)
        chapter_name = chapter_meta.get("chapter_name")

        return {
            "story_key": normalized_story_key,
            "story_id": story_meta.get("id") or relative_story_key.replace("/", "_"),
            "activity_id": review_meta.get("story_group") or activity_id,
            "activity_name": activity_name,
            "story_name": review_meta.get("story_name") or stage_meta.get("name"),
            "story_code": review_meta.get("story_code") or stage_meta.get("code"),
            "avg_tag": review_meta.get("avg_tag"),
            "story_sort": review_meta.get("story_sort"),
            "stage_id": stage_id,
            "stage_code": stage_meta.get("code"),
            "stage_name": stage_meta.get("name"),
            "stage_type": stage_meta.get("stageType"),
            "zone_id": zone_id,
            "zone_name": zone_name,
            "chapter_name": chapter_name,
            "trigger_type": trigger.get("type"),
        }
