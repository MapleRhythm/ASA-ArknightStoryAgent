from __future__ import annotations

import json
from pathlib import Path

from asa_arknight_story_agent.data.story_text import normalize_text


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_story_key(story_key: str) -> str:
    normalized = story_key.replace("\\", "/").strip()
    if normalized.endswith(".txt"):
        normalized = normalized[:-4]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.lower().startswith("[uc]info/"):
        normalized = normalized[9:]
    return normalized.lower()


def normalize_stage_id(stage_id: str | None) -> str | None:
    if not stage_id:
        return None
    normalized = normalize_text(str(stage_id)).strip().lower()
    return normalized or None


def extract_stage_id_from_story_key(story_key: str) -> str | None:
    story_name = Path(story_key).name.lower()
    if story_name.startswith("level_"):
        stage_id = story_name[6:]
    else:
        return None

    for suffix in ("_beg", "_end"):
        if stage_id.endswith(suffix):
            trimmed = stage_id[: -len(suffix)]
            return trimmed or None
    return stage_id or None


def extract_activity_hint(relative_story_key: str) -> str | None:
    parts = Path(relative_story_key).parts
    if len(parts) > 1 and parts[0] == "activities":
        return parts[1].lower()
    if len(parts) > 2 and parts[0] == "obt" and parts[1] == "roguelike":
        return parts[2].lower()
    if len(parts) > 1:
        return parts[0].lower()
    return None


def render_zone_name(zone_meta: dict) -> str | None:
    parts: list[str] = []
    for key in ("zoneNameFirst", "zoneNameSecond", "zoneNameThird"):
        value = normalize_text(str(zone_meta.get(key) or ""))
        if value and value not in parts:
            parts.append(value)
    if not parts:
        return None
    return " ".join(parts)


def extract_stage_id(item: dict) -> str | None:
    required = item.get("requiredStages") or []
    if required:
        return required[0].get("stageId")
    return None


def build_story_review_lookup(excel_root: Path) -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    review_table = load_json(excel_root / "story_review_table.json")
    lookup: dict[str, dict] = {}
    stage_lookup: dict[str, dict] = {}
    activity_lookup: dict[str, str] = {}
    for group_id, group in review_table.items():
        group_name = normalize_text(str(group.get("name") or ""))
        if group_name:
            activity_lookup[group_id.lower()] = group_name
        for item in group.get("infoUnlockDatas", []):
            story_txt = item.get("storyTxt")
            if not story_txt:
                continue
            stage_id = normalize_stage_id(extract_stage_id(item))
            payload = {
                "story_group": item.get("storyGroup") or group_id,
                "story_group_name": group_name or None,
                "story_code": item.get("storyCode"),
                "story_name": item.get("storyName"),
                "avg_tag": item.get("avgTag"),
                "story_sort": item.get("storySort"),
                "stage_id": stage_id,
            }
            lookup[normalize_story_key(story_txt)] = payload
            if stage_id and stage_id not in stage_lookup:
                stage_lookup[stage_id] = payload
    return lookup, stage_lookup, activity_lookup


def build_character_name_lookup(excel_root: Path) -> dict[str, str]:
    character_table = load_json(excel_root / "character_table.json")
    lookup: dict[str, str] = {}
    for char_id, payload in character_table.items():
        if not isinstance(payload, dict):
            continue
        name = normalize_text(str(payload.get("name") or ""))
        if name:
            lookup[char_id] = name
    return lookup


def build_story_speaker_lookup(excel_root: Path) -> dict[str, str]:
    cached = getattr(build_story_speaker_lookup, "_cache", {})
    cache_key = excel_root.resolve().as_posix()
    if cache_key in cached:
        return cached[cache_key]

    lookup = build_character_name_lookup(excel_root)
    handbook_table = load_json(excel_root / "handbook_info_table.json")
    for npc_id, payload in (handbook_table.get("npcDict") or {}).items():
        if not isinstance(payload, dict):
            continue
        name = normalize_text(str(payload.get("name") or ""))
        if name:
            lookup[str(npc_id)] = name

    story_variables_path = excel_root.parent / "story" / "story_variables.json"
    if story_variables_path.exists():
        story_variables = load_json(story_variables_path)
        for avatar_key, target_id in story_variables.items():
            normalized_avatar = normalize_text(str(avatar_key or ""))
            normalized_target = normalize_text(str(target_id or ""))
            if not normalized_avatar or not normalized_target:
                continue
            resolved_name = lookup.get(normalized_target)
            if not resolved_name:
                continue
            lookup[normalized_avatar] = resolved_name
            lookup[f"${normalized_avatar}"] = resolved_name

    cached[cache_key] = lookup
    build_story_speaker_lookup._cache = cached  # type: ignore[attr-defined]
    return lookup


def build_stage_lookup(excel_root: Path) -> dict[str, dict]:
    stage_table = load_json(excel_root / "stage_table.json")
    return stage_table.get("stages", {})


def build_story_lookup(excel_root: Path) -> dict[str, dict]:
    story_table = load_json(excel_root / "story_table.json")
    return {normalize_story_key(key): value for key, value in story_table.items()}


def build_zone_lookup(excel_root: Path) -> dict[str, dict]:
    zone_table = load_json(excel_root / "zone_table.json")
    return zone_table.get("zones", {})


def build_chapter_lookup(excel_root: Path) -> dict[str, dict]:
    zone_table = load_json(excel_root / "zone_table.json")
    chapter_table = load_json(excel_root / "chapter_table.json")
    chapter_lookup: dict[str, dict] = {}
    for zone_id, payload in zone_table.get("mainlineAdditionInfo", {}).items():
        if not isinstance(payload, dict):
            continue
        chapter = chapter_table.get(payload.get("chapterId") or "")
        if not isinstance(chapter, dict):
            continue
        chapter_name = normalize_text(str(chapter.get("chapterName") or ""))
        if chapter_name:
            chapter_lookup[zone_id] = {
                "chapter_id": chapter.get("chapterId"),
                "chapter_name": chapter_name,
            }
    return chapter_lookup


def build_activity_lookup(excel_root: Path) -> dict[str, dict]:
    activity_table = load_json(excel_root / "activity_table.json")
    return activity_table.get("basicInfo", {})


def build_zone_to_activity_lookup(excel_root: Path) -> dict[str, str]:
    activity_table = load_json(excel_root / "activity_table.json")
    zone_to_activity = activity_table.get("zoneToActivity", {})
    return {str(key): str(value) for key, value in zone_to_activity.items() if key and value}


def build_roguelike_activity_lookup(excel_root: Path) -> dict[str, str]:
    roguelike_table = load_json(excel_root / "roguelike_topic_table.json")
    lookup: dict[str, str] = {}
    for topic_id, payload in (roguelike_table.get("topics") or {}).items():
        if not isinstance(payload, dict):
            continue
        topic_name = normalize_text(str(payload.get("name") or ""))
        if not topic_name:
            continue
        normalized_topic_id = normalize_text(str(payload.get("id") or topic_id)).lower()
        if normalized_topic_id:
            lookup[normalized_topic_id] = topic_name
        if normalized_topic_id.startswith("rogue_"):
            lookup[normalized_topic_id.replace("rogue_", "ro")] = topic_name
    return lookup
