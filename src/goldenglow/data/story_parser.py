from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from goldenglow.config import MANUAL_OPERATOR_ALIAS_SOURCE_PATH

NAME_LINE_RE = re.compile(r'^\[name="([^"]+)"\]\s*(.*)$')
HEADER_RE = re.compile(r"^\[HEADER.*\]\s*(.+)$")
TAG_RE = re.compile(r"<[^>]+>")
DIALOG_HEAD_RE = re.compile(r'dialogHead="([^"]+)"', re.IGNORECASE)
ALIAS_NAME_PATTERN = r"[A-Za-z\u4e00-\u9fff·\.\-]{1,32}"
PROFILE_CODENAME_RE = re.compile(rf"^【代号】\s*({ALIAS_NAME_PATTERN})$", re.MULTILINE)
REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:本名|真名)(?:为|是|叫|：|:)?\s*({ALIAS_NAME_PATTERN})(?:[，。；\n]|$)"
)
CODENAME_REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:干员)?({ALIAS_NAME_PATTERN})，本名({ALIAS_NAME_PATTERN})(?:[，。；\n]|$)"
)
REAL_NAME_OPERATOR_CODENAME_RE = re.compile(
    rf"(?:^|[。；\n])({ALIAS_NAME_PATTERN})，[\s\S]{{0,200}}?以干员[\"“”']+({ALIAS_NAME_PATTERN})[\"“”']+身份"
)
SELF_INTRO_REAL_NAME_RE = re.compile(
    rf"(?:^|[。；\n])(?:您好|你好)?[，,]?(?:我叫|我是)\s*({ALIAS_NAME_PATTERN})(?:[，,。]|$)"
)
SELF_INTRO_CODENAME_RE = re.compile(
    rf"(?:代号[^。\n]{{0,40}}?)?(?:您叫我|您可以叫我|就叫我|叫我)"
    rf"[，,:：\s\"“”']*({ALIAS_NAME_PATTERN})[\"“”']*(?:就好|吧|即可|好了|。|，|$)"
)
OPERATOR_IDENTITY_CODENAME_RE = re.compile(
    rf"以干员[\"“”']+({ALIAS_NAME_PATTERN})[\"“”']+身份"
)


@dataclass(slots=True)
class StorySegment:
    speaker: str | None
    text: str
    segment_type: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    cleaned = TAG_RE.sub("", text)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_story_key(story_key: str) -> str:
    normalized = story_key.replace("\\", "/").strip()
    if normalized.endswith(".txt"):
        normalized = normalized[:-4]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.lower().startswith("[uc]info/"):
        normalized = normalized[9:]
    return normalized.lower()


def _normalize_stage_id(stage_id: str | None) -> str | None:
    if not stage_id:
        return None
    normalized = normalize_text(str(stage_id)).strip().lower()
    return normalized or None


def _extract_stage_id_from_story_key(story_key: str) -> str | None:
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


def _extract_activity_hint(relative_story_key: str) -> str | None:
    parts = Path(relative_story_key).parts
    if len(parts) > 1 and parts[0] == "activities":
        return parts[1].lower()
    if len(parts) > 2 and parts[0] == "obt" and parts[1] == "roguelike":
        return parts[2].lower()
    if len(parts) > 1:
        return parts[0].lower()
    return None


def _render_zone_name(zone_meta: dict) -> str | None:
    parts: list[str] = []
    for key in ("zoneNameFirst", "zoneNameSecond", "zoneNameThird"):
        value = normalize_text(str(zone_meta.get(key) or ""))
        if value and value not in parts:
            parts.append(value)
    if not parts:
        return None
    return " ".join(parts)


def render_segment(segment: StorySegment) -> str:
    if segment.speaker:
        return f"{segment.speaker}：{segment.text}"
    return segment.text


def _build_story_speaker_lookup(excel_root: Path) -> dict[str, str]:
    cached = getattr(_build_story_speaker_lookup, "_cache", {})
    cache_key = excel_root.resolve().as_posix()
    if cache_key in cached:
        return cached[cache_key]

    lookup = _build_character_name_lookup(excel_root)
    handbook_table = _load_json(excel_root / "handbook_info_table.json")
    for npc_id, payload in (handbook_table.get("npcDict") or {}).items():
        if not isinstance(payload, dict):
            continue
        name = normalize_text(str(payload.get("name") or ""))
        if name:
            lookup[str(npc_id)] = name

    story_variables_path = excel_root.parent / "story" / "story_variables.json"
    if story_variables_path.exists():
        story_variables = _load_json(story_variables_path)
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
    _build_story_speaker_lookup._cache = cached  # type: ignore[attr-defined]
    return lookup


def _resolve_dialog_head_speaker(line: str, speaker_lookup: dict[str, str]) -> str | None:
    match = DIALOG_HEAD_RE.search(line)
    if not match:
        return None
    dialog_head = normalize_text(match.group(1))
    if not dialog_head:
        return None
    return speaker_lookup.get(dialog_head) or speaker_lookup.get(dialog_head.lstrip("$")) or dialog_head


def _extract_tag_trailing_text(line: str) -> str:
    if "]" not in line:
        return ""
    trailing = line.rsplit("]", 1)[1].strip()
    if trailing.startswith("\\"):
        trailing = trailing[1:].strip()
    return normalize_text(trailing)


def parse_story_file(path: Path, speaker_lookup: dict[str, str] | None = None) -> list[StorySegment]:
    segments: list[StorySegment] = []
    pending_plain: list[str] = []
    pending_speaker: str | None = None
    resolved_speaker_lookup = speaker_lookup or {}

    def flush_pending_plain() -> None:
        nonlocal pending_speaker
        if not pending_plain:
            return
        text = normalize_text(" ".join(pending_plain))
        pending_plain.clear()
        if text:
            speaker = pending_speaker
            segment_type = "dialogue" if speaker else "narration"
            _append_segment(StorySegment(speaker=speaker, text=text, segment_type=segment_type))
        pending_speaker = None

    def _append_segment(segment: StorySegment) -> None:
        if not segment.text:
            return
        if (
            segments
            and segments[-1].speaker == segment.speaker
            and segments[-1].segment_type == segment.segment_type
        ):
            segments[-1].text = f"{segments[-1].text}\n{segment.text}"
            return
        segments.append(segment)

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush_pending_plain()
            continue

        header_match = HEADER_RE.match(line)
        if header_match:
            flush_pending_plain()
            header_text = normalize_text(header_match.group(1))
            if header_text:
                _append_segment(StorySegment(speaker=None, text=header_text, segment_type="header"))
            continue

        speaker_match = NAME_LINE_RE.match(line)
        if speaker_match:
            flush_pending_plain()
            speaker = normalize_text(speaker_match.group(1))
            text = normalize_text(speaker_match.group(2))
            if text:
                _append_segment(StorySegment(speaker=speaker, text=text, segment_type="dialogue"))
            continue

        if line.startswith("["):
            flush_pending_plain()
            pending_speaker = _resolve_dialog_head_speaker(line, resolved_speaker_lookup)
            trailing_text = _extract_tag_trailing_text(line)
            if trailing_text:
                segment_type = "dialogue" if pending_speaker else "narration"
                _append_segment(StorySegment(speaker=pending_speaker, text=trailing_text, segment_type=segment_type))
                pending_speaker = None
            continue

        pending_plain.append(line)

    flush_pending_plain()
    return segments


def _build_story_review_lookup(excel_root: Path) -> dict[str, dict]:
    review_table = _load_json(excel_root / "story_review_table.json")
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
            stage_id = _normalize_stage_id(_extract_stage_id(item))
            payload = {
                "story_group": item.get("storyGroup") or group_id,
                "story_group_name": group_name or None,
                "story_code": item.get("storyCode"),
                "story_name": item.get("storyName"),
                "avg_tag": item.get("avgTag"),
                "story_sort": item.get("storySort"),
                "stage_id": stage_id,
            }
            lookup[_normalize_story_key(story_txt)] = payload
            if stage_id and stage_id not in stage_lookup:
                stage_lookup[stage_id] = payload
    _build_story_review_lookup.stage_lookup = stage_lookup  # type: ignore[attr-defined]
    _build_story_review_lookup.activity_lookup = activity_lookup  # type: ignore[attr-defined]
    return lookup


def _build_stage_lookup(excel_root: Path) -> dict[str, dict]:
    stage_table = _load_json(excel_root / "stage_table.json")
    return stage_table.get("stages", {})


def _build_story_lookup(excel_root: Path) -> dict[str, dict]:
    story_table = _load_json(excel_root / "story_table.json")
    return {_normalize_story_key(key): value for key, value in story_table.items()}


def _build_zone_lookup(excel_root: Path) -> dict[str, dict]:
    zone_table = _load_json(excel_root / "zone_table.json")
    return zone_table.get("zones", {})


def _build_chapter_lookup(excel_root: Path) -> dict[str, dict]:
    zone_table = _load_json(excel_root / "zone_table.json")
    chapter_table = _load_json(excel_root / "chapter_table.json")
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


def _build_activity_lookup(excel_root: Path) -> dict[str, dict]:
    activity_table = _load_json(excel_root / "activity_table.json")
    return activity_table.get("basicInfo", {})


def _build_zone_to_activity_lookup(excel_root: Path) -> dict[str, str]:
    activity_table = _load_json(excel_root / "activity_table.json")
    zone_to_activity = activity_table.get("zoneToActivity", {})
    return {str(key): str(value) for key, value in zone_to_activity.items() if key and value}


def _build_roguelike_activity_lookup(excel_root: Path) -> dict[str, str]:
    roguelike_table = _load_json(excel_root / "roguelike_topic_table.json")
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


def _build_character_name_lookup(excel_root: Path) -> dict[str, str]:
    character_table = _load_json(excel_root / "character_table.json")
    lookup: dict[str, str] = {}
    for char_id, payload in character_table.items():
        if not isinstance(payload, dict):
            continue
        name = normalize_text(str(payload.get("name") or ""))
        if name:
            lookup[char_id] = name
    return lookup


def _clean_alias_name(candidate: str, *, min_len: int = 2) -> str:
    cleaned = normalize_text(candidate).strip("，。；：:、,.;!?？！()（）[]【】<>《》\"' ")
    if len(cleaned) < min_len:
        return ""
    cleaned = cleaned.removesuffix("就好").removesuffix("好了").removesuffix("即可")
    cleaned = cleaned.removesuffix("吧").removesuffix("啊").removesuffix("呀").removesuffix("吗").removesuffix("呢").removesuffix("了")
    if len(cleaned) > 12:
        return ""
    if any(token in cleaned for token in ("干员", "本名", "真名", "代号", "小姐", "先生")):
        return ""
    if "的" in cleaned:
        return ""
    if any(
        token in cleaned
        for token in (
            "代表我",
            "开始吧",
            "可行不通",
            "怎么样",
            "的话",
            "不用担心",
            "多多指教",
            "什么",
        )
    ):
        return ""
    return cleaned


def _extract_real_name_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in CODENAME_REAL_NAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(2))
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_OPERATOR_CODENAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    for match in SELF_INTRO_REAL_NAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1))
        if cleaned:
            candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def _extract_codename_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in PROFILE_CODENAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in CODENAME_REAL_NAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in REAL_NAME_OPERATOR_CODENAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(2), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in SELF_INTRO_CODENAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    for match in OPERATOR_IDENTITY_CODENAME_RE.finditer(text):
        cleaned = _clean_alias_name(match.group(1), min_len=1)
        if cleaned:
            candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def build_operator_alias_lookup(excel_root: Path) -> dict[str, list[str]]:
    cache_key = f"{excel_root.resolve().as_posix()}::{MANUAL_OPERATOR_ALIAS_SOURCE_PATH.resolve().as_posix()}"
    cached = getattr(build_operator_alias_lookup, "_cache", {})
    if cache_key in cached:
        return cached[cache_key]

    raw_groups = json.loads(MANUAL_OPERATOR_ALIAS_SOURCE_PATH.read_text(encoding="utf-8"))
    alias_lookup: dict[str, set[str]] = {}
    for primary_alias, related_aliases in raw_groups.items():
        if not isinstance(related_aliases, list):
            continue
        group = [primary_alias, *related_aliases]
        normalized_group: list[str] = []
        for candidate in group:
            cleaned = normalize_text(str(candidate or ""))
            if cleaned and cleaned not in normalized_group:
                normalized_group.append(cleaned)
        if len(normalized_group) <= 1:
            continue
        for alias in normalized_group:
            related = alias_lookup.setdefault(alias, set())
            related.update(item for item in normalized_group if item != alias)

    normalized_lookup = {alias: sorted(related) for alias, related in sorted(alias_lookup.items()) if related}
    cached[cache_key] = normalized_lookup
    build_operator_alias_lookup._cache = cached  # type: ignore[attr-defined]
    return normalized_lookup


def _collect_related_aliases(text: str, alias_lookup: dict[str, list[str]]) -> list[str]:
    if not text or not alias_lookup:
        return []
    related_terms: list[str] = []
    for alias, related in alias_lookup.items():
        if alias in text:
            related_terms.extend(item for item in related if item not in text)
    return list(dict.fromkeys(related_terms))


def _extract_stage_id(item: dict) -> str | None:
    required = item.get("requiredStages") or []
    if required:
        return required[0].get("stageId")
    return None


def _resolve_story_meta(relative_story_key: str, excel_root: Path) -> dict:
    review_lookup = _resolve_story_meta.review_lookup
    review_stage_lookup = _resolve_story_meta.review_stage_lookup
    review_activity_lookup = _resolve_story_meta.review_activity_lookup
    stage_lookup = _resolve_story_meta.stage_lookup
    story_lookup = _resolve_story_meta.story_lookup
    zone_lookup = _resolve_story_meta.zone_lookup
    chapter_lookup = _resolve_story_meta.chapter_lookup
    activity_lookup = _resolve_story_meta.activity_lookup
    zone_to_activity_lookup = _resolve_story_meta.zone_to_activity_lookup
    roguelike_activity_lookup = _resolve_story_meta.roguelike_activity_lookup

    normalized_story_key = _normalize_story_key(relative_story_key)
    review_meta = review_lookup.get(normalized_story_key, {})
    story_meta = story_lookup.get(normalized_story_key, {})
    trigger = story_meta.get("trigger") or {}
    condition = story_meta.get("condition") or {}
    required_stages = condition.get("requiredStages") or []
    stage_id = _normalize_stage_id(
        review_meta.get("stage_id")
        or trigger.get("key")
        or (required_stages[0].get("stageId") if required_stages else None)
        or _extract_stage_id_from_story_key(normalized_story_key)
    )
    if not review_meta and stage_id:
        review_meta = review_stage_lookup.get(stage_id, {})
    stage_meta = stage_lookup.get(stage_id or "", {})
    zone_id = stage_meta.get("zoneId")
    zone_meta = zone_lookup.get(zone_id or "", {})
    chapter_meta = chapter_lookup.get(zone_id or "", {})

    activity_hint = _extract_activity_hint(normalized_story_key)
    activity_id = (
        review_meta.get("story_group")
        or zone_to_activity_lookup.get(zone_id or "")
        or activity_hint
    )
    activity_meta = activity_lookup.get(str(activity_id or ""), {})
    activity_name = (
        review_meta.get("story_group_name")
        or normalize_text(str(activity_meta.get("name") or ""))
        or review_activity_lookup.get(str(activity_id or "").lower())
        or roguelike_activity_lookup.get(str(activity_id or "").lower())
        or None
    )
    zone_name = _render_zone_name(zone_meta)
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


_resolve_story_meta.review_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.review_stage_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.review_activity_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.stage_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.story_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.zone_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.chapter_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.activity_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.zone_to_activity_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.roguelike_activity_lookup = {}  # type: ignore[attr-defined]


def build_handbook_documents(excel_root: Path) -> list[dict]:
    handbook_table = _load_json(excel_root / "handbook_info_table.json")
    handbook_dict = handbook_table.get("handbookDict", {})
    character_name_lookup = _build_character_name_lookup(excel_root)
    alias_lookup = build_operator_alias_lookup(excel_root)

    documents: list[dict] = []
    source_path = (excel_root / "handbook_info_table.json").as_posix()

    for char_id, payload in handbook_dict.items():
        if not isinstance(payload, dict):
            continue
        codename = normalize_text(character_name_lookup.get(char_id, ""))
        story_blocks = payload.get("storyTextAudio") or []
        for block_index, block in enumerate(story_blocks):
            if not isinstance(block, dict):
                continue
            story_title = normalize_text(str(block.get("storyTitle") or ""))
            stories = block.get("stories") or []
            text_parts: list[str] = []
            for item in stories:
                if not isinstance(item, dict):
                    continue
                story_text = normalize_text(str(item.get("storyText") or ""))
                if story_text:
                    text_parts.append(story_text)
            if not text_parts:
                continue

            clean_text_parts = [part for part in (story_title, "\n\n".join(text_parts)) if part]
            clean_text = "\n".join(clean_text_parts).strip()
            if not clean_text:
                continue

            search_parts = [
                codename,
                char_id,
                "干员档案",
                story_title,
                clean_text,
            ]
            search_parts.extend(_collect_related_aliases("\n".join(search_parts), alias_lookup))
            document = {
                "id": f"handbook/{char_id}#chunk-{block_index:04d}",
                "chunk_index": block_index,
                "source_path": source_path,
                "clean_text": clean_text,
                "search_text": "\n".join(part for part in search_parts if part).strip(),
                "segments": [
                    {
                        "speaker": None,
                        "text": clean_text,
                        "segment_type": "handbook",
                    }
                ],
                "story_key": f"handbook/{char_id}",
                "story_id": f"handbook/{char_id}",
                "activity_id": "operator_handbook",
                "activity_name": "干员档案",
                "story_name": story_title or codename or char_id,
                "story_code": None,
                "avg_tag": "档案",
                "story_sort": block_index,
                "stage_id": None,
                "stage_code": None,
                "stage_name": None,
                "stage_type": None,
                "zone_id": None,
                "zone_name": None,
                "chapter_name": None,
                "trigger_type": "HANDBOOK",
            }
            documents.append(document)
    return documents


def build_charword_documents(excel_root: Path) -> list[dict]:
    charword_table = _load_json(excel_root / "charword_table.json")
    char_words = charword_table.get("charWords", {})
    character_name_lookup = _build_character_name_lookup(excel_root)
    alias_lookup = build_operator_alias_lookup(excel_root)

    documents: list[dict] = []
    source_path = (excel_root / "charword_table.json").as_posix()

    for voice_id, payload in char_words.items():
        if not isinstance(payload, dict):
            continue
        char_id = str(payload.get("charId") or "").strip()
        voice_text = normalize_text(str(payload.get("voiceText") or ""))
        if not char_id or not voice_text:
            continue
        codename = normalize_text(character_name_lookup.get(char_id, ""))
        voice_title = normalize_text(str(payload.get("voiceTitle") or ""))
        clean_text_parts = [part for part in (voice_title, voice_text) if part]
        clean_text = "\n".join(clean_text_parts).strip()
        search_parts = [
            codename,
            char_id,
            "干员语音",
            voice_title,
            voice_text,
        ]
        search_parts.extend(_collect_related_aliases("\n".join(search_parts), alias_lookup))
        documents.append(
            {
                "id": f"charword/{voice_id}",
                "chunk_index": 0,
                "source_path": source_path,
                "clean_text": clean_text,
                "search_text": "\n".join(part for part in search_parts if part).strip(),
                "segments": [
                    {
                        "speaker": codename or None,
                        "text": voice_text,
                        "segment_type": "voice",
                    }
                ],
                "story_key": f"charword/{voice_id}",
                "story_id": f"charword/{voice_id}",
                "activity_id": "operator_voice",
                "activity_name": "干员语音",
                "story_name": voice_title or codename or char_id,
                "story_code": None,
                "avg_tag": "语音",
                "story_sort": int(payload.get("voiceIndex") or 0),
                "stage_id": None,
                "stage_code": None,
                "stage_name": None,
                "stage_type": None,
                "zone_id": None,
                "zone_name": None,
                "chapter_name": None,
                "trigger_type": "VOICE",
            }
        )
    return documents


def chunk_segments(
    segments: list[StorySegment],
    max_chars: int = 420,
    overlap_segments: int = 1,
) -> list[list[StorySegment]]:
    chunks: list[list[StorySegment]] = []
    current: list[StorySegment] = []
    current_len = 0

    def segment_len(segment: StorySegment) -> int:
        return len(render_segment(segment))

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append(current)
        if overlap_segments > 0:
            current = current[-overlap_segments:]
            current_len = sum(segment_len(item) for item in current)
        else:
            current = []
            current_len = 0

    for segment in segments:
        seg_len = segment_len(segment)
        if current and current_len + seg_len + 1 > max_chars:
            flush()
        current.append(segment)
        current_len += seg_len + 1

    if current:
        chunks.append(current)
    return chunks


def build_story_documents(
    story_root: Path,
    excel_root: Path,
    max_chars: int = 420,
    overlap_segments: int = 1,
) -> list[dict]:
    alias_lookup = build_operator_alias_lookup(excel_root)
    speaker_lookup = _build_story_speaker_lookup(excel_root)
    _resolve_story_meta.review_lookup = _build_story_review_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.review_stage_lookup = _build_story_review_lookup.stage_lookup  # type: ignore[attr-defined]
    _resolve_story_meta.review_activity_lookup = _build_story_review_lookup.activity_lookup  # type: ignore[attr-defined]
    _resolve_story_meta.stage_lookup = _build_stage_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.story_lookup = _build_story_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.zone_lookup = _build_zone_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.chapter_lookup = _build_chapter_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.activity_lookup = _build_activity_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.zone_to_activity_lookup = _build_zone_to_activity_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.roguelike_activity_lookup = _build_roguelike_activity_lookup(excel_root)  # type: ignore[attr-defined]

    documents: list[dict] = []
    for story_path in sorted(story_root.rglob("*.txt")):
        relative_story_key = story_path.relative_to(story_root).with_suffix("").as_posix()
        segments = parse_story_file(story_path, speaker_lookup=speaker_lookup)
        if not segments:
            continue

        story_meta = _resolve_story_meta(relative_story_key, excel_root)
        segment_chunks = chunk_segments(
            segments=segments,
            max_chars=max_chars,
            overlap_segments=overlap_segments,
        )
        for chunk_index, segment_chunk in enumerate(segment_chunks):
            text_lines = [render_segment(item) for item in segment_chunk]
            clean_text = "\n".join(text_lines).strip()
            if not clean_text:
                continue
            search_parts = [
                story_meta.get("activity_id") or "",
                story_meta.get("activity_name") or "",
                story_meta.get("story_name") or "",
                story_meta.get("story_code") or "",
                story_meta.get("stage_id") or "",
                story_meta.get("stage_code") or "",
                story_meta.get("stage_name") or "",
                story_meta.get("zone_id") or "",
                story_meta.get("chapter_name") or "",
                story_meta.get("zone_name") or "",
                story_meta.get("avg_tag") or "",
                clean_text,
            ]
            search_parts.extend(_collect_related_aliases("\n".join(search_parts), alias_lookup))
            search_text = "\n".join(part for part in search_parts if part).strip()
            document = {
                "id": f"{relative_story_key}#chunk-{chunk_index:04d}",
                "chunk_index": chunk_index,
                "source_path": story_path.as_posix(),
                "clean_text": clean_text,
                "search_text": search_text,
                "segments": [
                    {
                        "speaker": item.speaker,
                        "text": item.text,
                        "segment_type": item.segment_type,
                    }
                    for item in segment_chunk
                ],
            }
            document.update(story_meta)
            documents.append(document)
    return documents


def build_corpus_documents(
    story_root: Path,
    excel_root: Path,
    max_chars: int = 420,
    overlap_segments: int = 1,
) -> list[dict]:
    documents = build_story_documents(
        story_root=story_root,
        excel_root=excel_root,
        max_chars=max_chars,
        overlap_segments=overlap_segments,
    )
    documents.extend(build_handbook_documents(excel_root))
    documents.extend(build_charword_documents(excel_root))
    return documents


def dumps_jsonl(records: Iterable[dict]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
