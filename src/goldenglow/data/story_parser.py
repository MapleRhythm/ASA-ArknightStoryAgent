from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NAME_LINE_RE = re.compile(r'^\[name="([^"]+)"\]\s*(.*)$')
HEADER_RE = re.compile(r"^\[HEADER.*\]\s*(.+)$")
TAG_RE = re.compile(r"<[^>]+>")


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


def render_segment(segment: StorySegment) -> str:
    if segment.speaker:
        return f"{segment.speaker}：{segment.text}"
    return segment.text


def parse_story_file(path: Path) -> list[StorySegment]:
    segments: list[StorySegment] = []
    pending_plain: list[str] = []

    def flush_pending_plain() -> None:
        if not pending_plain:
            return
        text = normalize_text(" ".join(pending_plain))
        pending_plain.clear()
        if text:
            _append_segment(StorySegment(speaker=None, text=text, segment_type="narration"))

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
            continue

        pending_plain.append(line)

    flush_pending_plain()
    return segments


def _build_story_review_lookup(excel_root: Path) -> dict[str, dict]:
    review_table = _load_json(excel_root / "story_review_table.json")
    lookup: dict[str, dict] = {}
    for group_id, group in review_table.items():
        for item in group.get("infoUnlockDatas", []):
            story_txt = item.get("storyTxt")
            if not story_txt:
                continue
            lookup[story_txt] = {
                "story_group": item.get("storyGroup") or group_id,
                "story_group_name": group.get("name"),
                "story_code": item.get("storyCode"),
                "story_name": item.get("storyName"),
                "avg_tag": item.get("avgTag"),
                "story_sort": item.get("storySort"),
                "stage_id": _extract_stage_id(item),
            }
    return lookup


def _build_stage_lookup(excel_root: Path) -> dict[str, dict]:
    stage_table = _load_json(excel_root / "stage_table.json")
    return stage_table.get("stages", {})


def _build_story_lookup(excel_root: Path) -> dict[str, dict]:
    return _load_json(excel_root / "story_table.json")


def _extract_stage_id(item: dict) -> str | None:
    required = item.get("requiredStages") or []
    if required:
        return required[0].get("stageId")
    return None


def _resolve_story_meta(relative_story_key: str, excel_root: Path) -> dict:
    review_lookup = _resolve_story_meta.review_lookup
    stage_lookup = _resolve_story_meta.stage_lookup
    story_lookup = _resolve_story_meta.story_lookup

    review_meta = review_lookup.get(relative_story_key, {})
    story_meta = story_lookup.get(relative_story_key, {})
    trigger = story_meta.get("trigger") or {}
    condition = story_meta.get("condition") or {}
    required_stages = condition.get("requiredStages") or []
    stage_id = (
        review_meta.get("stage_id")
        or trigger.get("key")
        or (required_stages[0].get("stageId") if required_stages else None)
    )
    stage_meta = stage_lookup.get(stage_id or "", {})

    story_path = Path(relative_story_key)
    activity_id = None
    if len(story_path.parts) > 1 and story_path.parts[0] == "activities":
        activity_id = story_path.parts[1]
    elif len(story_path.parts) > 1:
        activity_id = story_path.parts[0]

    return {
        "story_key": relative_story_key,
        "story_id": story_meta.get("id") or relative_story_key.replace("/", "_"),
        "activity_id": review_meta.get("story_group") or activity_id,
        "activity_name": review_meta.get("story_group_name"),
        "story_name": review_meta.get("story_name") or stage_meta.get("name"),
        "story_code": review_meta.get("story_code") or stage_meta.get("code"),
        "avg_tag": review_meta.get("avg_tag"),
        "story_sort": review_meta.get("story_sort"),
        "stage_id": stage_id,
        "stage_code": stage_meta.get("code"),
        "stage_name": stage_meta.get("name"),
        "stage_type": stage_meta.get("stageType"),
        "zone_id": stage_meta.get("zoneId"),
        "trigger_type": trigger.get("type"),
    }


_resolve_story_meta.review_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.stage_lookup = {}  # type: ignore[attr-defined]
_resolve_story_meta.story_lookup = {}  # type: ignore[attr-defined]


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
    _resolve_story_meta.review_lookup = _build_story_review_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.stage_lookup = _build_stage_lookup(excel_root)  # type: ignore[attr-defined]
    _resolve_story_meta.story_lookup = _build_story_lookup(excel_root)  # type: ignore[attr-defined]

    documents: list[dict] = []
    for story_path in sorted(story_root.rglob("*.txt")):
        relative_story_key = story_path.relative_to(story_root).with_suffix("").as_posix()
        segments = parse_story_file(story_path)
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
                story_meta.get("activity_name") or "",
                story_meta.get("story_name") or "",
                story_meta.get("stage_code") or "",
                story_meta.get("avg_tag") or "",
                clean_text,
            ]
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


def dumps_jsonl(records: Iterable[dict]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
