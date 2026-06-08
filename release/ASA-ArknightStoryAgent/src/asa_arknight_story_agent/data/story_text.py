from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


NAME_LINE_RE = re.compile(r'^\[name="([^"]+)"\]\s*(.*)$')
HEADER_RE = re.compile(r"^\[HEADER.*\]\s*(.+)$")
TAG_RE = re.compile(r"<[^>]+>")
DIALOG_HEAD_RE = re.compile(r'dialogHead="([^"]+)"', re.IGNORECASE)


@dataclass(slots=True)
class StorySegment:
    speaker: str | None
    text: str
    segment_type: str


def normalize_text(text: str) -> str:
    cleaned = TAG_RE.sub("", text)
    cleaned = cleaned.replace("\u3000", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def render_segment(segment: StorySegment) -> str:
    if segment.speaker:
        return f"{segment.speaker}：{segment.text}"
    return segment.text


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
