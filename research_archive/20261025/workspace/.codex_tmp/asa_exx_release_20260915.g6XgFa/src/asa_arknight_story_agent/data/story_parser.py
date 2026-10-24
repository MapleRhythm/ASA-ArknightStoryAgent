from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from asa_arknight_story_agent.data.operator_aliases import (
    build_operator_alias_lookup,
    collect_related_aliases,
)
from asa_arknight_story_agent.data.story_meta_resolver import StoryMetadataResolver
from asa_arknight_story_agent.data.story_metadata import (
    build_character_name_lookup,
    build_story_speaker_lookup,
    load_json,
)
from asa_arknight_story_agent.data.story_text import (
    chunk_segments,
    normalize_text,
    parse_story_file,
    render_segment,
)

_collect_related_aliases = collect_related_aliases


def build_handbook_documents(excel_root: Path) -> list[dict]:
    handbook_table = load_json(excel_root / "handbook_info_table.json")
    handbook_dict = handbook_table.get("handbookDict", {})
    character_name_lookup = build_character_name_lookup(excel_root)
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
    charword_table = load_json(excel_root / "charword_table.json")
    char_words = charword_table.get("charWords", {})
    character_name_lookup = build_character_name_lookup(excel_root)
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


def build_story_documents(
    story_root: Path,
    excel_root: Path,
    max_chars: int = 420,
    overlap_segments: int = 1,
) -> list[dict]:
    alias_lookup = build_operator_alias_lookup(excel_root)
    speaker_lookup = build_story_speaker_lookup(excel_root)
    metadata_resolver = StoryMetadataResolver.from_excel_root(excel_root)

    documents: list[dict] = []
    for story_path in sorted(story_root.rglob("*.txt")):
        relative_story_key = story_path.relative_to(story_root).with_suffix("").as_posix()
        segments = parse_story_file(story_path, speaker_lookup=speaker_lookup)
        if not segments:
            continue

        story_meta = metadata_resolver.resolve(relative_story_key)
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
