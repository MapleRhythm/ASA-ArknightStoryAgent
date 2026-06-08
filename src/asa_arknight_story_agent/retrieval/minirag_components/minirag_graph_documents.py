from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import (
    extract_alias_entities,
    extract_generic_text_entities,
    metadata_entities,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_scope import document_chapter_scope_key


def source_name_candidates(document: dict[str, Any]) -> list[str]:
    source_path = Path(str(document.get("source_path") or ""))
    names = []
    if source_path.name:
        names.append(source_path.name)
    story_id = str(document.get("story_id") or "")
    if story_id:
        names.append(Path(story_id).name + ".json")
    return names


def collect_document_graph_entries(
    documents: list[dict[str, Any]],
    *,
    alias_lookup: dict[str, str],
    teacher_cooccurrence: dict[str, set[str]],
    teacher_evidence_entities: dict[str, list[str]],
    progress: bool = False,
    progress_interval: int = 1000,
    started: float | None = None,
) -> dict[str, Any]:
    entity_to_doc_indices: dict[str, list[int]] = {}
    entity_to_doc_weights: dict[str, dict[str, float]] = {}
    doc_to_entities: list[list[str]] = []
    doc_chapter_keys: list[str] = []
    chapter_doc_indices: dict[str, list[int]] = {}

    extra_lookup = dict(alias_lookup)
    for entity in teacher_cooccurrence:
        if entity not in extra_lookup and len(entity) >= 2:
            extra_lookup[entity] = entity

    started = time.time() if started is None else started
    total_documents = len(documents)
    progress_interval = max(1, progress_interval)
    for doc_index, document in enumerate(documents):
        chapter_key = document_chapter_scope_key(document)
        doc_chapter_keys.append(chapter_key)
        if chapter_key:
            chapter_doc_indices.setdefault(chapter_key, []).append(doc_index)
        text = "\n".join(
            str(document.get(key) or "")
            for key in ("search_text", "clean_text", "activity_name", "story_name", "stage_code", "stage_name")
        )
        entities = []
        entities.extend(extract_alias_entities(text, extra_lookup))
        entities.extend(extract_generic_text_entities(text))
        for source_name in source_name_candidates(document):
            for evidence_id in (f"E{int(document.get('chunk_index', -1)) + 1}", str(document.get("chunk_index") or "")):
                entities.extend(teacher_evidence_entities.get(f"{source_name}::{evidence_id}", []))
        doc_id = str(document.get("id") or "")
        if doc_id:
            entities.extend(teacher_evidence_entities.get(f"doc_id::{doc_id}", []))
        entities.extend(metadata_entities(document))
        deduped = []
        seen = set()
        counts = Counter(entities)
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                deduped.append(entity)
                entity_to_doc_indices.setdefault(entity, []).append(doc_index)
                entity_to_doc_weights.setdefault(entity, {})[str(doc_index)] = float(counts[entity])
        doc_to_entities.append(deduped)
        if progress and (
            doc_index == 0
            or (doc_index + 1) % progress_interval == 0
            or doc_index + 1 == total_documents
        ):
            elapsed = time.time() - started
            done = doc_index + 1
            docs_per_second = done / max(elapsed, 1e-6)
            eta_seconds = (total_documents - done) / max(docs_per_second, 1e-6)
            print(
                f"[minirag-build] docs={done}/{total_documents} "
                f"entities={len(entity_to_doc_indices)} "
                f"elapsed={elapsed:.1f}s eta={eta_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    return {
        "entity_to_doc_indices": entity_to_doc_indices,
        "entity_to_doc_weights": entity_to_doc_weights,
        "doc_to_entities": doc_to_entities,
        "doc_chapter_keys": doc_chapter_keys,
        "chapter_doc_indices": chapter_doc_indices,
    }
