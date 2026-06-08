from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import DOCUMENTS_PATH, MINIRAG_GRAPH_PATH, OPERATOR_ALIAS_MAP_PATH
from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import (
    build_alias_lookup,
    load_alias_map,
    load_jsonl,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_graph_documents import (
    collect_document_graph_entries,
    source_name_candidates,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_teacher_relations import collect_teacher_relations


def build_minirag_graph(
    documents: list[dict[str, Any]],
    alias_map: dict[str, list[str]],
    teacher_annotations: list[dict[str, Any]] | None = None,
    *,
    progress: bool = False,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    started = time.time()
    alias_lookup = build_alias_lookup(alias_map)

    if progress:
        print(
            f"[minirag-build] collect teacher relations annotations={len(teacher_annotations or [])}",
            file=sys.stderr,
            flush=True,
        )
    teacher_relations, teacher_cooccurrence, teacher_evidence_entities = collect_teacher_relations(teacher_annotations)
    if progress:
        print(
            f"[minirag-build] teacher_relations={len(teacher_relations)} "
            f"teacher_entities={len(teacher_cooccurrence)}",
            file=sys.stderr,
            flush=True,
        )

    graph_entries = collect_document_graph_entries(
        documents,
        alias_lookup=alias_lookup,
        teacher_cooccurrence=teacher_cooccurrence,
        teacher_evidence_entities=teacher_evidence_entities,
        progress=progress,
        progress_interval=progress_interval,
        started=started,
    )
    entity_to_doc_indices = graph_entries["entity_to_doc_indices"]

    return {
        "version": 4 if teacher_relations else 2,
        "documents_path": str(DOCUMENTS_PATH),
        "document_count": len(documents),
        "entity_count": len(entity_to_doc_indices),
        "alias_map": alias_map,
        **graph_entries,
        "doc_id_to_index": {
            str(document.get("id")): doc_index
            for doc_index, document in enumerate(documents)
            if str(document.get("id") or "").strip()
        },
        "teacher_relations": teacher_relations,
        "entity_cooccurrence": {
            head: sorted(tails) for head, tails in teacher_cooccurrence.items()
        },
    }


def build_and_save_minirag_graph(
    *,
    documents_path: Path = DOCUMENTS_PATH,
    alias_map_path: Path = OPERATOR_ALIAS_MAP_PATH,
    output_path: Path = MINIRAG_GRAPH_PATH,
    teacher_annotations_path: Path | list[Path] | None = None,
    progress: bool = False,
    progress_interval: int = 1000,
) -> dict[str, Any]:
    started = time.time()
    if progress:
        print(f"[minirag-build] load documents {documents_path}", file=sys.stderr, flush=True)
    documents = load_jsonl(documents_path)
    if progress:
        print(f"[minirag-build] documents={len(documents)}", file=sys.stderr, flush=True)
        print(f"[minirag-build] load aliases {alias_map_path}", file=sys.stderr, flush=True)
    alias_map = load_alias_map(alias_map_path)
    teacher_annotations: list[dict[str, Any]] | None = None
    if isinstance(teacher_annotations_path, list):
        teacher_annotations = []
        for path in teacher_annotations_path:
            if path.exists():
                if progress:
                    print(f"[minirag-build] load annotations {path}", file=sys.stderr, flush=True)
                teacher_annotations.extend(load_jsonl(path))
    elif teacher_annotations_path and teacher_annotations_path.exists():
        if progress:
            print(f"[minirag-build] load annotations {teacher_annotations_path}", file=sys.stderr, flush=True)
        teacher_annotations = load_jsonl(teacher_annotations_path)
    graph = build_minirag_graph(
        documents,
        alias_map,
        teacher_annotations=teacher_annotations,
        progress=progress,
        progress_interval=progress_interval,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"[minirag-build] write graph {output_path}", file=sys.stderr, flush=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
    if progress:
        print(
            f"[minirag-build] done elapsed={time.time() - started:.1f}s "
            f"documents={graph['document_count']} entities={graph['entity_count']} "
            f"teacher_relations={len(graph.get('teacher_relations', []))}",
            file=sys.stderr,
            flush=True,
        )
    return graph
