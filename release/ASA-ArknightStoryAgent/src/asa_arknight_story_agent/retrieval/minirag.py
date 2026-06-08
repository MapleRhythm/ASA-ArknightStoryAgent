from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import MINIRAG_GRAPH_PATH
from asa_arknight_story_agent.retrieval.minirag_components.minirag_graph_builder import (
    build_and_save_minirag_graph,
    build_minirag_graph,
    collect_teacher_relations,
    source_name_candidates,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_scope import (
    CHAPTER_SCOPE_EXCLUDED_ACTIVITY_IDS,
    document_chapter_scope_key,
    document_chapter_scope_label,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_runtime import MiniRAGSearchMixin


# Compatibility aliases for historical imports from retrieval.minirag.
from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import (  # noqa: E402,F401
    ENTITY_RUN_RE,
    METADATA_ENTITY_PREFIXES,
    build_alias_lookup,
    extract_alias_entities,
    extract_generic_text_entities,
    is_generic_entity_candidate as _is_generic_entity_candidate,
    load_alias_map,
    load_jsonl,
    metadata_entities,
    relation_gate_terms as _relation_gate_terms,
)
_collect_teacher_relations = collect_teacher_relations
_source_name_candidates = source_name_candidates


@dataclass(slots=True)
class MiniRAGIndex(MiniRAGSearchMixin):
    entity_to_doc_indices: dict[str, list[int]]
    entity_to_doc_weights: dict[str, dict[int, float]]
    doc_to_entities: list[list[str]]
    doc_chapter_keys: list[str]
    chapter_doc_indices: dict[str, list[int]]
    alias_lookup: dict[str, str]
    generic_entities: set[str]
    teacher_relations: list[dict[str, str]]
    relation_adjacency: dict[str, list[dict[str, str]]]
    relation_evidence_doc_indices: dict[tuple[str, str, str, str], list[int]]

    @classmethod
    def load(cls, path: Path = MINIRAG_GRAPH_PATH) -> "MiniRAGIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        alias_map = payload.get("alias_map") if isinstance(payload.get("alias_map"), dict) else {}
        teacher_relations = [
            item
            for item in payload.get("teacher_relations", [])
            if isinstance(item, dict)
        ]
        relation_adjacency: dict[str, list[dict[str, str]]] = {}
        doc_id_to_index = {
            str(doc_id): int(index)
            for doc_id, index in (payload.get("doc_id_to_index") or {}).items()
            if str(doc_id).strip()
        }
        relation_evidence_doc_indices: dict[tuple[str, str, str, str], list[int]] = {}
        for relation in teacher_relations:
            head = str(relation.get("head") or "")
            tail = str(relation.get("tail") or "")
            relation_text = str(relation.get("relation") or "")
            source_name = str(relation.get("source_name") or "")
            if head:
                relation_adjacency.setdefault(head, []).append(relation)
            if tail:
                relation_adjacency.setdefault(tail, []).append(relation)
            key = (head, relation_text, tail, source_name)
            evidence_indices: list[int] = []
            for doc_id in str(relation.get("evidence_id") or "").split("|"):
                doc_id = doc_id.strip()
                if doc_id in doc_id_to_index:
                    evidence_indices.append(doc_id_to_index[doc_id])
            if evidence_indices:
                relation_evidence_doc_indices[key] = list(dict.fromkeys(evidence_indices))
        return cls(
            entity_to_doc_indices={
                str(key): [int(index) for index in value]
                for key, value in payload.get("entity_to_doc_indices", {}).items()
                if isinstance(value, list)
            },
            entity_to_doc_weights={
                str(entity): {
                    int(doc_index): float(weight)
                    for doc_index, weight in weights.items()
                }
                for entity, weights in payload.get("entity_to_doc_weights", {}).items()
                if isinstance(weights, dict)
            },
            doc_to_entities=[
                [str(entity) for entity in entities]
                for entities in payload.get("doc_to_entities", [])
                if isinstance(entities, list)
            ],
            doc_chapter_keys=[
                str(key)
                for key in payload.get("doc_chapter_keys", [])
            ]
            if isinstance(payload.get("doc_chapter_keys"), list)
            else [],
            chapter_doc_indices={
                str(chapter_key): [int(index) for index in indices]
                for chapter_key, indices in (payload.get("chapter_doc_indices") or {}).items()
                if isinstance(indices, list)
            },
            alias_lookup=build_alias_lookup(alias_map),
            generic_entities=set(str(key) for key in payload.get("entity_to_doc_indices", {})),
            teacher_relations=teacher_relations,
            relation_adjacency=relation_adjacency,
            relation_evidence_doc_indices=relation_evidence_doc_indices,
        )
