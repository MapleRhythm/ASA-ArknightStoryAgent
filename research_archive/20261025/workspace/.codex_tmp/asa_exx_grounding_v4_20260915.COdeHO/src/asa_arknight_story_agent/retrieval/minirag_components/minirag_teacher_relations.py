from __future__ import annotations

from typing import Any


def collect_teacher_relations(
    teacher_annotations: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, list[str]]]:
    """Aggregate teacher-extracted triples and evidence entity hints."""
    if not teacher_annotations:
        return [], {}, {}
    relations: list[dict[str, str]] = []
    cooccurrence: dict[str, set[str]] = {}
    evidence_to_entities: dict[str, list[str]] = {}
    for annotation in teacher_annotations:
        if not isinstance(annotation, dict):
            continue
        batch_id = str(annotation.get("batch_id") or "").strip()
        if batch_id:
            for relation_item in annotation.get("relations") or []:
                if not isinstance(relation_item, dict):
                    continue
                head = str(relation_item.get("head") or "").strip()
                relation = str(relation_item.get("relation") or "").strip()
                tail = str(relation_item.get("tail") or "").strip()
                if not head or not relation or not tail:
                    continue
                evidence_doc_ids = [
                    str(doc_id).strip()
                    for doc_id in relation_item.get("evidence_doc_ids") or []
                    if str(doc_id).strip()
                ]
                if not evidence_doc_ids:
                    evidence_doc_id = str(relation_item.get("doc_id") or "").strip()
                    if evidence_doc_id:
                        evidence_doc_ids = [evidence_doc_id]
                relations.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "evidence_id": "|".join(evidence_doc_ids),
                        "source_name": batch_id,
                    }
                )
                cooccurrence.setdefault(head, set()).add(tail)
                cooccurrence.setdefault(tail, set()).add(head)
                for doc_id in evidence_doc_ids:
                    evidence_to_entities.setdefault(f"doc_id::{doc_id}", []).extend([head, tail])
            continue
        doc_id = str(annotation.get("doc_id") or "").strip()
        if doc_id:
            entity_names: list[str] = []
            for entity in annotation.get("entities") or []:
                if isinstance(entity, dict):
                    name = str(entity.get("name") or "").strip()
                    if name:
                        entity_names.append(name)
                        for alias in entity.get("aliases") or []:
                            alias_text = str(alias or "").strip()
                            if alias_text:
                                entity_names.append(alias_text)
            for relation_item in annotation.get("relations") or []:
                if not isinstance(relation_item, dict):
                    continue
                head = str(relation_item.get("head") or "").strip()
                relation = str(relation_item.get("relation") or "").strip()
                tail = str(relation_item.get("tail") or "").strip()
                if not head or not relation or not tail:
                    continue
                relations.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "evidence_id": doc_id,
                        "source_name": doc_id,
                    }
                )
                cooccurrence.setdefault(head, set()).add(tail)
                cooccurrence.setdefault(tail, set()).add(head)
                entity_names.extend([head, tail])
            if entity_names:
                evidence_to_entities.setdefault(f"doc_id::{doc_id}", []).extend(entity_names)
            continue
        for triple in annotation.get("entity_relations") or []:
            if not isinstance(triple, dict):
                continue
            head = str(triple.get("head") or "").strip()
            relation = str(triple.get("relation") or "").strip()
            tail = str(triple.get("tail") or "").strip()
            if not head or not relation or not tail:
                continue
            relations.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "evidence_id": str(triple.get("evidence_id") or ""),
                    "source_name": str(annotation.get("source_name") or ""),
                }
            )
            cooccurrence.setdefault(head, set()).add(tail)
            cooccurrence.setdefault(tail, set()).add(head)
            source_name = str(annotation.get("source_name") or "")
            evidence_id = str(triple.get("evidence_id") or "")
            if source_name and evidence_id:
                evidence_to_entities.setdefault(f"{source_name}::{evidence_id}", []).extend([head, tail])
    return relations, cooccurrence, evidence_to_entities
