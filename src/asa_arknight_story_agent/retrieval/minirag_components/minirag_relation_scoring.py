from __future__ import annotations

from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import relation_gate_terms


class MiniRAGRelationScoringMixin:
    def _add_relation_scores(
        self,
        scores: dict[int, float],
        query: str,
        query_entities: list[str],
        *,
        edge_weight: float = 1.5,
        endpoint_weight: float = 0.5,
        evidence_weight: float = 2.4,
        weak_evidence_weight: float = 0.45,
        allowed_doc_indices: set[int] | None = None,
    ) -> None:
        seen_relations: set[tuple[str, str, str, str]] = set()
        query_entity_set = set(query_entities)
        query_terms = relation_gate_terms(query)
        for entity in query_entities:
            for relation in self.relation_adjacency.get(entity, []):
                key = (
                    str(relation.get("head") or ""),
                    str(relation.get("relation") or ""),
                    str(relation.get("tail") or ""),
                    str(relation.get("source_name") or ""),
                )
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                if not self._relation_allowed(relation, allowed_doc_indices):
                    continue
                endpoints = [str(relation.get("head") or ""), str(relation.get("tail") or "")]
                matched_endpoints = sum(
                    1
                    for endpoint in endpoints
                    if endpoint in query_entity_set or endpoint in query or endpoint in entity
                )
                relation_text = str(relation.get("relation") or "")
                relation_terms = relation_gate_terms(" ".join([*endpoints, relation_text]))
                relation_overlap = bool(query_terms & relation_terms)
                endpoint_pair_match = matched_endpoints >= 2
                endpoint_phrase_match = any(
                    endpoint and len(endpoint) >= 3 and endpoint in query
                    for endpoint in endpoints
                )
                strong_relation_match = endpoint_pair_match or (
                    matched_endpoints >= 1 and (relation_overlap or endpoint_phrase_match)
                )
                relation_bonus = edge_weight * max(1, matched_endpoints)
                effective_evidence_weight = (
                    evidence_weight if strong_relation_match else weak_evidence_weight
                )
                evidence_doc_indices = self._filter_doc_indices(
                    self.relation_evidence_doc_indices.get(key, [])[:8],
                    allowed_doc_indices,
                )
                for rank, doc_index in enumerate(evidence_doc_indices):
                    scores[doc_index] = scores.get(doc_index, 0.0) + relation_bonus * effective_evidence_weight / (rank + 1)
                for endpoint in endpoints:
                    endpoint_docs = self._filter_doc_indices(
                        self.entity_to_doc_indices.get(endpoint, [])[:128],
                        allowed_doc_indices,
                    )
                    for rank, doc_index in enumerate(endpoint_docs):
                        scores[doc_index] = scores.get(doc_index, 0.0) + endpoint_weight / (rank + 1)
                if strong_relation_match:
                    for endpoint in endpoints:
                        endpoint_docs = self._filter_doc_indices(
                            self.entity_to_doc_indices.get(endpoint, [])[:32],
                            allowed_doc_indices,
                        )
                        for rank, doc_index in enumerate(endpoint_docs):
                            scores[doc_index] = scores.get(doc_index, 0.0) + relation_bonus / (rank + 1)

    def _relation_allowed(
        self,
        relation: dict[str, str],
        allowed_doc_indices: set[int] | None,
    ) -> bool:
        if allowed_doc_indices is None:
            return True
        key = (
            str(relation.get("head") or ""),
            str(relation.get("relation") or ""),
            str(relation.get("tail") or ""),
            str(relation.get("source_name") or ""),
        )
        evidence_indices = self.relation_evidence_doc_indices.get(key, [])
        return bool(evidence_indices and any(index in allowed_doc_indices for index in evidence_indices))

    def _filter_doc_indices(
        self,
        doc_indices: list[int],
        allowed_doc_indices: set[int] | None,
    ) -> list[int]:
        if allowed_doc_indices is None:
            return doc_indices
        return [doc_index for doc_index in doc_indices if doc_index in allowed_doc_indices]

    def _relation_neighbors(
        self,
        entity: str,
        *,
        allowed_doc_indices: set[int] | None = None,
    ) -> list[str]:
        neighbors: list[str] = []
        seen: set[str] = set()
        for relation in self.relation_adjacency.get(entity, []):
            if not self._relation_allowed(relation, allowed_doc_indices):
                continue
            for endpoint in (str(relation.get("head") or ""), str(relation.get("tail") or "")):
                if endpoint and endpoint != entity and endpoint not in seen:
                    seen.add(endpoint)
                    neighbors.append(endpoint)
        return neighbors
