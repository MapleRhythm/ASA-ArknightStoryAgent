from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.retrieval.minirag_components.minirag_scope import document_chapter_scope_key


class MiniRAGSearchOrchestrationMixin:
    def search(
        self,
        query: str,
        documents: list[dict[str, Any]],
        *,
        top_k: int = 40,
        propagation_weight: float = 0.35,
        chapter_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        query_entities = self._query_entities(query)
        if not query_entities:
            return []

        allowed_doc_indices: set[int] | None = None
        if chapter_scope:
            if self.chapter_doc_indices:
                allowed_doc_indices = set(self.chapter_doc_indices.get(chapter_scope, []))
            else:
                allowed_doc_indices = {
                    index
                    for index, document in enumerate(documents)
                    if document_chapter_scope_key(document) == chapter_scope
                }
            if not allowed_doc_indices:
                return []

        scores: dict[int, float] = {}
        for entity in query_entities:
            direct_docs = self._filter_doc_indices(
                self.entity_to_doc_indices.get(entity, []),
                allowed_doc_indices,
            )
            for rank, doc_index in enumerate(direct_docs):
                edge_weight = self.entity_to_doc_weights.get(entity, {}).get(doc_index, 1.0)
                scores[doc_index] = scores.get(doc_index, 0.0) + min(edge_weight, 4.0) / (rank + 1)
        self._add_ppr_scores(
            scores,
            query_entities,
            propagation_weight=propagation_weight,
            allowed_doc_indices=allowed_doc_indices,
        )
        self._add_relation_scores(scores, query, query_entities, allowed_doc_indices=allowed_doc_indices)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {
                "doc_index": doc_index,
                "score": float(score),
                "minirag_score": float(score),
                "document": documents[doc_index],
            }
            for doc_index, score in ranked
            if 0 <= doc_index < len(documents)
        ]
