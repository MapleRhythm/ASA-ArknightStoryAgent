from __future__ import annotations

from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import METADATA_ENTITY_PREFIXES


class MiniRAGPropagationMixin:
    @staticmethod
    def _entity_node_weight(entity: str) -> float:
        if entity.startswith(METADATA_ENTITY_PREFIXES):
            return 0.25
        return 1.0

    def _entity_doc_limit(self, entity: str) -> int:
        if entity.startswith(METADATA_ENTITY_PREFIXES):
            return 48
        return 128

    def _add_ppr_scores(
        self,
        scores: dict[int, float],
        query_entities: list[str],
        *,
        propagation_weight: float = 0.35,
        max_hops: int = 3,
        doc_entity_limit: int = 16,
        max_active_entities: int = 256,
        relation_traversal_weight: float = 1.4,
        min_entity_mass: float = 1e-4,
        allowed_doc_indices: set[int] | None = None,
    ) -> None:
        """Lightweight Personalized PageRank-style propagation on the hetero graph."""
        if not query_entities or max_hops <= 0 or propagation_weight <= 0:
            return

        query_entity_set = set(query_entities)
        entity_mass: dict[str, float] = {
            entity: self._entity_node_weight(entity)
            for entity in query_entities
            if entity in self.entity_to_doc_indices or entity in self.relation_adjacency
        }

        for hop in range(1, max_hops + 1):
            if not entity_mass:
                return
            hop_decay = propagation_weight ** hop
            doc_mass: dict[int, float] = {}

            for entity, mass in entity_mass.items():
                if mass < min_entity_mass:
                    continue
                node_weight = self._entity_node_weight(entity)
                docs = self._filter_doc_indices(
                    self.entity_to_doc_indices.get(entity, [])[: self._entity_doc_limit(entity)],
                    allowed_doc_indices,
                )
                for rank, doc_index in enumerate(docs):
                    edge_weight = min(self.entity_to_doc_weights.get(entity, {}).get(doc_index, 1.0), 4.0)
                    contribution = mass * node_weight * edge_weight / (rank + 1)
                    if contribution <= 0:
                        continue
                    doc_mass[doc_index] = doc_mass.get(doc_index, 0.0) + contribution
                    scores[doc_index] = scores.get(doc_index, 0.0) + contribution * hop_decay

            next_entity_mass: dict[str, float] = {}
            for doc_index, mass in doc_mass.items():
                if not (0 <= doc_index < len(self.doc_to_entities)):
                    continue
                entities = self.doc_to_entities[doc_index][:doc_entity_limit]
                if not entities:
                    continue
                share = mass / max(len(entities), 1)
                for offset, neighbor_entity in enumerate(entities, start=1):
                    if hop > 1 and neighbor_entity in query_entity_set:
                        continue
                    neighbor_weight = self._entity_node_weight(neighbor_entity)
                    next_entity_mass[neighbor_entity] = next_entity_mass.get(neighbor_entity, 0.0) + (
                        share * neighbor_weight / offset
                    )

            for entity, mass in entity_mass.items():
                for neighbor_entity in self._relation_neighbors(entity, allowed_doc_indices=allowed_doc_indices):
                    if hop > 1 and neighbor_entity in query_entity_set:
                        continue
                    next_entity_mass[neighbor_entity] = next_entity_mass.get(neighbor_entity, 0.0) + (
                        mass * relation_traversal_weight
                    )

            entity_mass = dict(
                sorted(
                    (
                        (entity, mass)
                        for entity, mass in next_entity_mass.items()
                        if mass >= min_entity_mass
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )[:max_active_entities]
            )
