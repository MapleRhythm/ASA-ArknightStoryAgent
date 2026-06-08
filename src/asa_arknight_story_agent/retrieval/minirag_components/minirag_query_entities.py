from __future__ import annotations

from asa_arknight_story_agent.retrieval.minirag_components.minirag_entities import (
    ENTITY_RUN_RE,
    METADATA_ENTITY_PREFIXES,
    extract_alias_entities,
    is_generic_entity_candidate,
)


class MiniRAGQueryEntitiesMixin:
    def _query_entities(self, query: str) -> list[str]:
        query_entities = extract_alias_entities(query, self.alias_lookup)
        for match in ENTITY_RUN_RE.finditer(query):
            token = match.group(0).strip()
            if is_generic_entity_candidate(token) and token in self.generic_entities and token not in query_entities:
                query_entities.append(token)
            for entity in (f"activity:{token}", f"story:{token}", f"stage:{token}"):
                if entity in self.entity_to_doc_indices and entity not in query_entities:
                    query_entities.append(entity)
        if len(query_entities) < 3:
            for entity in sorted(self.generic_entities, key=len, reverse=True):
                if len(query_entities) >= 8:
                    break
                if entity.startswith(METADATA_ENTITY_PREFIXES):
                    continue
                if len(entity) < 3 and entity not in self.alias_lookup.values():
                    continue
                if entity in query and entity not in query_entities:
                    query_entities.append(entity)
        return query_entities
