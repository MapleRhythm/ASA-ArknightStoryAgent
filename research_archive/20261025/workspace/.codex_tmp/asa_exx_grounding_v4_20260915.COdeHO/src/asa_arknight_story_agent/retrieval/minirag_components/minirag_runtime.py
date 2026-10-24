from __future__ import annotations

from asa_arknight_story_agent.retrieval.minirag_components.minirag_propagation import (
    MiniRAGPropagationMixin,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_query_entities import (
    MiniRAGQueryEntitiesMixin,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_relation_scoring import (
    MiniRAGRelationScoringMixin,
)
from asa_arknight_story_agent.retrieval.minirag_components.minirag_search import (
    MiniRAGSearchOrchestrationMixin,
)


class MiniRAGSearchMixin(
    MiniRAGSearchOrchestrationMixin,
    MiniRAGPropagationMixin,
    MiniRAGRelationScoringMixin,
    MiniRAGQueryEntitiesMixin,
):
    """MiniRAG query execution assembled from focused helper mixins."""
