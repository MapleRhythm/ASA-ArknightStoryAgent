from __future__ import annotations

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_query_modes import (
    HybridQueryModesMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_query_terms import (
    HybridQueryTermsMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_terms import (
    HIGH_RERANK_QUERY_TYPES,
    LOW_RERANK_QUERY_TYPES,
)


class HybridQueryAnalysisMixin(HybridQueryTermsMixin, HybridQueryModesMixin):
    """Query classification and token extraction helpers for hybrid retrieval."""

    def _original_query_bonus_scale(self, query_mode: str | None) -> float:
        if query_mode in LOW_RERANK_QUERY_TYPES:
            return 0.45
        if query_mode in HIGH_RERANK_QUERY_TYPES:
            return 0.3
        return 0.6
