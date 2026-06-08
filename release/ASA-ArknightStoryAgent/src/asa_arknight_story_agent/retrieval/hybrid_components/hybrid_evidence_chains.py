from __future__ import annotations

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_building import (
    HybridEvidenceChainBuildingMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_roles import (
    HybridEvidenceChainRolesMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_chain_scoring import (
    HybridEvidenceChainScoringMixin,
)


class HybridEvidenceChainsMixin(
    HybridEvidenceChainScoringMixin,
    HybridEvidenceChainBuildingMixin,
    HybridEvidenceChainRolesMixin,
):
    """Evidence-chain retrieval reranking built from focused helper mixins."""
