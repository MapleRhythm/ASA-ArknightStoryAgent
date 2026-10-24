from __future__ import annotations

from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_document_meta import (
    HybridDocumentMetaMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_focus_adjustments import (
    HybridFocusAdjustmentsMixin,
)
from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_support_scores import (
    HybridSupportScoresMixin,
)


class HybridScoringMixin(
    HybridFocusAdjustmentsMixin,
    HybridSupportScoresMixin,
    HybridDocumentMetaMixin,
):
    """Document metadata, support scoring, and retrieval focus adjustments."""
