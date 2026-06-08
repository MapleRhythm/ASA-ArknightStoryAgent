from __future__ import annotations

from typing import TYPE_CHECKING, Any

from asa_arknight_story_agent.config import QueryConfig
from asa_arknight_story_agent.inference.evidence.preparation import EvidencePreparationMixin
from asa_arknight_story_agent.inference.generation_pipeline import GenerationPipelineMixin
from asa_arknight_story_agent.inference.pipeline.constants import (
    PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
)
from asa_arknight_story_agent.inference.pipeline.orchestration import PipelineOrchestrationMixin
from asa_arknight_story_agent.inference.model_runtime.runners import LlamaCppRunner, VllmRunner
from asa_arknight_story_agent.inference.retrieval.pipeline import RetrievalPipelineMixin
from asa_arknight_story_agent.inference.web_context.config import WebContextConfig, build_web_context_config

if TYPE_CHECKING:
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever


class CPUInferencePipeline(
    PipelineOrchestrationMixin,
    EvidencePreparationMixin,
    GenerationPipelineMixin,
    RetrievalPipelineMixin,
):
    def __init__(
        self,
        *,
        retriever: ArknightsHybridRetriever,
        generator: LlamaCppRunner | VllmRunner,
        query_config: QueryConfig | None = None,
        max_retrieval_rounds: int = 2,
        prompt_evidence_top_k: int = 8,
        prompt_evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
        prompt_conclusion_evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.72,
        enable_pyramid_order: bool = False,
        enable_crag_refinement: bool = False,
        crag_refine_top_sentences: int = 4,
        crag_refine_max_sentences: int = 24,
        self_consistency_samples: int = 1,
        self_consistency_temperature: float = 0.7,
        answer_grounding_mode: str = "weak",
        max_follow_up_rounds: int | None = None,
        use_model_hypothesis: bool = True,
        use_model_conclusion_generation: bool = True,
        use_model_retrieval_planner: bool | None = None,
        conclusion_prompt_mode: str = "full",
        enable_evidence_pinning: bool = False,
        web_context_config: dict[str, Any] | WebContextConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_config = query_config or QueryConfig()
        self.max_retrieval_rounds = min(2, max(1, int(max_retrieval_rounds)))
        if not use_model_hypothesis:
            raise ValueError("heuristic hypothesis generation is disabled; set use_model_hypothesis=true")
        if use_model_retrieval_planner is not None:
            use_model_conclusion_generation = use_model_retrieval_planner
        if not use_model_conclusion_generation:
            raise ValueError("heuristic conclusion generation is disabled; set use_model_conclusion_generation=true")
        self.use_model_hypothesis = use_model_hypothesis
        self.use_model_conclusion_generation = use_model_conclusion_generation
        self.conclusion_prompt_mode = conclusion_prompt_mode.strip().lower()
        if self.conclusion_prompt_mode not in {"full", "minimal"}:
            raise ValueError("conclusion_prompt_mode must be 'full' or 'minimal'")
        self.enable_evidence_pinning = enable_evidence_pinning
        self.prompt_evidence_top_k = max(1, prompt_evidence_top_k)
        self.prompt_evidence_max_chars_per_doc = max(120, prompt_evidence_max_chars_per_doc)
        self.prompt_conclusion_evidence_max_total_chars = max(
            self.prompt_evidence_max_chars_per_doc,
            prompt_conclusion_evidence_max_total_chars,
        )
        self.enable_mmr = enable_mmr
        self.mmr_lambda = min(1.0, max(0.0, mmr_lambda))
        self.enable_pyramid_order = enable_pyramid_order
        self.enable_crag_refinement = enable_crag_refinement
        self.crag_refine_top_sentences = max(1, crag_refine_top_sentences)
        self.crag_refine_max_sentences = max(self.crag_refine_top_sentences, crag_refine_max_sentences)
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.self_consistency_temperature = max(0.0, self_consistency_temperature)
        self.answer_grounding_mode = answer_grounding_mode.strip().lower()
        self.web_context_config = (
            web_context_config
            if isinstance(web_context_config, WebContextConfig)
            else build_web_context_config(web_context_config)
        )
