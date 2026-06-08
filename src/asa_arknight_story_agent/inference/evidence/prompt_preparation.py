from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.definitions.definition_evidence import raw_exact_definition_evidence
from asa_arknight_story_agent.inference.evidence.pinning import pin_anchor_evidence
from asa_arknight_story_agent.inference.evidence.texts import is_web_context_item
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_mmr import select_prompt_evidence_mmr
from asa_arknight_story_agent.inference.evidence.prompt_ordering import (
    apply_pyramid_evidence_order,
    merge_forced_prompt_evidence,
    select_prompt_evidence,
)
from asa_arknight_story_agent.inference.retrieval.merge import merge_evidence_keep_order


def merge_raw_definition_evidence(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not pipeline.enable_evidence_pinning:
        return evidence
    raw_exact_evidence = raw_exact_definition_evidence(
        question,
        hypothesis,
        limit=max(1, min(2, pipeline.prompt_evidence_top_k // 4 or 1)),
    )
    if not raw_exact_evidence:
        return evidence
    return merge_evidence_keep_order(
        raw_exact_evidence,
        evidence,
        limit=max(pipeline.query_config.reranker_candidate_top_k, pipeline.prompt_evidence_top_k * 2),
    )


def prepare_prompt_evidence(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    forced_evidence = [
        item
        for item in evidence
        if is_web_context_item(item) and pipeline.web_context_config.force_prompt_evidence
    ]
    if pipeline.enable_evidence_pinning:
        forced_evidence.extend(
            raw_exact_definition_evidence(
                question,
                hypothesis,
                limit=max(1, min(2, pipeline.prompt_evidence_top_k // 4 or 1)),
            )
        )
    if pipeline.enable_mmr:
        selected = select_prompt_evidence_mmr(
            evidence,
            prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
            lambda_mult=pipeline.mmr_lambda,
        )
    else:
        selected = select_prompt_evidence(
            question,
            hypothesis,
            evidence,
            prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
        )
    if pipeline.enable_evidence_pinning:
        selected = pin_anchor_evidence(
            question,
            hypothesis,
            evidence,
            selected,
            limit=pipeline.prompt_evidence_top_k,
        )
    if pipeline.enable_crag_refinement:
        selected = pipeline.refine_evidence_strips(question, hypothesis, selected)
    if pipeline.enable_pyramid_order:
        selected = apply_pyramid_evidence_order(selected)
        if pipeline.enable_evidence_pinning:
            selected = pin_anchor_evidence(
                question,
                hypothesis,
                selected,
                selected,
                limit=pipeline.prompt_evidence_top_k,
            )
    if forced_evidence:
        selected = merge_forced_prompt_evidence(
            forced_evidence,
            selected,
            limit=pipeline.prompt_evidence_top_k,
        )
    return selected
