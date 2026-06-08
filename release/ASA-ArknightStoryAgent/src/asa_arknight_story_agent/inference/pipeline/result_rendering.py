from __future__ import annotations

from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, InferenceResult


def build_retrieval_query_from_trace(retrieval_trace: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"[round {step['round']}]"
            + "\n"
            + "\n".join(step["queries"])
            for step in retrieval_trace
            if step.get("queries")
        ]
    )


def simplify_evidence_for_result(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    simplified_evidence = []
    for item in evidence:
        doc = item["document"]
        simplified_evidence.append(
            {
                "id": doc["id"],
                "activity_name": doc.get("activity_name"),
                "story_name": doc.get("story_name"),
                "stage_code": doc.get("stage_code"),
                "avg_tag": doc.get("avg_tag"),
                "source_path": doc.get("source_path"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "evidence_chain_model_score": item.get("evidence_chain_model_score"),
                "evidence_chain_roles": item.get("evidence_chain_roles"),
                "evidence_chain_text": item.get("evidence_chain_text"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "minirag_score": item.get("minirag_score"),
                "clean_text": doc["clean_text"],
            }
        )
    return simplified_evidence


def build_prompt_evidence_runtime_metadata(pipeline: Any) -> dict[str, Any]:
    return {
        "top_k": pipeline.prompt_evidence_top_k,
        "mmr_enabled": pipeline.enable_mmr,
        "mmr_lambda": pipeline.mmr_lambda,
        "pyramid_order_enabled": pipeline.enable_pyramid_order,
        "evidence_pinning_enabled": pipeline.enable_evidence_pinning,
        "moegirl_downweight_enabled": True,
        "near_duplicate_dedupe_enabled": True,
        "crag_refinement_enabled": pipeline.enable_crag_refinement,
        "crag_refine_top_sentences": pipeline.crag_refine_top_sentences,
        "crag_refine_max_sentences": pipeline.crag_refine_max_sentences,
        "web_context_enabled": pipeline.web_context_config.enabled,
        "web_context_max_pages": pipeline.web_context_config.max_pages,
        "web_context_max_total_chars": pipeline.web_context_config.max_total_chars,
        "minirag_chapter_isolation": pipeline.query_config.minirag_chapter_isolation,
        "minirag_auto_second_retrieval": pipeline.query_config.minirag_auto_second_retrieval,
        "minirag_scope_seed_top_k": pipeline.query_config.minirag_scope_seed_top_k,
        "minirag_expansion_query_top_k": pipeline.query_config.minirag_expansion_query_top_k,
        "scoped_chapter_search_enabled": pipeline.query_config.enable_scoped_chapter_search,
        "scoped_chapter_dense_top_k": pipeline.query_config.scoped_chapter_dense_top_k,
        "scoped_chapter_sparse_top_k": pipeline.query_config.scoped_chapter_sparse_top_k,
        "same_story_sweep_enabled": pipeline.query_config.enable_same_story_sweep,
        "same_story_sweep_max_seed_docs": pipeline.query_config.same_story_sweep_max_seed_docs,
        "same_story_sweep_max_docs_per_story": pipeline.query_config.same_story_sweep_max_docs_per_story,
    }


def build_inference_result(
    *,
    pipeline: Any,
    question: str,
    current_hypothesis: HypothesisDocument,
    state: PipelineRunState,
) -> InferenceResult:
    return InferenceResult(
        question=question,
        intent=current_hypothesis.intent,
        hypothesis=asdict(current_hypothesis),
        model_runtime={
            **pipeline.generator.describe_runtime(),
            "prompt_evidence_strategy": build_prompt_evidence_runtime_metadata(pipeline),
            "conclusion_self_consistency": {
                "samples": pipeline.self_consistency_samples,
                "temperature": pipeline.self_consistency_temperature,
            },
        },
        retrieval_query=build_retrieval_query_from_trace(state.retrieval_trace),
        retrieval_trace=state.retrieval_trace,
        evidence=simplify_evidence_for_result(state.evidence),
        answer=state.final_answer,
    )
