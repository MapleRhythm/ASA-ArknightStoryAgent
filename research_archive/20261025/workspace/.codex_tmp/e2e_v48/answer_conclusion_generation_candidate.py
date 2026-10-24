"""Isolated trace-only candidate; production module remains untouched."""
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from answer_conclusion_generation_original import (
    build_conclusion_prompt,
    default_failed_conclusion,
    sample_grounded_conclusions,
    select_self_consistent_conclusion,
    should_try_final_direct_answer,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def record_final_direct_attempt(
    retrieval_trace: list[dict[str, Any]],
    primary_conclusions: list[ConclusionResult],
    *,
    conclusion: ConclusionResult | None = None,
    error: Exception | None = None,
) -> None:
    if not retrieval_trace:
        return
    fallback = (
        {"status": "error", "error": f"{type(error).__name__}: {error}"}
        if error is not None
        else {"status": "ok", "conclusion": asdict(conclusion)}
    )
    retrieval_trace[-1].setdefault("final_direct_fallback_attempts", []).append({
        "primary_conclusions": [asdict(item) for item in primary_conclusions],
        "primary_generation_failures": deepcopy(retrieval_trace[-1].get("generation_failures", [])),
        "fallback": fallback,
    })


def generate_conclusion_from_model(
    *,
    pipeline: Any,
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    current_round: int,
) -> ConclusionResult:
    prompt_evidence = pipeline.prepare_prompt_evidence(question, current_hypothesis, evidence)
    evidence_packing_trace: list[dict[str, Any]] = []
    prompt, evidence_prompt_text = build_conclusion_prompt(
        question,
        current_hypothesis,
        evidence,
        retrieval_trace,
        current_round,
        pipeline.max_retrieval_rounds,
        pipeline.prompt_evidence_top_k,
        prompt_evidence=prompt_evidence,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
        prompt_mode=pipeline.conclusion_prompt_mode,
        grounding_mode=pipeline.answer_grounding_mode,
        evidence_packing_trace=evidence_packing_trace,
    )
    if retrieval_trace:
        retrieval_trace[-1]["evidence_packing"] = evidence_packing_trace
    conclusions = sample_grounded_conclusions(
        pipeline=pipeline,
        question=question,
        current_hypothesis=current_hypothesis,
        prompt_evidence=prompt_evidence,
        prompt=prompt,
        evidence_prompt_text=evidence_prompt_text,
        current_round=current_round,
        retrieval_trace=retrieval_trace,
        evidence_packing_trace=evidence_packing_trace,
    )

    if not conclusions:
        if should_try_final_direct_answer(
            pipeline=pipeline,
            current_hypothesis=current_hypothesis,
            prompt_evidence=prompt_evidence,
            current_round=current_round,
        ):
            try:
                direct = pipeline.generate_direct_answer(question, current_hypothesis, prompt_evidence)
            except Exception as exc:
                record_final_direct_attempt(retrieval_trace, conclusions, error=exc)
                print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
            else:
                record_final_direct_attempt(retrieval_trace, conclusions, conclusion=direct)
                return direct
        return default_failed_conclusion(max_round_reached=current_round >= pipeline.max_retrieval_rounds)

    winning_conclusion = select_self_consistent_conclusion(conclusions)
    if (
        winning_conclusion.next_action in {"retrieve_more", "abstain"}
        and should_try_final_direct_answer(
            pipeline=pipeline,
            current_hypothesis=current_hypothesis,
            prompt_evidence=prompt_evidence,
            current_round=current_round,
        )
    ):
        try:
            direct = pipeline.generate_direct_answer(question, current_hypothesis, prompt_evidence)
        except Exception as exc:
            record_final_direct_attempt(retrieval_trace, conclusions, error=exc)
            print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
        else:
            record_final_direct_attempt(retrieval_trace, conclusions, conclusion=direct)
            return direct
    return winning_conclusion
