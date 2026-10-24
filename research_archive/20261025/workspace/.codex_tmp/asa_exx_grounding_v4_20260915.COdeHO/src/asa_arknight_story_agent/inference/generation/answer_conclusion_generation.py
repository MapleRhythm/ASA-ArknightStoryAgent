from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

from asa_arknight_story_agent.inference.generation.conclusion_generation import (
    default_failed_conclusion,
    normalize_minimal_conclusion_output,
    parse_conclusion_output,
    select_self_consistent_conclusion,
)
from asa_arknight_story_agent.inference.grounding.validation import (
    has_answerable_evidence,
    validate_conclusion_grounding,
)
from asa_arknight_story_agent.inference.payload.truncated_answer_recovery import (
    recover_truncated_grounded_answer,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
from asa_arknight_story_agent.inference.evidence.full_document_contract import validate_full_document_prompt
from asa_arknight_story_agent.inference.generation.evidence_id_repair import repair_evidence_id_conclusion
from asa_arknight_story_agent.inference.grounding.partial_support import preserve_partial_support


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
    validate_full_document_prompt(pipeline, prompt, prompt_evidence)
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
    if pipeline.answer_grounding_mode == "evidence_id":
        # Development trials showed repeated candidates and budget exhaustion,
        # not dependable corrections. Keep this bounded experiment opt-in for
        # library users; the locked official CLI does not enable it.
        if getattr(pipeline, "enable_experimental_grounding_repair", False):
            winning_conclusion = repair_evidence_id_conclusion(
                pipeline=pipeline, question=question, hypothesis=current_hypothesis,
                evidence=prompt_evidence, evidence_prompt_text=evidence_prompt_text,
                conclusion=winning_conclusion, current_round=current_round,
                retrieval_trace=retrieval_trace,
            )
        return preserve_partial_support(
            conclusion=winning_conclusion, question=question, evidence=prompt_evidence,
            evidence_prompt_text=evidence_prompt_text, retrieval_trace=retrieval_trace,
        )
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


def sample_grounded_conclusions(
    *,
    pipeline: Any,
    question: str,
    current_hypothesis: HypothesisDocument,
    prompt_evidence: list[dict[str, Any]],
    prompt: str,
    evidence_prompt_text: str | None = None,
    current_round: int,
    retrieval_trace: list[dict[str, Any]] | None = None,
    evidence_packing_trace: list[dict[str, Any]] | None = None,
) -> list[ConclusionResult]:
    conclusions: list[ConclusionResult] = []
    sample_count = pipeline.self_consistency_samples
    for sample_index in range(sample_count):
        raw_output = ""
        normalized_output = ""
        try:
            generation_max_tokens = (
                pipeline.generator.max_tokens
                if pipeline.answer_grounding_mode.strip().lower() == "evidence_id"
                else min(max(pipeline.generator.max_tokens, 1536), 2048)
            )
            raw_output = pipeline.generator.generate(
                prompt,
                max_tokens=generation_max_tokens,
                temperature=pipeline.self_consistency_temperature if sample_count > 1 else 0.1,
                top_p=0.9 if sample_count > 1 else 0.8,
                repeat_penalty=pipeline.generator.repeat_penalty,
            )
            normalized_output = normalize_minimal_conclusion_output(raw_output, pipeline.conclusion_prompt_mode)
            try:
                raw_payload = json.loads(raw_output.strip())
                parse_status = "strict_json" if isinstance(raw_payload, dict) else "non_object_json"
            except json.JSONDecodeError:
                parse_status = (
                    "assistant_prefix_repaired"
                    if normalized_output != raw_output
                    else "repaired_or_json_like"
                )
            conclusion = parse_conclusion_output(
                normalized_output,
                question=question,
                current_hypothesis=current_hypothesis,
                max_round_reached=current_round >= pipeline.max_retrieval_rounds,
            )
            conclusion = validate_conclusion_grounding(
                question=question,
                hypothesis=current_hypothesis,
                evidence=prompt_evidence,
                conclusion=conclusion,
                max_round_reached=current_round >= pipeline.max_retrieval_rounds,
                mode=pipeline.answer_grounding_mode,
                evidence_prompt_text=(
                    prompt if evidence_prompt_text is None else evidence_prompt_text
                ),
            )
            conclusion.generation_diagnostics = {
                **conclusion.generation_diagnostics,
                "raw_output": raw_output,
                "normalized_output": normalized_output,
                "parse_status": parse_status,
                "grounding_status": conclusion.generation_diagnostics.get(
                    "grounding_status",
                    "accepted" if conclusion.next_action == "answer_directly" else conclusion.next_action,
                ),
                "grounding_warnings": list(conclusion.grounding_warnings),
                "evidence_packing": evidence_packing_trace or [],
            }
            conclusions.append(conclusion)
        except Exception as exc:
            recovered = recover_truncated_grounded_answer(
                normalized_output or raw_output,
                question=question,
                max_round_reached=current_round >= pipeline.max_retrieval_rounds,
            )
            if recovered is not None:
                try:
                    recovered = validate_conclusion_grounding(
                        question=question,
                        hypothesis=current_hypothesis,
                        evidence=prompt_evidence,
                        conclusion=recovered,
                        max_round_reached=current_round >= pipeline.max_retrieval_rounds,
                        mode=pipeline.answer_grounding_mode,
                        evidence_prompt_text=(
                            prompt if evidence_prompt_text is None else evidence_prompt_text
                        ),
                    )
                    recovered.generation_diagnostics = {
                        **recovered.generation_diagnostics,
                        "raw_output": raw_output,
                        "normalized_output": normalized_output,
                        "parse_status": "recovered_truncated",
                        "initial_error": f"{type(exc).__name__}: {exc}",
                        "grounding_status": recovered.generation_diagnostics.get(
                            "grounding_status",
                            "accepted" if recovered.next_action == "answer_directly" else recovered.next_action,
                        ),
                        "grounding_warnings": list(recovered.grounding_warnings),
                        "evidence_packing": evidence_packing_trace or [],
                    }
                    conclusions.append(recovered)
                    continue
                except Exception:
                    pass
            if sample_count == 1 and not (
                current_round >= pipeline.max_retrieval_rounds
                and current_hypothesis.intent != "out_of_scope"
                and has_answerable_evidence(prompt_evidence)
            ):
                raise
            # Preserve a failed sample in the trace when other
            # self-consistency samples can still decide the action.
            if retrieval_trace:
                retrieval_trace[-1].setdefault("generation_failures", []).append(
                    {
                        "sample_index": sample_index,
                        "raw_output": raw_output,
                        "normalized_output": normalized_output,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
    return conclusions


def should_try_final_direct_answer(
    *,
    pipeline: Any,
    current_hypothesis: HypothesisDocument,
    prompt_evidence: list[dict[str, Any]],
    current_round: int,
) -> bool:
    return (
        pipeline.answer_grounding_mode != "evidence_id"
        and current_round >= pipeline.max_retrieval_rounds
        and current_hypothesis.intent != "out_of_scope"
        and has_answerable_evidence(prompt_evidence)
    )
