from __future__ import annotations

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
    )
    conclusions = sample_grounded_conclusions(
        pipeline=pipeline,
        question=question,
        current_hypothesis=current_hypothesis,
        prompt_evidence=prompt_evidence,
        prompt=prompt,
        evidence_prompt_text=evidence_prompt_text,
        current_round=current_round,
    )

    if not conclusions:
        if should_try_final_direct_answer(
            pipeline=pipeline,
            current_hypothesis=current_hypothesis,
            prompt_evidence=prompt_evidence,
            current_round=current_round,
        ):
            try:
                return pipeline.generate_direct_answer(question, current_hypothesis, prompt_evidence)
            except Exception as exc:
                print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
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
            return pipeline.generate_direct_answer(question, current_hypothesis, prompt_evidence)
        except Exception as exc:
            print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
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
) -> list[ConclusionResult]:
    conclusions: list[ConclusionResult] = []
    sample_count = pipeline.self_consistency_samples
    for sample_index in range(sample_count):
        raw_output = ""
        normalized_output = ""
        try:
            generation_max_tokens = (
                min(pipeline.generator.max_tokens, 512)
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
                evidence_prompt_text=evidence_prompt_text or prompt,
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
                        evidence_prompt_text=evidence_prompt_text or prompt,
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
        current_round >= pipeline.max_retrieval_rounds
        and current_hypothesis.intent != "out_of_scope"
        and has_answerable_evidence(prompt_evidence)
    )
