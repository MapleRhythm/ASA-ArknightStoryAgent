from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.json_utils import extract_json_object, repair_json_like_output
from asa_arknight_story_agent.inference.model_runtime.output_cleaning import sanitize_generation_output
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.generation.direct_answer_prompt import build_answer_prompt


def generate_direct_answer_from_model(
    *,
    pipeline: Any,
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> ConclusionResult:
    prompt_evidence = pipeline.prepare_prompt_evidence(question, current_hypothesis, evidence)
    prompt, evidence_prompt_text = build_answer_prompt(
        question,
        current_hypothesis,
        evidence,
        prompt_evidence_top_k=pipeline.prompt_evidence_top_k,
        prompt_evidence=prompt_evidence,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
    )
    raw_output = pipeline.generator.generate(
        prompt,
        max_tokens=min(max(pipeline.generator.max_tokens, 1536), 2048),
        temperature=0.1,
        top_p=0.8,
        repeat_penalty=1.0,
    )
    if not raw_output.lstrip().startswith(("{", "<think>")):
        raw_output = "{" + raw_output
    raw_output = repair_json_like_output(raw_output)
    payload = extract_json_object(raw_output)
    if payload:
        conclusion = normalize_conclusion_payload(
            payload,
            question=question,
            dialogue_context=current_hypothesis.dialogue_context,
            current_intent=current_hypothesis.intent,
            current_hypothesis=current_hypothesis,
            max_round_reached=True,
        )
    else:
        answer = sanitize_generation_output(raw_output, prompt).strip()
        if not answer:
            answer = "现有检索证据不足以确认。"
        conclusion = ConclusionResult(
            next_action="answer_directly",
            answer=answer,
            missing_slots=[],
            clarification_question="",
            follow_up_hypothesis=None,
        )
    return validate_conclusion_grounding(
        question=question,
        hypothesis=current_hypothesis,
        evidence=prompt_evidence,
        conclusion=conclusion,
        max_round_reached=True,
        mode=pipeline.answer_grounding_mode,
        evidence_prompt_text=evidence_prompt_text,
    )
