"""One bounded correction on the same visible evidence; never a judge bypass."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from typing import Any

from asa_arknight_story_agent.inference.evidence.full_document_contract import validate_full_document_prompt
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
from asa_arknight_story_agent.inference.generation.exx_prompt import EXX_SYSTEM_PROMPT, render_exx_user_prompt
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def fact_key(fact: dict) -> tuple:
    return (
        str(fact.get("fact", "")).strip(),
        tuple(sorted(ref["evidence_id"] for ref in fact.get("evidence_refs", []))),
    )


def repair_evidence_id_conclusion(
    *,
    pipeline: Any,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict],
    evidence_prompt_text: str,
    conclusion: ConclusionResult,
    current_round: int,
    retrieval_trace: list[dict],
) -> ConclusionResult:
    diagnostics = conclusion.generation_diagnostics
    if (
        pipeline.answer_grounding_mode != "evidence_id"
        or diagnostics.get("grounding_failure_kind") != "lexical_support"
        or not diagnostics.get("grounding_candidate")
        or any(step.get("grounding_repair_attempts") for step in retrieval_trace)
    ):
        return conclusion

    candidate = diagnostics["grounding_candidate"]
    locked = [item["fact"] for item in diagnostics.get("grounding_partial_support", [])]
    compact = lambda facts: [
        {"fact": fact["fact"], "evidence_ids": [ref["evidence_id"] for ref in fact["evidence_refs"]]}
        for fact in facts
    ]
    # Feedback is data before the evidence section; it cannot add E-labels to
    # the visible evidence map. Keep source order and bodies byte-for-byte.
    feedback = "\n".join((
        "repair_task: 修复上一份候选答案，不进行新检索。",
        "候选答案不是证据，词面检查也不是语义裁判；只能根据下方原文修正。",
        "删除无必要的题目复述、主语修饰和套话，保留原文支持的核心事实。",
        "不要扩写动机，不得把假设、否定、转述或其他事件变成已发生的事实。",
        "不要因措辞告警直接改成证据不足。证据确实不足时才选择retrieve_more或abstain。",
        "必须逐字保留locked_facts中的事实及其E编号；不得通过删掉子问来假装修复成功。",
        "只输出grounded_action_exx_v1 JSON；不要输出quote或解释。",
        "candidate: " + json.dumps(compact(candidate), ensure_ascii=False),
        "locked_facts: " + json.dumps(compact(locked), ensure_ascii=False),
        "lexical_flags: " + json.dumps(diagnostics.get("grounding_issues", []), ensure_ascii=False),
    ))
    body = render_exx_user_prompt(
        question=question, hypothesis=asdict(hypothesis),
        round_value=f"{current_round}/{pipeline.max_retrieval_rounds}",
        evidence_text=evidence_prompt_text,
    )
    prompt = render_qwen_chat_prompt(EXX_SYSTEM_PROMPT, feedback + "\n" + body, assistant_prefix="")
    validate_full_document_prompt(pipeline, prompt, evidence)
    evidence_start = prompt.index("\nevidence:\n") + len("\nevidence:\n")
    actual_brief = prompt[evidence_start:evidence_start + len(evidence_prompt_text)]
    if (
        actual_brief != evidence_prompt_text
        or evidence_id_text_map_from_prompt(actual_brief)
        != evidence_id_text_map_from_prompt(evidence_prompt_text)
    ):
        raise ValueError("Repair changed the visible evidence IDs or text")
    record = {
        "status": "started", "same_evidence": True,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "original_raw_output": diagnostics.get("raw_output", ""),
        "original_issues": list(diagnostics.get("grounding_issues", [])),
    }
    if retrieval_trace:
        retrieval_trace[-1].setdefault("grounding_repair_attempts", []).append(record)
    try:
        raw = pipeline.generator.generate(
            prompt, max_tokens=pipeline.generator.max_tokens,
            temperature=0.0, top_p=1.0, repeat_penalty=pipeline.generator.repeat_penalty,
        )
        record["raw_output"] = raw
        record["generation"] = deepcopy(getattr(pipeline.generator, "last_generation_diagnostics", {}))
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Repair requires a strict JSON object")
        repaired = normalize_conclusion_payload(
            payload, question=question, dialogue_context=hypothesis.dialogue_context,
            current_intent=hypothesis.intent, current_hypothesis=hypothesis,
            max_round_reached=current_round >= pipeline.max_retrieval_rounds,
        )
        if repaired.next_action != "answer_directly":
            record["status"] = "no_answer_keep_original"
            return conclusion
        repaired_keys = {fact_key(fact) for fact in repaired.supported_facts}
        if not all(fact_key(fact) in repaired_keys for fact in locked):
            record["status"] = "dropped_locked_fact"
            return conclusion
        # A shorter candidate may omit a required subanswer. Do not call that
        # full success; leave the original candidate/partial facts available.
        if len(repaired.supported_facts) < len(candidate):
            record["status"] = "dropped_candidate_fact"
            return conclusion
        checked = validate_conclusion_grounding(
            question=question, hypothesis=hypothesis, evidence=evidence,
            conclusion=repaired, max_round_reached=current_round >= pipeline.max_retrieval_rounds,
            mode="evidence_id", evidence_prompt_text=evidence_prompt_text,
        )
        record["issues"] = checked.generation_diagnostics.get("grounding_issues", [])
        if checked.next_action != "answer_directly":
            record["status"] = "still_rejected_keep_original"
            return conclusion
        record["status"] = "accepted_after_same_evidence_repair"
        checked.generation_diagnostics.update({
            "raw_output": raw, "normalized_output": raw, "parse_status": "strict_json",
            "grounding_status": record["status"],
            "original_raw_output": diagnostics.get("raw_output", ""),
            "grounding_repair": deepcopy(record),
            "evidence_packing": diagnostics.get("evidence_packing", []),
        })
        return checked
    except Exception as exc:
        record.update(status="repair_error_keep_original", error=f"{type(exc).__name__}: {exc}")
        return conclusion
