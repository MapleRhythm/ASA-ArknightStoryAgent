from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
from asa_arknight_story_agent.inference.evidence.rendering import render_short_evidence_brief
from asa_arknight_story_agent.inference.retrieval.minirag_prompt_hints import render_minirag_hints_for_prompt
from asa_arknight_story_agent.inference.pipeline.constants import (
    PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_ordering import select_prompt_evidence


def build_answer_prompt(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
    prompt_evidence: list[dict[str, Any]] | None = None,
    evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
) -> tuple[str, str]:
    """Build the grounded direct-answer prompt and its rendered evidence brief.

    The brief is returned so quote grounding validates against the exact text
    shown to the model.
    """
    selected_evidence = (
        prompt_evidence
        if prompt_evidence is not None
        else select_prompt_evidence(
            question,
            hypothesis,
            evidence,
            prompt_evidence_top_k=prompt_evidence_top_k,
        )
    )
    system_prompt = "你是《明日方舟》剧情问答系统的证据锚定回答模块。只输出指定 JSON。"
    evidence_brief = render_short_evidence_brief(
        selected_evidence,
        max_chars_per_doc=evidence_max_chars_per_doc,
        max_total_chars=evidence_max_total_chars,
    )
    minirag_hints = render_minirag_hints_for_prompt(selected_evidence, hypothesis)
    user_prompt = "\n".join(
        [
            "task: grounded_final_answer",
            f"question: {question}",
            "hypothesis: " + json.dumps(asdict(hypothesis), ensure_ascii=False),
            "evidence_brief:",
            evidence_brief,
            "minirag_hints: " + minirag_hints,
            "output_schema: grounded_action_v1",
            "action_set: answer_directly,abstain",
            'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}',
            'abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
            "rules: JSON only；只能使用 evidence_brief 中的证据；单条 quote 必须从 evidence_brief 原文精确复制，推荐20-60字，硬上限80字；每个 supported_fact 最多2条 quote 且总长<=160字；supported_facts最多6条，所有quote总长最好<=400字；final_answer 只能使用 supported_facts 和 inferred_facts；证据不足则 abstain；不要输出 current_round、confidence、decision、missing_slots、clarification_question。",
        ]
    )
    return render_qwen_chat_prompt(system_prompt, user_prompt), evidence_brief
