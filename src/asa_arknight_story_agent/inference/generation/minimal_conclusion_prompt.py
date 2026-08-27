from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
from asa_arknight_story_agent.inference.generation.exx_prompt import (
    EXX_SYSTEM_PROMPT,
    render_exx_user_prompt,
)
from asa_arknight_story_agent.inference.evidence.rendering import render_short_evidence_brief
from asa_arknight_story_agent.inference.retrieval.minirag_prompt_hints import render_minirag_hints_for_prompt
from asa_arknight_story_agent.inference.pipeline.constants import (
    PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_ordering import select_prompt_evidence


def build_minimal_conclusion_prompt(
    *,
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    current_round: int,
    max_retrieval_rounds: int,
    prompt_evidence_top_k: int,
    prompt_evidence: list[dict[str, Any]] | None = None,
    evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    grounding_mode: str = "quote",
) -> tuple[str, str]:
    """Build the minimal conclusion prompt and its rendered evidence brief.

    The brief is returned alongside the prompt so grounding checks can validate
    quotes against exactly the text the model saw, not the untruncated corpus.
    """
    selected_evidence = (
        prompt_evidence
        if prompt_evidence is not None
        else select_prompt_evidence(
            question,
            current_hypothesis,
            evidence,
            prompt_evidence_top_k=prompt_evidence_top_k,
        )
    )
    system_prompt = (
        EXX_SYSTEM_PROMPT
        if grounding_mode.strip().lower() == "evidence_id"
        else "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
    )
    evidence_brief = render_short_evidence_brief(
        selected_evidence,
        max_chars_per_doc=evidence_max_chars_per_doc,
        max_total_chars=evidence_max_total_chars,
        preserve_complete_evidence=grounding_mode.strip().lower() == "evidence_id",
        label_on_own_line=grounding_mode.strip().lower() == "evidence_id",
    )
    minirag_hints = render_minirag_hints_for_prompt(selected_evidence, current_hypothesis)
    evidence_id_only = grounding_mode.strip().lower() == "evidence_id"
    answer_schema = (
        'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_ids":["E1"]}]}'
        if evidence_id_only
        else 'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}'
    )
    grounding_rules = (
        "rules: 只使用当前可见证据；回答时把每个可核验原子事实绑定到1至2个当前存在的E编号；不要复制引文，不要输出evidence_refs、quote、final_answer、answer或inferred_facts；supported_facts为1至8条；证据不足且未到最后一轮才retrieve_more，最后一轮不足则abstain。"
        if evidence_id_only
        else "rules: JSON only；只能使用 evidence_brief 中的证据；每条证据以 [E编号] 开头；evidence_id 必须填该证据的编号（如 E1），不得编造不存在的编号；单条 quote 必须从该编号对应的证据原文复制，推荐20-60字，硬上限80字；每个 supported_fact 最多2条 quote 且总长<=160字；supported_facts最多6条，所有quote总长最好<=400字；final_answer 只能使用 supported_facts 和 inferred_facts；证据不足才 retrieve_more；不要输出 current_round、confidence、decision、missing_slots、clarification_question。"
    )
    if evidence_id_only:
        user_prompt = render_exx_user_prompt(
            question=question,
            hypothesis=asdict(current_hypothesis),
            round_value=f"{current_round}/{max_retrieval_rounds}",
            evidence_text=evidence_brief,
        )
        # LLaMA-Factory's qwen3_nothink template starts the supervised target
        # at the first ``{``.  Do not prefill it at runtime, otherwise the
        # adapter sees a different assistant prefix than it was trained on.
        return render_qwen_chat_prompt(system_prompt, user_prompt, assistant_prefix=""), evidence_brief

    user_prompt = "\n".join(
        [
            "task: conclusion_generation",
            f"question: {question}",
            "hypothesis: " + json.dumps(asdict(current_hypothesis), ensure_ascii=False),
            f"round: {current_round}/{max_retrieval_rounds}",
            "evidence_brief:",
            evidence_brief,
            "minirag_hints: " + minirag_hints,
            "output_schema: grounded_action_v1",
            "action_set: answer_directly,retrieve_more,abstain",
            answer_schema,
            'retrieve_more: {"next_action":"retrieve_more","follow_up_hypothesis":{"question":"","query_type":"","entities":[],"keywords":[],"expected_answer_type":"","dialogue_context":""}}',
            'abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
            'follow_up_hypothesis_fields: question,query_type,entities,keywords,expected_answer_type,dialogue_context',
            grounding_rules,
        ]
    )
    return render_qwen_chat_prompt(system_prompt, user_prompt), evidence_brief
