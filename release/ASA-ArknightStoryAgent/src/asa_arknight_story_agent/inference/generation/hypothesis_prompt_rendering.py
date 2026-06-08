from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
from asa_arknight_story_agent.inference.evidence.rendering import render_evidence_blocks
from asa_arknight_story_agent.inference.pipeline.constants import (
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_ordering import select_prompt_evidence
from asa_arknight_story_agent.inference.planning.query_understanding import render_dialogue_context_for_prompt
from asa_arknight_story_agent.inference.common.text_utils import truncate_text


def build_hypothesis_prompt(question: str, dialogue_context: str = "") -> str:
    rendered_dialogue_context = render_dialogue_context_for_prompt(dialogue_context)
    system_prompt = "你是《明日方舟》剧情检索系统的 hypothesis_builder。只输出 JSON。"
    user_prompt = "\n".join(
        [
            f"task: {INITIAL_HYPOTHESIS_TASK_TYPE}",
            f"question: {question}",
            f"dialogue_context: {rendered_dialogue_context}",
            "output_schema: hypothesis_v2",
            "fields: question,intent,query_type,entities,keywords,expected_answer_type,dialogue_context",
            "intent_set: character_relation,compare,event_summary,out_of_scope,persona_chat,plot_fact,plot_reasoning,timeline",
            "query_type_set: fact,relation,causality,reasoning,reveal,mystery,answerability",
            "rules: 输出合法 JSON；只写检索线索；entities/keywords 用短词；不要重复词；不要回答问题。",
        ]
    )
    return render_qwen_chat_prompt(system_prompt, user_prompt)


def build_follow_up_hypothesis_prompt(
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    unresolved_points: list[str],
    retrieval_trace: list[dict[str, Any]],
    previous_conclusion: ConclusionResult,
    current_round: int,
    max_retrieval_rounds: int,
    prompt_evidence_top_k: int,
    prompt_evidence: list[dict[str, Any]] | None = None,
) -> str:
    rendered_dialogue_context = truncate_text(
        render_dialogue_context_for_prompt(current_hypothesis.dialogue_context),
        PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
    )
    system_prompt = "你是《明日方舟》剧情检索系统的 follow_up_hypothesis_builder。只输出 JSON。"
    evidence_brief = render_evidence_blocks(
        prompt_evidence
        if prompt_evidence is not None
        else select_prompt_evidence(
            question,
            current_hypothesis,
            evidence,
            prompt_evidence_top_k=prompt_evidence_top_k,
        ),
        max_chars_per_doc=260,
        max_total_chars=1200,
    )
    user_prompt = "\n".join(
        [
            f"task: {FOLLOW_UP_HYPOTHESIS_TASK_TYPE}",
            f"question: {question}",
            f"dialogue_context: {rendered_dialogue_context}",
            f"round: {current_round}/{max_retrieval_rounds}",
            "hypothesis:",
            json.dumps(asdict(current_hypothesis), ensure_ascii=False),
            "missing_slots:",
            json.dumps(unresolved_points[:6], ensure_ascii=False),
            "evidence_brief:",
            evidence_brief,
            "previous_action:",
            previous_conclusion.next_action,
            "output_schema: follow_up_hypothesis_v2",
            "fields: question,query_type,entities,keywords,expected_answer_type,dialogue_context",
            "rules: 输出合法 JSON；只写下一轮检索线索；entities/keywords 用短词；不要重复词；不要回答问题。",
            "若问题属于阴谋/真相/识破/曝光类，且现有证据只有侧面线索，可结合你已有的《明日方舟》剧情知识补充可能相关的专名、地点、行动和结果作为检索关键词；这些内容只能用于检索，不得直接当作答案。",
            "若现有证据只确认到上位事件但缺少具体因果、行动链或时间线，也可结合你已有剧情知识补充相关专名、地点、别称和事件名作为检索关键词；这些内容只能用于检索，不得直接当作答案。",
        ]
    )
    return render_qwen_chat_prompt(system_prompt, user_prompt)
