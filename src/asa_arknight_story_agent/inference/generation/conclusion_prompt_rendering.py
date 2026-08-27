from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
from asa_arknight_story_agent.inference.generation.conclusion_prompt_rules import CONCLUSION_OUTPUT_RULES
from asa_arknight_story_agent.inference.evidence.rendering import render_evidence_blocks
from asa_arknight_story_agent.inference.generation.history_rendering import (
    render_generation_history,
    render_retrieval_history,
)
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
from asa_arknight_story_agent.inference.pipeline.constants import (
    PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
    PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    PROMPT_GENERATION_HISTORY_MAX_CHARS,
    PROMPT_HISTORY_MAX_ROUNDS,
    PROMPT_RETRIEVAL_HISTORY_MAX_CHARS,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.prompt_ordering import select_prompt_evidence
from asa_arknight_story_agent.inference.planning.query_understanding import render_dialogue_context_for_prompt
from asa_arknight_story_agent.inference.common.text_utils import truncate_text


def build_conclusion_prompt(
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    current_round: int,
    max_retrieval_rounds: int,
    prompt_evidence_top_k: int,
    prompt_evidence: list[dict[str, Any]] | None = None,
    evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    prompt_mode: str = "full",
    grounding_mode: str = "quote",
) -> tuple[str, str]:
    """Build the conclusion prompt and the evidence text shown to the model.

    The second return value is the exact evidence text embedded in the prompt
    (the minimal brief, or the full rendered blocks), so quote grounding can
    validate against what the model actually saw.
    """
    rendered_dialogue_context = truncate_text(
        render_dialogue_context_for_prompt(current_hypothesis.dialogue_context),
        PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
    )
    rendered_evidence = render_evidence_blocks(
        prompt_evidence
        if prompt_evidence is not None
        else select_prompt_evidence(
            question,
            current_hypothesis,
            evidence,
            prompt_evidence_top_k=prompt_evidence_top_k,
        ),
        max_chars_per_doc=evidence_max_chars_per_doc,
        max_total_chars=evidence_max_total_chars,
    )
    # Evidence-id output uses the compact structured-facts schema.  Reuse that
    # schema even if a legacy deployment still requests the verbose prompt.
    if prompt_mode == "minimal" or grounding_mode.strip().lower() == "evidence_id":
        return build_minimal_conclusion_prompt(
            question=question,
            current_hypothesis=current_hypothesis,
            evidence=evidence,
            current_round=current_round,
            max_retrieval_rounds=max_retrieval_rounds,
            prompt_evidence_top_k=prompt_evidence_top_k,
            prompt_evidence=prompt_evidence,
            evidence_max_chars_per_doc=evidence_max_chars_per_doc,
            evidence_max_total_chars=evidence_max_total_chars,
            grounding_mode=grounding_mode,
        )
    system_prompt = "\n".join(
        [
            "你是《明日方舟》剧情问答系统中的 conclusion_generator。",
            "你的任务是基于当前证据生成阶段性结论，并判断是否还需要继续检索。",
            "不要输出思维过程。",
            "输出必须是单个 JSON 对象，不要使用 markdown 代码块。",
            "不要依赖系统做字段补全或兜底，字段缺失会直接视为失败。",
        ]
    )
    user_prompt = "\n".join(
        [
            "请根据以下信息生成当前阶段结论。",
            "",
            f"用户原问题: {question}",
            "多轮问答上下文:",
            rendered_dialogue_context,
            "",
            f"当前检索轮次: 第 {current_round} 轮 / 最多 {max_retrieval_rounds} 轮",
            "当前假设文档(JSON):",
            json.dumps(asdict(current_hypothesis), ensure_ascii=False, indent=2),
            "",
            "历史生成结果:",
            render_generation_history(
                retrieval_trace,
                max_rounds=PROMPT_HISTORY_MAX_ROUNDS,
                max_total_chars=PROMPT_GENERATION_HISTORY_MAX_CHARS,
            ),
            "",
            "历史检索上下文:",
            render_retrieval_history(
                retrieval_trace,
                max_rounds=PROMPT_HISTORY_MAX_ROUNDS,
                max_total_chars=PROMPT_RETRIEVAL_HISTORY_MAX_CHARS,
            ),
            "",
            "当前证据:",
            rendered_evidence,
            "",
            "输出要求:",
            *CONCLUSION_OUTPUT_RULES,
        ]
    )
    return render_qwen_chat_prompt(system_prompt, user_prompt), rendered_evidence
