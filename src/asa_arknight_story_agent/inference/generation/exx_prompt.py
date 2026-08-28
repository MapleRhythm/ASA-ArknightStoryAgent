from __future__ import annotations

import json
from typing import Any


EXX_PROTOCOL = "grounded_action_exx_v1"
EXX_SYSTEM_PROMPT = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
EXX_RULES = (
    "只使用当前可见证据；next_action只能是answer_directly、retrieve_more或abstain；"
    "answer_directly时只输出next_action和1至8条supported_facts，每个可核验原子事实的"
    "evidence_ids绑定1至2个当前存在的E编号；retrieve_more时输出follow_up_hypothesis；"
    "abstain时输出reason；"
    "不要复制引文，不要输出evidence_refs、quote、final_answer、answer或inferred_facts；"
    "只保留完整回答问题所必需的最少事实，通常1至4条，禁止重复或近义改写同一事实；"
    "当前证据足够时不要补检索；证据不足且未到最后一轮时才retrieve_more，最后一轮则abstain。"
)


def compact_hypothesis(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "{}"
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value or {}, ensure_ascii=False, separators=(",", ":"))


def render_exx_user_prompt(
    *,
    question: str,
    hypothesis: Any,
    round_value: str,
    evidence_text: str,
) -> str:
    """Render the one canonical Exx prompt shared by training and runtime."""
    return "\n".join(
        (
            "task: grounded_action_generation",
            f"question: {question}",
            f"hypothesis: {compact_hypothesis(hypothesis)}",
            f"round: {round_value}",
            "evidence:",
            evidence_text,
            f"output_schema: {EXX_PROTOCOL}",
            f"rules: {EXX_RULES}",
        )
    )
