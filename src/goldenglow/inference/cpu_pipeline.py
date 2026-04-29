from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from goldenglow.config import QueryConfig
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


QUESTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CHINESE_TOKEN_SPLIT_RE = re.compile(r"[的是和与及或为在把被让给从向对将]")
COMMON_NON_ENTITY_WORDS = {
    "为什么",
    "为何",
    "怎么",
    "如何",
    "什么",
    "哪些",
    "哪里",
    "谁",
    "多少",
    "剧情",
    "故事",
    "角色",
    "明日方舟",
    "请问",
    "一下",
    "知道",
    "告诉我",
    "解释",
    "分析",
    "时候",
    "最后",
    "现在",
}
IDENTITY_HINT_WORDS = {
    "身份",
    "真实身份",
    "身世",
    "来历",
    "真相",
    "是谁",
    "谁",
    "父亲",
    "母亲",
    "亲生父亲",
    "后人",
    "关系",
}
STORY_HINT_WORDS = {
    "故事",
    "经历",
    "过往",
    "相遇",
    "后来",
    "往事",
    "渊源",
}
RELATION_TERMS = {
    "后人",
    "父亲",
    "母亲",
    "亲生父亲",
    "家人",
    "老师",
    "师父",
    "学生",
    "弟子",
}
TITLE_TERMS = {
    "太师",
    "真龙",
    "禁军",
    "大理寺",
}
LEGACY_INTENT_MAP = {
    "plot_explanation": "plot_reasoning",
    "plot_qa": "plot_fact",
    "follow_up": "plot_fact",
    "clarification_needed": "out_of_scope",
}
BRIDGE_STOP_WORDS = COMMON_NON_ENTITY_WORDS | {
    "身份",
    "真实身份",
    "身世",
    "来历",
    "真相",
    "后人",
    "父亲",
    "母亲",
    "亲生父亲",
    "名字",
    "秘密",
    "事情",
    "下场",
    "说法",
    "孩子",
}

LLAMA_TIMING_LINE_RE = re.compile(r"^\[\s*Prompt:.*\]$", re.MULTILINE)
INHERITANCE_RE = re.compile(r"([\u4e00-\u9fff]{2,8})的(后人|女儿|儿子|传人)")
KINSHIP_RE = re.compile(r"(亲生父亲|父亲|母亲|家人|老师|师父|弟子|学生)")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
DIALOGUE_ROLE_PREFIX_RE = re.compile(r"^(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
HYPOTHESIS_INTENTS = {
    "plot_fact",
    "plot_reasoning",
    "timeline",
    "character_relation",
    "event_summary",
    "compare",
    "persona_chat",
    "out_of_scope",
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"
INITIAL_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "intent",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
CONCLUSION_SCHEMA_FIELDS = (
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
)
ROLE_LABEL_MAP = {
    "user": "用户",
    "assistant": "助手",
}
PRONOUN_REFERENCES = {
    "她们",
    "他们",
    "它们",
    "她",
    "他",
    "它",
    "这位",
    "那位",
    "这个人",
    "那个人",
}
NOISY_RETRIEVAL_TOKENS = {
    "user",
    "assistant",
    "同伴关系",
    "身份关系",
    "事实问答",
    "综合剧情问答",
}
NOISY_TOKEN_MARKERS = (
    "什么",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "哪里",
    "哪儿",
    "是否",
    "有没有",
    "故事",
)
ENTITY_EXCLUDE_MARKERS = (
    "之间",
    "故事",
    "经历",
    "过往",
    "渊源",
    "关系",
)
PROMPT_DIALOGUE_CONTEXT_MAX_CHARS = 600
PROMPT_HISTORY_MAX_ROUNDS = 2
PROMPT_GENERATION_HISTORY_MAX_CHARS = 1200
PROMPT_RETRIEVAL_HISTORY_MAX_CHARS = 1200
PROMPT_FOLLOW_UP_EVIDENCE_MAX_TOTAL_CHARS = 2600
PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS = 3200
PROMPT_EVIDENCE_MAX_CHARS_PER_DOC = 520


@dataclass(slots=True)
class HypothesisDocument:
    question: str
    intent: str
    entities: list[str]
    keywords: list[str]
    expected_answer_type: str
    dialogue_context: str


@dataclass(slots=True)
class InferenceResult:
    question: str
    intent: str
    hypothesis: dict[str, Any]
    model_runtime: dict[str, Any]
    retrieval_query: str
    retrieval_trace: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str


@dataclass(slots=True)
class ConclusionResult:
    next_action: str
    answer: str
    missing_slots: list[str]
    clarification_question: str


class ModelOutputError(RuntimeError):
    pass


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def _truncate_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def _extract_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in QUESTION_TOKEN_RE.findall(text):
        parts = [raw_token] if raw_token.isascii() else [part for part in CHINESE_TOKEN_SPLIT_RE.split(raw_token) if part]
        for part in parts:
            normalized = part.strip()
            if (
                not normalized
                or normalized in COMMON_NON_ENTITY_WORDS
                or normalized in NOISY_RETRIEVAL_TOKENS
                or normalized in PRONOUN_REFERENCES
                or len(normalized) == 1 and not normalized.isascii()
                or any(marker in normalized for marker in NOISY_TOKEN_MARKERS)
                or normalized.endswith("吗")
            ):
                continue
            tokens.append(normalized)
    return _dedupe_keep_order(tokens)


def _parse_dialogue_context(dialogue_context: str) -> list[tuple[str | None, str]]:
    entries: list[tuple[str | None, str]] = []
    for raw_line in dialogue_context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        role_match = DIALOGUE_ROLE_PREFIX_RE.match(line)
        if role_match:
            role = role_match.group(1).lower()
            content = role_match.group(2).strip()
            if content:
                entries.append((role, content))
            continue
        entries.append((None, line))
    return entries


def _sanitize_dialogue_context(dialogue_context: str, *, for_prompt: bool = False) -> str:
    rendered_lines: list[str] = []
    for role, content in _parse_dialogue_context(dialogue_context):
        if not content:
            continue
        if for_prompt and role in ROLE_LABEL_MAP:
            rendered_lines.append(f"{ROLE_LABEL_MAP[role]}: {content}")
        else:
            rendered_lines.append(content)
    return "\n".join(rendered_lines).strip()


def _is_entity_candidate(token: str) -> bool:
    if (
        not token
        or token in STORY_HINT_WORDS
        or token in NOISY_RETRIEVAL_TOKENS
        or token in PRONOUN_REFERENCES
        or any(marker in token for marker in ENTITY_EXCLUDE_MARKERS)
    ):
        return False
    return True


def _extract_context_entities(dialogue_context: str) -> list[str]:
    parsed_entries = _parse_dialogue_context(dialogue_context)
    prioritized_texts = [content for role, content in parsed_entries if role == "user"]
    prioritized_texts.extend(content for role, content in parsed_entries if role == "assistant")
    prioritized_texts.extend(content for role, content in parsed_entries if role is None)

    entities: list[str] = []
    for content in prioritized_texts[-4:]:
        entities.extend(token for token in _extract_content_tokens(content) if _is_entity_candidate(token))
    return _dedupe_keep_order(entities)[:6]


def render_dialogue_context_for_prompt(dialogue_context: str) -> str:
    normalized = _sanitize_dialogue_context(dialogue_context, for_prompt=True)
    if not normalized:
        return "无"
    return normalized


def _resolve_referential_question(question: str, entities: list[str]) -> str:
    normalized_question = question.strip()
    if not normalized_question or not entities:
        return normalized_question
    anchor = "和".join(entities[:2]) if len(entities) >= 2 else entities[0]
    resolved = normalized_question
    for pronoun in sorted(PRONOUN_REFERENCES, key=len, reverse=True):
        if pronoun in resolved:
            resolved = resolved.replace(pronoun, anchor, 1)
            break
    return resolved


def detect_intent(question: str) -> tuple[str, str]:
    if any(token in question for token in STORY_HINT_WORDS):
        return "event_summary", "共同经历"
    if any(token in question for token in ("关系", "什么关系", "关联")):
        return "character_relation", "身份关系"
    if any(token in question for token in ("时间线", "先后", "之前", "之后", "何时", "什么时候")):
        return "timeline", "时间线"
    if any(token in question for token in ("对比", "区别", "不同", "相比")):
        return "compare", "对比分析"
    if any(token in question for token in ("总结", "概括", "发生了什么", "讲了什么")):
        return "event_summary", "剧情总结"
    if any(token in question for token in ("为什么", "为何", "原因", "动机", "目的")):
        return "plot_reasoning", "原因/动机"
    if any(token in question for token in ("怎么", "如何", "经过", "发生了什么", "流程")):
        return "plot_reasoning", "过程解释"
    if any(token in question for token in ("谁", "哪里", "哪儿", "何时", "什么时候", "什么", "是否", "有没有")):
        return "plot_fact", "事实问答"
    return "plot_fact", "综合剧情问答"


def extract_entities(question: str, dialogue_context: str = "") -> list[str]:
    question_entities = [token for token in _extract_content_tokens(question) if _is_entity_candidate(token)]
    if any(pronoun in question for pronoun in PRONOUN_REFERENCES):
        return _dedupe_keep_order(_extract_context_entities(dialogue_context) + question_entities)[:12]
    return _dedupe_keep_order(_extract_context_entities(dialogue_context) + question_entities)[:12]


def build_hypothesis(question: str, dialogue_context: str = "") -> HypothesisDocument:
    intent, answer_type = detect_intent(question)
    entities = extract_entities(question, dialogue_context)
    sanitized_context = _sanitize_dialogue_context(dialogue_context)
    question_keywords = _extract_content_tokens(question)
    context_entities = _extract_context_entities(dialogue_context)

    keywords = _dedupe_keep_order(
        context_entities
        + entities
        + question_keywords
    )[:16]

    if any(token in question for token in STORY_HINT_WORDS):
        story_keywords = ["共同经历", "相遇", "同行", "冲突", "过往"]
        keywords = _dedupe_keep_order(keywords + story_keywords)[:20]

    return HypothesisDocument(
        question=question,
        intent=intent,
        entities=entities,
        keywords=keywords,
        expected_answer_type=answer_type,
        dialogue_context=sanitized_context,
    )


def build_hypothesis_prompt(question: str, dialogue_context: str = "") -> str:
    rendered_dialogue_context = render_dialogue_context_for_prompt(dialogue_context)
    system_prompt = "\n".join(
        [
            "你是《明日方舟》剧情问答系统中的 hypothesis_builder。",
            "你的唯一任务是把用户问题改写成服务检索的假设文档。",
            "不要回答问题，不要解释，不要输出思维过程。",
            "输出必须是单个 JSON 对象，不要使用 markdown 代码块。",
            "如果信息不确定，可以保守填写，但字段必须完整。",
            "优先补足可检索线索：角色、别名、组织、地点、章节、活动、关系词、时间线词。",
        ]
    )
    user_prompt = "\n".join(
        [
            "请根据下面的输入生成假设文档 JSON。",
            "",
            f"用户问题: {question}",
            "多轮问答上下文:",
            rendered_dialogue_context,
            "",
            "字段要求:",
            '- "question": 原问题字符串',
            '- "intent": 从以下集合中选择一个: '
            + ", ".join(sorted(HYPOTHESIS_INTENTS)),
            '- "entities": 角色名/组织/地点/事件等实体数组',
            '- "keywords": 用于检索的关键词数组，优先保留可能引出证据的词',
            '- "expected_answer_type": 例如 事实问答 / 原因动机 / 身份关系 / 时间线 / 过程解释',
            '- "dialogue_context": 原样保留上下文；没有多轮上下文时可省略，系统按空字符串处理',
            "",
            "生成要求:",
            "1. 如果问题是追问，要根据上下文补全代词指向。",
            "1.1 如果问题里出现“她/他/她们/他们/这位/那位”等代词或指代说法，必须结合上下文替换为完整角色名。",
            "1.2 如果上下文已经出现明确人物名，entities 至少要包含这些人物名，不要只保留代词。",
            "2. 如果问题涉及身份、父母、后人、来历、真相，要主动补充关系线索与上位称谓。",
            "3. 如果问题是在追问“她们/他们之间有什么故事、经历、过往”，要把上下文中的人物名补全到 entities，并在 keywords 中加入“共同经历、相遇、同行、冲突、过往”等可检索短语。",
            "4. keywords 应该比 entities 更宽一些，包含同义改写和检索扩展短语。",
            "5. 不要虚构最终答案，只输出有助于召回的猜测性线索。",
            "6. 输出必须是合法 JSON，且字段严格限制为 question、intent、entities、keywords、expected_answer_type、dialogue_context。",
            "7. 你的输出第一字符必须是 {",
            "8. entities 和 keywords 中禁止出现 她/他/她们/他们/这位/那位/user/assistant 等代词或对话标签，必须写完整名字。",
            "9. entities 里禁止出现问句残片或描述性短语，例如“她们之间有什么故”“什么要启动”“么关系”“事吗”这类都不能算实体。",
            "10. keywords 里禁止出现无信息量碎片；优先输出完整人物名、称谓、关系词、事件词。",
            "",
            "反例:",
            '上下文: "闪灵和夜莺的关系 / 她们是同伴关系"；问题: "她们之间有什么故事吗"',
            '错误 entities: ["她们", "同伴关系", "她们之间有什么故"]',
            '正确 entities: ["闪灵", "夜莺"]',
            '正确 keywords: ["闪灵", "夜莺", "共同经历", "相遇", "同行", "冲突", "过往"]',
        ]
    )
    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n{"
    )


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
) -> str:
    rendered_dialogue_context = _truncate_text(
        render_dialogue_context_for_prompt(current_hypothesis.dialogue_context),
        PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
    )
    system_prompt = "\n".join(
        [
            "你是《明日方舟》剧情问答系统中的 follow_up_hypothesis_builder。",
            "你的任务是在当前检索轮次后，根据现有证据重写一份更强的补充检索假设文档。",
            "不要回答用户问题，不要解释，不要输出思维过程。",
            "输出必须是单个 JSON 对象，不要使用 markdown 代码块。",
            "新的假设文档必须服务于下一轮检索，优先补充桥接实体、关系线索、上位称谓、可能章节活动。",
            "只能使用原问题、当前假设文档和证据中能支持的线索，不要凭空捏造剧情结论。",
        ]
    )
    user_prompt = "\n".join(
        [
            "请基于以下信息，生成新的补充检索 hypothesis JSON。",
            "",
            f"用户原问题: {question}",
            "多轮问答上下文:",
            rendered_dialogue_context,
            "",
            f"当前检索轮次: 第 {current_round} 轮 / 最多 {max_retrieval_rounds} 轮",
            "当前假设文档(JSON):",
            json.dumps(asdict(current_hypothesis), ensure_ascii=False, indent=2),
            "",
            "上一轮结论生成结果(JSON):",
            json.dumps(asdict(previous_conclusion), ensure_ascii=False, indent=2),
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
            render_evidence_blocks(
                evidence[:prompt_evidence_top_k],
                max_chars_per_doc=PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
                max_total_chars=PROMPT_FOLLOW_UP_EVIDENCE_MAX_TOTAL_CHARS,
            ),
            "",
            "当前未解点:",
            "\n".join(f"- {point}" for point in unresolved_points) if unresolved_points else "- 缺少可以直接回答问题的桥接关系",
            "",
            "字段要求:",
            '- "question": 原问题字符串',
            f'- `intent` 不需要重新生成，固定继承上一轮: "{current_hypothesis.intent}"',
            '- "entities": 应包含原锚点实体，以及证据中出现的关键桥接对象；最多 6 项，每项尽量不超过 8 个字，不要把完整查询句塞进 entities',
            '- "keywords": 第二轮召回关键词；最多 12 项，每项尽量不超过 12 个字，优先短语，不要输出冗长自然语言句子',
            '- "expected_answer_type": 例如 身份关系 / 原因动机 / 时间线 / 过程解释',
            '- "dialogue_context": 原样保留当前上下文',
            "",
            "生成要求:",
            "0. 如果当前问题或历史上下文里出现“她/他/她们/他们/这位/那位”等代词，必须回填成完整角色名后再写入 entities 和 keywords。",
            "1. 如果首轮证据出现了关键称谓或关系词，必须把它们吸收到 entities 或 keywords 中。",
            "2. 如果问题是身份/身世类，优先构造“人物 + 称谓 + 关系”这种二次检索线索。",
            "3. 如果当前证据还缺关键桥接对象，可以保守提出 1 到 3 个高价值候选词。",
            "4. 优先缩小缺口，不要重复首轮已经失败的宽泛查询词。",
            "5. 不要把剧情经历、地点、会面对象当作“身份答案”本身；补充检索要围绕身份标签、别名、关系锚点展开。",
            "6. 对于人物关系追问，优先输出短关键词，如“共同经历”“并肩作战”“约定”，不要扩写成整句。",
            "7. 不要把最终答案写死在 JSON 中，仍然以检索友好为目标。",
            "8. 输出必须是合法 JSON，且字段严格限制为 question、entities、keywords、expected_answer_type、dialogue_context。",
            "9. 你的输出第一字符必须是 {，最后一个字符必须是 }。",
            "10. entities 和 keywords 中禁止出现代词、空泛指称和对话标签，必须优先写完整角色名或明确称谓。",
        ]
    )
    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n{"
    )


def build_conclusion_prompt(
    question: str,
    current_hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    current_round: int,
    max_retrieval_rounds: int,
    prompt_evidence_top_k: int,
) -> str:
    rendered_dialogue_context = _truncate_text(
        render_dialogue_context_for_prompt(current_hypothesis.dialogue_context),
        PROMPT_DIALOGUE_CONTEXT_MAX_CHARS,
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
            render_evidence_blocks(
                evidence[:prompt_evidence_top_k],
                max_chars_per_doc=PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
                max_total_chars=PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
            ),
            "",
            "输出要求:",
            '1. 必须输出 JSON，字段严格包含 "question"、"next_action"、"answer"、"missing_slots"、"clarification_question"。',
            '2. next_action 只能是 "answer_directly"、"retrieve_more"、"clarify_user"、"abstain"。',
            '3. 当 next_action = "answer_directly" 或 "abstain" 时，answer 必须非空。',
            '4. 当 next_action = "clarify_user" 时，clarification_question 必须非空，answer 可为空。',
            '5. 当 next_action = "retrieve_more" 时，answer 必须为空字符串，missing_slots 应给出具体可检索缺口。',
            "6. 如果现有证据已经足够，请选择 answer_directly，不要为了流程强行继续检索。",
            "7. 如果问题本身歧义很大，请选择 clarify_user。",
            "8. 如果继续检索仍然缺乏明确方向，或已接近轮次上限，可选择 abstain。",
            "9. 如果问题属于“是谁 / 什么身份 / 来历 / 真相 / 关系”类，answer_directly 的第一句必须先直接回答核心判断，优先使用“X是Y”或“现有证据不足以确认X是谁/身份是什么”格式。",
            "10. 如果证据只支持侧面经历、行动轨迹、见闻或计划，不能把这些内容当作“是谁”的答案；此时应选择 retrieve_more 或 abstain。",
            "11. 禁止用“他去过哪里、见过谁、做过什么”替代核心身份判断。",
            "12. 若答案只能确认到部分事实，也要先明确标注“已确认部分 / 无法确认部分”，不要把推测包装成确定事实。",
            "13. 你的输出第一字符必须是 {",
        ]
    )
    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n{"
    )


def build_retrieval_query(hypothesis: HypothesisDocument) -> str:
    resolved_question = _resolve_referential_question(hypothesis.question, hypothesis.entities)
    lines = [resolved_question]
    if hypothesis.dialogue_context:
        lines.append(f"上下文: {render_dialogue_context_for_prompt(hypothesis.dialogue_context)}")
    if hypothesis.entities:
        lines.append("实体: " + " ".join(hypothesis.entities))
    if hypothesis.keywords:
        lines.append("关键词: " + " ".join(hypothesis.keywords[:10]))
    if hypothesis.expected_answer_type:
        lines.append(f"回答类型: {hypothesis.expected_answer_type}")
    return "\n".join(lines)


def extract_bridge_terms(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> list[str]:
    counts: dict[str, int] = {}
    known_terms = set(hypothesis.entities) | set(hypothesis.keywords) | set(extract_entities(question))

    for item in evidence[:6]:
        text = item["document"]["clean_text"]

        for title in TITLE_TERMS:
            if title in text:
                counts[title] = counts.get(title, 0) + 3

        for match in INHERITANCE_RE.finditer(text):
            phrase = match.group(1)
            if phrase not in known_terms:
                counts[phrase] = counts.get(phrase, 0) + 3

        for match in KINSHIP_RE.finditer(text):
            phrase = match.group(1)
            counts[phrase] = counts.get(phrase, 0) + 2

        for token in QUESTION_TOKEN_RE.findall(text):
            normalized = token.strip()
            if (
                not normalized
                or normalized in known_terms
                or normalized in BRIDGE_STOP_WORDS
                or (len(normalized) == 1 and not normalized.isascii())
            ):
                continue
            score = 1
            if normalized in TITLE_TERMS:
                score += 2
            if normalized in RELATION_TERMS:
                score += 2
            counts[normalized] = counts.get(normalized, 0) + score

    filtered_counts = {
        term: score
        for term, score in counts.items()
        if term in TITLE_TERMS or term in RELATION_TERMS or score >= 2
    }

    ranked = sorted(
        filtered_counts.items(),
        key=lambda item: (
            item[0] not in TITLE_TERMS,
            item[0] not in RELATION_TERMS,
            -item[1],
            len(item[0]),
        ),
    )
    return [term for term, _ in ranked[:6]]


def build_follow_up_queries(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if not any(token in question for token in IDENTITY_HINT_WORDS):
        return [], []

    bridge_terms = extract_bridge_terms(question, hypothesis, evidence)
    anchor = hypothesis.entities[0] if hypothesis.entities else ""

    queries: list[str] = []
    if anchor:
        queries.extend(
            [
                f"{anchor} 身世 真相",
                f"{anchor} 身份 来历",
            ]
        )

    for term in bridge_terms:
        if anchor:
            queries.append(f"{anchor} {term}")
        if term in TITLE_TERMS:
            queries.append(f"{term} 是谁")
            if anchor:
                queries.append(f"{anchor} {term} 什么关系")
        if term in RELATION_TERMS and anchor:
            queries.append(f"{anchor} {term} 是谁")

    if anchor and any("真相" in item["document"]["clean_text"] for item in evidence[:4]):
        queries.append(f"{anchor} 身世 全部真相")

    deduped_queries = []
    seen: set[str] = {question.strip()}
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:6], bridge_terms


def enrich_hypothesis(
    hypothesis: HypothesisDocument,
    bridge_terms: list[str],
    follow_up_queries: list[str],
) -> HypothesisDocument:
    extra_entities = [
        term
        for term in bridge_terms
        if term not in RELATION_TERMS and term not in TITLE_TERMS and len(term) <= 6
    ]
    extra_keywords = bridge_terms + [
        token
        for query in follow_up_queries
        for token in QUESTION_TOKEN_RE.findall(query)
        if token not in COMMON_NON_ENTITY_WORDS
    ]
    return HypothesisDocument(
        question=hypothesis.question,
        intent=hypothesis.intent,
        entities=_dedupe_keep_order(hypothesis.entities + extra_entities)[:12],
        keywords=_dedupe_keep_order(hypothesis.keywords + extra_keywords)[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )


def merge_hypotheses(base: HypothesisDocument, follow_up: HypothesisDocument) -> HypothesisDocument:
    return HypothesisDocument(
        question=base.question,
        intent=base.intent,
        entities=_dedupe_keep_order(base.entities + follow_up.entities)[:12],
        keywords=_dedupe_keep_order(base.keywords + follow_up.keywords)[:20],
        expected_answer_type=follow_up.expected_answer_type or base.expected_answer_type,
        dialogue_context=base.dialogue_context,
    )


def build_fallback_follow_up_hypothesis(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    missing_slots: list[str],
) -> HypothesisDocument:
    bridge_terms = extract_bridge_terms(question, hypothesis, evidence)
    fallback_queries = build_follow_up_hypothesis_queries(question, hypothesis) + build_missing_slot_queries(
        hypothesis,
        missing_slots,
    )

    if hypothesis.intent in {"character_relation", "event_summary"} or any(
        token in question for token in STORY_HINT_WORDS
    ):
        fallback_queries.extend(["共同经历", "并肩作战", "过往", "约定", "同行"])

    fallback_queries = _dedupe_keep_order(fallback_queries)[:12]
    enriched = enrich_hypothesis(hypothesis, bridge_terms, fallback_queries)

    if hypothesis.intent in {"character_relation", "event_summary"} or any(
        token in question for token in STORY_HINT_WORDS
    ):
        enriched = HypothesisDocument(
            question=enriched.question,
            intent=enriched.intent,
            entities=enriched.entities[:8],
            keywords=_dedupe_keep_order(enriched.keywords + ["共同经历", "并肩作战", "过往", "约定", "同行"])[:20],
            expected_answer_type="共同经历" if hypothesis.intent == "event_summary" else enriched.expected_answer_type,
            dialogue_context=enriched.dialogue_context,
        )

    return enriched


def build_follow_up_hypothesis_queries(
    question: str,
    hypothesis: HypothesisDocument,
) -> list[str]:
    queries: list[str] = []
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""

    if primary_entity and any(token in question for token in IDENTITY_HINT_WORDS):
        queries.extend(
            [
                f"{primary_entity} 身份 来历",
                f"{primary_entity} 身世 真相",
            ]
        )

    queries.extend(hypothesis.keywords[:6])

    for entity in hypothesis.entities[:4]:
        queries.append(entity)
        for keyword in hypothesis.keywords[:4]:
            if keyword != entity:
                queries.append(f"{entity} {keyword}")

    deduped_queries: list[str] = []
    seen: set[str] = {question.strip()}
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:8]


def build_missing_slot_queries(
    hypothesis: HypothesisDocument,
    missing_slots: list[str],
) -> list[str]:
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""
    queries: list[str] = []
    for slot in missing_slots[:6]:
        queries.append(slot)
        if primary_entity:
            queries.append(f"{primary_entity} {slot}")
    deduped_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:8]


def merge_ranked_hits(*ranked_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ranked in ranked_lists:
        for item in ranked:
            doc_index = int(item["doc_index"])
            if doc_index in seen:
                continue
            seen.add(doc_index)
            merged.append(item)
    return merged


def rerank_hits(
    retriever: ArknightsHybridRetriever,
    rerank_query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    if not retriever.reranker:
        return hits[:top_k]
    scores = retriever.reranker.score(
        query=rerank_query,
        documents=[item["document"]["search_text"] for item in hits],
        batch_size=batch_size,
    )
    reranked = []
    for item, score in zip(hits, scores):
        payload = dict(item)
        payload["rerank_score"] = float(score)
        reranked.append(payload)
    reranked.sort(key=lambda item: item.get("rerank_score", float("-inf")), reverse=True)
    return reranked[:top_k]


def render_evidence_blocks(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    blocks = []
    total_chars = 0
    for index, item in enumerate(evidence, start=1):
        doc = item["document"]
        clean_text = str(doc["clean_text"])
        if max_chars_per_doc is not None:
            clean_text = _truncate_text(clean_text, max_chars_per_doc)
        block = [
            f"[证据 {index}]",
            f"id: {doc['id']}",
            f"activity_name: {doc.get('activity_name') or ''}",
            f"story_name: {doc.get('story_name') or ''}",
            f"stage_code: {doc.get('stage_code') or ''}",
            f"avg_tag: {doc.get('avg_tag') or ''}",
            f"source_path: {doc.get('source_path') or ''}",
            "clean_text:",
            clean_text,
        ]
        rendered_block = "\n".join(block)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def summarize_evidence_for_trace(
    evidence: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for item in evidence[:limit]:
        doc = item["document"]
        snippet = re.sub(r"\s+", " ", doc["clean_text"]).strip()[:80]
        summary.append(
            {
                "id": str(doc["id"]),
                "activity_name": str(doc.get("activity_name") or ""),
                "story_name": str(doc.get("story_name") or ""),
                "stage_code": str(doc.get("stage_code") or ""),
                "snippet": snippet,
            }
        )
    return summary


def render_retrieval_history(
    retrieval_trace: list[dict[str, Any]],
    *,
    max_rounds: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    if not retrieval_trace:
        return "无"

    blocks: list[str] = []
    selected_steps = retrieval_trace[-max_rounds:] if max_rounds is not None else retrieval_trace
    total_chars = 0
    for step in selected_steps:
        lines = [f"[检索轮次 {step.get('round', '?')}]"]
        planner_action = str(step.get("planner_action") or "initial_retrieval").strip()
        lines.append(f"planner_action: {planner_action}")

        queries = [str(query).strip() for query in step.get("queries") or [] if str(query).strip()]
        if queries:
            lines.append("queries: " + " | ".join(queries[:6]))

        missing_slots = [
            str(slot).strip()
            for slot in step.get("missing_slots") or []
            if str(slot).strip()
        ]
        if missing_slots:
            lines.append("missing_slots: " + " | ".join(missing_slots[:6]))

        clarification_question = str(step.get("clarification_question") or "").strip()
        if clarification_question:
            lines.append(f"clarification_question: {clarification_question}")

        evidence_summary = step.get("evidence_summary") or []
        if evidence_summary:
            evidence_lines = []
            for item in evidence_summary[:3]:
                label = (
                    item.get("stage_code")
                    or item.get("story_name")
                    or item.get("activity_name")
                    or item.get("id")
                    or ""
                )
                snippet = str(item.get("snippet") or "").strip()
                evidence_lines.append(f"{label}: {snippet}")
            lines.append("evidence: " + " | ".join(evidence_lines))

        rendered_block = "\n".join(lines)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def render_generation_history(
    retrieval_trace: list[dict[str, Any]],
    *,
    max_rounds: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    if not retrieval_trace:
        return "无"

    blocks: list[str] = []
    selected_steps = retrieval_trace[-max_rounds:] if max_rounds is not None else retrieval_trace
    total_chars = 0
    for step in selected_steps:
        lines = [f"[生成历史 第{step.get('round', '?')}轮]"]
        hypothesis = step.get("hypothesis") or {}
        if isinstance(hypothesis, dict):
            intent = str(hypothesis.get("intent") or "").strip()
            entities = [str(item).strip() for item in hypothesis.get("entities") or [] if str(item).strip()]
            keywords = [str(item).strip() for item in hypothesis.get("keywords") or [] if str(item).strip()]
            if intent:
                lines.append(f"intent: {intent}")
            if entities:
                lines.append("entities: " + " | ".join(entities[:6]))
            if keywords:
                lines.append("keywords: " + " | ".join(keywords[:8]))

        conclusion = step.get("conclusion") or {}
        if isinstance(conclusion, dict) and conclusion:
            next_action = str(conclusion.get("next_action") or "").strip()
            answer = str(conclusion.get("answer") or "").strip()
            missing_slots = [
                str(item).strip()
                for item in conclusion.get("missing_slots") or []
                if str(item).strip()
            ]
            if next_action:
                lines.append(f"conclusion_action: {next_action}")
            if missing_slots:
                lines.append("conclusion_missing_slots: " + " | ".join(missing_slots[:6]))
            if answer:
                lines.append("conclusion_answer: " + re.sub(r"\s+", " ", answer)[:120])
        rendered_block = "\n".join(lines)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def build_unresolved_points(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    previous_missing_slots: list[str],
) -> list[str]:
    unresolved = list(previous_missing_slots)
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""

    if primary_entity and any(token in question for token in IDENTITY_HINT_WORDS):
        unresolved.append(f"需要确认 {primary_entity} 的真实身份或身世来源")

    for term in extract_bridge_terms(question, hypothesis, evidence)[:4]:
        unresolved.append(f"需要确认与“{term}”相关的桥接关系")

    if retrieval_trace:
        last_step = retrieval_trace[-1]
        last_queries = [
            str(query).strip()
            for query in last_step.get("queries") or []
            if str(query).strip()
        ]
        if last_queries:
            unresolved.append("需要在前轮检索基础上进一步缩小范围，避免重复使用相同查询")

    return _dedupe_keep_order([item for item in unresolved if item.strip()])[:8]


def build_answer_prompt(question: str, hypothesis: HypothesisDocument, evidence: list[dict[str, Any]]) -> str:
    rendered_dialogue_context = render_dialogue_context_for_prompt(hypothesis.dialogue_context)
    system_prompt = "\n".join(
        [
            "你是一个专业的《明日方舟》剧情问答助手。",
            "回答时优先事实，其次语气；优先证据，其次印象。",
            "请基于给定剧情证据作答，不要编造证据中没有出现的剧情。",
            "表达风格保持轻柔、礼貌、略带犹豫感，但不要过度口癖化。",
            "不要输出思维过程，不要输出链路分析。",
            "如果证据不足，明确说“现有检索证据不足以确认”。",
        ]
    )

    user_prompt = "\n\n".join(
        [
            f"用户问题:\n{question}",
            "多轮问答上下文:\n" + rendered_dialogue_context,
            "假设文档(JSON):\n" + json.dumps(asdict(hypothesis), ensure_ascii=False, indent=2),
            "检索证据:\n" + render_evidence_blocks(evidence),
            "\n请直接回答用户问题，并在必要时区分：",
            "1. 明确剧情事实",
            "2. 基于多段证据的归纳",
            "3. 无法确认的部分",
        ]
    )

    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def sanitize_generation_output(text: str, prompt: str) -> str:
    output = text.strip()
    if output.startswith(prompt):
        output = output[len(prompt):].lstrip()
    output = LLAMA_TIMING_LINE_RE.sub("", output).strip()
    output = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL).strip()
    output = re.sub(r"^warning:.*$", "", output, flags=re.MULTILINE).strip()
    output = re.sub(r"^(main|common_|llama_|load_|print_info:|system_info:|sampler ).*$", "", output, flags=re.MULTILINE).strip()
    output = output.replace("[end of text]", "").strip()
    return output


def extract_json_object(text: str) -> dict[str, Any] | None:
    fenced_match = JSON_BLOCK_RE.search(text)
    candidate = fenced_match.group(1) if fenced_match else text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(candidate[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None

    # Tolerate truncated JSON that is otherwise structurally valid except for
    # missing closing braces at the end of generation.
    if depth > 0 and not in_string:
        repaired_candidate = candidate[start:] + ("}" * depth)
        try:
            parsed = json.loads(repaired_candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [item for item in value if isinstance(item, (str, int, float))]
    else:
        return []
    return _dedupe_keep_order([str(item).strip() for item in items if str(item).strip()])[:limit]


def normalize_hypothesis_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str | None = None,
) -> HypothesisDocument:
    is_follow_up = current_intent is not None
    allowed_fields = FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS if is_follow_up else INITIAL_HYPOTHESIS_SCHEMA_FIELDS
    allowed_keys = set(allowed_fields)
    if is_follow_up:
        # Be tolerant here: some model outputs still echo `intent` even though
        # follow-up prompts ask it to inherit the previous round's intent.
        allowed_keys.add("intent")
    extra_keys = set(payload) - allowed_keys
    if extra_keys:
        raise ModelOutputError(f"unexpected hypothesis fields: {sorted(extra_keys)}")
    optional_missing_fields = {"dialogue_context"}
    if not is_follow_up:
        optional_missing_fields.update({"question", "intent"})
    missing_fields = [
        field
        for field in allowed_fields
        if field not in payload and field not in optional_missing_fields
    ]
    if missing_fields:
        raise ModelOutputError(f"missing hypothesis fields: {missing_fields}")

    inferred_intent, inferred_answer_type = detect_intent(question)
    intent = current_intent or str(payload.get("intent", "")).strip() or inferred_intent
    intent = LEGACY_INTENT_MAP.get(intent, intent)
    if intent not in HYPOTHESIS_INTENTS:
        raise ModelOutputError(f"invalid hypothesis intent: {intent or '<empty>'}")

    entities = _normalize_string_list(payload.get("entities"), limit=12)
    if not entities:
        raise ModelOutputError("hypothesis must contain non-empty entities")

    keywords = _normalize_string_list(payload.get("keywords"), limit=20)
    if not keywords:
        raise ModelOutputError("hypothesis must contain non-empty keywords")

    expected_answer_type = str(payload.get("expected_answer_type", "")).strip() or inferred_answer_type
    if not expected_answer_type:
        raise ModelOutputError("hypothesis must contain expected_answer_type")

    return HypothesisDocument(
        question=question,
        intent=intent,
        entities=entities,
        keywords=keywords,
        expected_answer_type=expected_answer_type,
        dialogue_context=dialogue_context.strip(),
    )


def normalize_conclusion_payload(
    payload: dict[str, Any],
    *,
    question: str,
    max_round_reached: bool = False,
) -> ConclusionResult:
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        raise ModelOutputError(f"unexpected conclusion fields: {sorted(extra_keys)}")
    missing_fields = [field for field in CONCLUSION_SCHEMA_FIELDS if field not in payload]
    if missing_fields:
        raise ModelOutputError(f"missing conclusion fields: {missing_fields}")
    payload_question = str(payload.get("question", "")).strip()
    if not payload_question:
        raise ModelOutputError("conclusion must contain question")
    next_action = str(payload.get("next_action", "")).strip()
    if next_action not in RETRIEVAL_ACTIONS:
        raise ModelOutputError(f"invalid conclusion action: {next_action or '<empty>'}")
    answer = str(payload.get("answer", "") or "").strip()
    missing_slots = _normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question", "")).strip()
    if next_action in {"answer_directly", "abstain"} and not answer:
        raise ModelOutputError(f"{next_action} requires non-empty answer")
    if next_action == "clarify_user" and not clarification_question:
        raise ModelOutputError("clarify_user requires clarification_question")
    if next_action == "retrieve_more":
        if answer:
            raise ModelOutputError("retrieve_more requires empty answer")
        if not missing_slots:
            raise ModelOutputError("retrieve_more requires non-empty missing_slots")
        if max_round_reached:
            next_action = "abstain"
            answer = "现有检索证据不足以确认，且已达到检索轮次上限。"
    return ConclusionResult(
        next_action=next_action,
        answer=answer,
        missing_slots=missing_slots,
        clarification_question=clarification_question,
    )


def validate_conclusion_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
) -> ConclusionResult:
    if conclusion.next_action != "answer_directly" or not conclusion.answer:
        return conclusion

    support_text_parts = [question, hypothesis.question, hypothesis.dialogue_context]
    support_text_parts.extend(hypothesis.entities)
    support_text_parts.extend(hypothesis.keywords)
    support_text_parts.extend(str(item["document"].get("clean_text") or "") for item in evidence[:6])
    support_text = "\n".join(part for part in support_text_parts if part).strip()
    support_tokens = set(_extract_content_tokens(support_text))

    answer_tokens = _extract_content_tokens(conclusion.answer)
    unsupported_tokens = [
        token
        for token in answer_tokens
        if token not in support_tokens and token not in IDENTITY_HINT_WORDS
    ]
    supported_tokens = [token for token in answer_tokens if token in support_tokens]

    if len(unsupported_tokens) >= 3 and len(unsupported_tokens) > len(supported_tokens):
        if max_round_reached:
            return ConclusionResult(
                next_action="abstain",
                answer="现有检索证据不足以确认，且已达到检索轮次上限。",
                missing_slots=[],
                clarification_question="",
            )
        return ConclusionResult(
            next_action="retrieve_more",
            answer="",
            missing_slots=_dedupe_keep_order(
                conclusion.missing_slots
                + [f"需要证据支持这些未落地说法：{', '.join(unsupported_tokens[:4])}"]
            )[:8],
            clarification_question="",
        )

    return conclusion


class LlamaCppRunner:
    backend_name = "llama.cpp"

    def __init__(
        self,
        *,
        llama_cli_path: Path,
        gguf_model_path: Path,
        lora_path: Path | None = None,
        threads: int | None = None,
        ctx_size: int = 8192,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
        gpu_layers: str | int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        flash_attn: str | None = None,
    ) -> None:
        self.llama_cli_path = llama_cli_path
        self.gguf_model_path = gguf_model_path
        self.lora_path = lora_path
        self.threads = threads or max(1, os.cpu_count() or 1)
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.gpu_layers = gpu_layers
        self.device = device
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.flash_attn = flash_attn

    def _has_gpu_backend(self) -> bool:
        bin_dir = self.llama_cli_path.parent
        for pattern in ("libggml-cuda*", "libggml-vulkan*", "libggml-hip*", "libggml-sycl*"):
            if any(bin_dir.glob(pattern)):
                return True
        return False

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "generator_backend": self.backend_name,
            "gguf_model_path": str(self.gguf_model_path),
            "base_model_path": None,
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "trained_sft_artifact": "model/lora/teacher_v2_plus_prompt_supplement_v2_qwen35_4b",
            "trained_sft_artifact_type": "LoRA adapter",
            "recommended_runtime_model": (
                "model/gguf/teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf"
            ),
            "runtime_mode": "merged_gguf" if not self.lora_path else "base_gguf_plus_lora_gguf",
            "llama_device": self.device,
            "gpu_layers": self.gpu_layers,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(
                "llama.cpp CLI not found: "
                f"{self.llama_cli_path}\n"
                "Please pass the real `--llama-cli` path, for example `/abs/path/to/llama.cpp/build/bin/llama-cli`."
            )
        if not self.gguf_model_path.exists():
            raise FileNotFoundError(
                "GGUF model not found: "
                f"{self.gguf_model_path}\n"
                "Please pass the real `--gguf-model` path to a converted GGUF file.\n"
                "Recommended runtime artifact in this repo: "
                "`model/gguf/teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf`."
            )
        if self.lora_path is not None and not self.lora_path.exists():
            raise FileNotFoundError(
                "LoRA path not found: "
                f"{self.lora_path}\n"
                "Please pass the real `--lora-path` directory or omit this option."
            )
        if self.lora_path is not None and self.lora_path.is_dir():
            raise FileNotFoundError(
                "llama.cpp does not load Hugging Face LoRA directories directly: "
                f"{self.lora_path}\n"
                "Use a GGUF LoRA adapter file, or omit `--lora-path` and run the merged GGUF "
                "`model/gguf/teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf`."
            )
        if self.device and self.device.lower() not in {"cpu", "none"} and not self._has_gpu_backend():
            raise RuntimeError(
                "The selected llama.cpp binary does not include a GPU backend.\n"
                f"Binary: {self.llama_cli_path}\n"
                "Current build appears CPU-only, so generation will be extremely slow.\n"
                "Rebuild llama.cpp with CUDA/HIP/Vulkan support, or switch to the `vllm` backend."
            )
        cmd = [
            str(self.llama_cli_path),
            "-m",
            str(self.gguf_model_path),
            "--no-warmup",
            "--no-display-prompt",
            "--simple-io",
            "--no-perf",
            "--no-conversation",
            "--no-jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "-t",
            str(self.threads),
            "-c",
            str(self.ctx_size),
            "-n",
            str(max_tokens if max_tokens is not None else self.max_tokens),
            "--temp",
            str(temperature if temperature is not None else self.temperature),
            "--top-p",
            str(top_p if top_p is not None else self.top_p),
            "--repeat-penalty",
            str(repeat_penalty if repeat_penalty is not None else self.repeat_penalty),
            "-p",
            prompt,
        ]
        if self.device:
            cmd.extend(["--device", self.device])
        if self.gpu_layers is not None:
            cmd.extend(["--gpu-layers", str(self.gpu_layers)])
        if self.batch_size is not None:
            cmd.extend(["--batch-size", str(self.batch_size)])
        if self.ubatch_size is not None:
            cmd.extend(["--ubatch-size", str(self.ubatch_size)])
        if self.flash_attn is not None:
            cmd.extend(["--flash-attn", self.flash_attn])
        if self.lora_path:
            cmd.extend(["--lora", str(self.lora_path)])

        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "llama.cpp inference failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{completed.stderr.strip()}\n"
                f"stdout:\n{completed.stdout.strip()}"
            )
        return sanitize_generation_output(completed.stdout, prompt)


class VllmRunner:
    backend_name = "vllm"

    def __init__(
        self,
        *,
        base_model_path: Path,
        lora_path: Path | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
        dtype: str = "auto",
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.tensor_parallel_size = max(1, tensor_parallel_size)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.dtype = dtype
        self._llm = None
        self._lora_request = None
        self._engine_init_error: Exception | None = None

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "generator_backend": self.backend_name,
            "gguf_model_path": None,
            "base_model_path": str(self.base_model_path),
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "trained_sft_artifact": "model/lora/teacher_v2_plus_prompt_supplement_v2_qwen35_4b",
            "trained_sft_artifact_type": "LoRA adapter",
            "recommended_runtime_model": str(self.base_model_path),
            "runtime_mode": "base_hf" if not self.lora_path else "base_hf_plus_lora_vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
        }

    def _ensure_engine(self):
        if self._llm is not None:
            return self._llm, self._lora_request
        if self._engine_init_error is not None:
            raise RuntimeError("vLLM engine initialization previously failed.") from self._engine_init_error
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                "Base model path not found for vLLM: "
                f"{self.base_model_path}\n"
                "Please pass a real `--base-model` path, for example `model/qwen3.5-4b`."
            )
        if self.lora_path is not None and not self.lora_path.exists():
            raise FileNotFoundError(
                "LoRA path not found for vLLM: "
                f"{self.lora_path}\n"
                "Please pass a real LoRA adapter directory or omit `--lora-path`."
            )
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed in the current environment. "
                "Run `bash scripts/install_train_vllm.sh` in the `train` environment first."
            ) from exc

        try:
            self._llm = LLM(
                model=str(self.base_model_path),
                trust_remote_code=True,
                enable_lora=self.lora_path is not None,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                disable_log_stats=True,
            )
            if self.lora_path is not None:
                self._lora_request = LoRARequest("goldenglow_sft", 1, str(self.lora_path))
            return self._llm, self._lora_request
        except Exception as exc:
            self._engine_init_error = exc
            raise

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        llm, lora_request = self._ensure_engine()
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed in the current environment. "
                "Run `bash scripts/install_train_vllm.sh` in the `train` environment first."
            ) from exc

        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            repetition_penalty=repeat_penalty if repeat_penalty is not None else self.repeat_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            skip_special_tokens=False,
        )
        outputs = llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no generation output.")
        return sanitize_generation_output(outputs[0].outputs[0].text, prompt)


class CPUInferencePipeline:
    def __init__(
        self,
        *,
        retriever: ArknightsHybridRetriever,
        generator: LlamaCppRunner | VllmRunner,
        query_config: QueryConfig | None = None,
        max_retrieval_rounds: int = 3,
        prompt_evidence_top_k: int = 8,
        max_follow_up_rounds: int | None = None,
        use_model_hypothesis: bool = True,
        use_model_conclusion_generation: bool = True,
        use_model_retrieval_planner: bool | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_config = query_config or QueryConfig()
        if max_follow_up_rounds is not None:
            max_retrieval_rounds = max_retrieval_rounds if max_retrieval_rounds != 3 else (max_follow_up_rounds + 1)
        self.max_retrieval_rounds = max(1, max_retrieval_rounds)
        if not use_model_hypothesis:
            raise ValueError("heuristic hypothesis generation is disabled; set use_model_hypothesis=true")
        if use_model_retrieval_planner is not None:
            use_model_conclusion_generation = use_model_retrieval_planner
        if not use_model_conclusion_generation:
            raise ValueError("heuristic conclusion generation is disabled; set use_model_conclusion_generation=true")
        self.use_model_hypothesis = use_model_hypothesis
        self.use_model_conclusion_generation = use_model_conclusion_generation
        self.prompt_evidence_top_k = max(1, prompt_evidence_top_k)

    def build_hypothesis(self, question: str, dialogue_context: str = "") -> HypothesisDocument:
        prompt = build_hypothesis_prompt(question, dialogue_context)
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(384, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
        if raw_output.lstrip().startswith('"'):
            raw_output = "{" + raw_output.lstrip()
        payload = extract_json_object(raw_output)
        if not payload:
            raise ModelOutputError(f"invalid hypothesis json: {raw_output}")
        model_hypothesis = normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=dialogue_context,
        )
        return model_hypothesis

    def build_follow_up_hypothesis(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        previous_conclusion: ConclusionResult,
        current_round: int,
    ) -> HypothesisDocument:
        unresolved_points = build_unresolved_points(
            question,
            current_hypothesis,
            evidence,
            retrieval_trace,
            previous_conclusion.missing_slots,
        )
        prompt = build_follow_up_hypothesis_prompt(
            question=question,
            current_hypothesis=current_hypothesis,
            evidence=evidence,
            unresolved_points=unresolved_points,
            retrieval_trace=retrieval_trace,
            previous_conclusion=previous_conclusion,
            current_round=current_round,
            max_retrieval_rounds=self.max_retrieval_rounds,
            prompt_evidence_top_k=self.prompt_evidence_top_k,
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=max(768, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
        if raw_output.lstrip().startswith('"'):
            raw_output = "{" + raw_output.lstrip()
        payload = extract_json_object(raw_output)
        if not payload:
            return build_fallback_follow_up_hypothesis(
                question=question,
                hypothesis=current_hypothesis,
                evidence=evidence,
                missing_slots=previous_conclusion.missing_slots,
            )
        follow_up_hypothesis = normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=current_hypothesis.dialogue_context,
            current_intent=current_hypothesis.intent,
        )
        return merge_hypotheses(current_hypothesis, follow_up_hypothesis)

    def generate_conclusion(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        current_round: int,
    ) -> ConclusionResult:
        prompt = build_conclusion_prompt(
            question,
            current_hypothesis,
            evidence,
            retrieval_trace,
            current_round,
            self.max_retrieval_rounds,
            self.prompt_evidence_top_k,
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(512, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
        if raw_output.lstrip().startswith('"'):
            raw_output = "{" + raw_output.lstrip()
        payload = extract_json_object(raw_output)
        if not payload:
            raise ModelOutputError(f"invalid conclusion json: {raw_output}")
        conclusion = normalize_conclusion_payload(
            payload,
            question=question,
            max_round_reached=current_round >= self.max_retrieval_rounds,
        )
        return conclusion

    def _search_queries(
        self,
        queries: list[str],
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        merged_dense = list(dense_hits)
        merged_sparse = list(sparse_hits)
        for query in queries:
            merged_dense = merge_ranked_hits(
                merged_dense,
                self.retriever.dense_search(query, top_k=self.query_config.dense_top_k),
            )
            merged_sparse = merge_ranked_hits(
                merged_sparse,
                self.retriever.sparse_search(query, top_k=self.query_config.sparse_top_k),
            )
        return merged_dense, merged_sparse

    def _finalize_hits(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved_question = _resolve_referential_question(question, hypothesis.entities)
        fused_hits = self.retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            top_k=self.query_config.fusion_top_k,
            rrf_k=self.query_config.rrf_k,
            dense_weight=self.query_config.dense_weight,
            sparse_weight=self.query_config.sparse_weight,
        )

        rerank_query = resolved_question
        if hypothesis.keywords:
            rerank_query = resolved_question + "\n检索线索: " + " ".join(hypothesis.keywords[:10])
        return rerank_hits(
            self.retriever,
            rerank_query,
            fused_hits,
            top_k=self.query_config.rerank_top_k,
            batch_size=self.query_config.rerank_batch_size,
        )

    def _retrieve_round(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        queries: list[str],
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        dense_hits, sparse_hits = self._search_queries(queries, dense_hits, sparse_hits)
        evidence = self._finalize_hits(question, hypothesis, dense_hits, sparse_hits)
        return dense_hits, sparse_hits, evidence

    def run(
        self,
        question: str,
        dialogue_context: str = "",
        progress_callback: Callable[[str], None] | None = None,
    ) -> InferenceResult:
        if progress_callback:
            progress_callback(INITIAL_HYPOTHESIS_TASK_TYPE)
        current_hypothesis = self.build_hypothesis(question, dialogue_context)
        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []
        retrieval_trace: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        final_answer = ""

        pending_queries = [
            _resolve_referential_question(question, current_hypothesis.entities),
            build_retrieval_query(current_hypothesis),
        ]
        current_hypothesis_task_type = INITIAL_HYPOTHESIS_TASK_TYPE

        for round_index in range(1, self.max_retrieval_rounds + 1):
            if progress_callback:
                progress_callback("retrieval")
            dense_hits, sparse_hits, evidence = self._retrieve_round(
                question,
                current_hypothesis,
                pending_queries,
                dense_hits,
                sparse_hits,
            )

            step_record: dict[str, Any] = {
                "round": round_index,
                "queries": list(pending_queries),
                "planner_action": "retrieval_completed",
                "hypothesis_task_type": current_hypothesis_task_type,
                "hypothesis": asdict(current_hypothesis),
                "evidence_summary": summarize_evidence_for_trace(evidence),
            }
            retrieval_trace.append(step_record)

            if progress_callback:
                progress_callback(CONCLUSION_TASK_TYPE)
            conclusion = self.generate_conclusion(
                question,
                current_hypothesis,
                evidence,
                retrieval_trace,
                round_index,
            )
            step_record["conclusion_task_type"] = CONCLUSION_TASK_TYPE
            step_record["conclusion"] = asdict(conclusion)
            step_record["planner_action"] = conclusion.next_action
            step_record["missing_slots"] = conclusion.missing_slots
            step_record["clarification_question"] = conclusion.clarification_question

            if conclusion.next_action == "answer_directly":
                final_answer = conclusion.answer
                break
            if conclusion.next_action == "clarify_user":
                final_answer = conclusion.clarification_question
                break
            if conclusion.next_action == "abstain":
                final_answer = conclusion.answer
                break

            if round_index >= self.max_retrieval_rounds:
                final_answer = conclusion.answer or "现有检索证据不足以确认。"
                break

            if progress_callback:
                progress_callback(FOLLOW_UP_HYPOTHESIS_TASK_TYPE)
            current_hypothesis = self.build_follow_up_hypothesis(
                question,
                current_hypothesis,
                evidence,
                retrieval_trace,
                conclusion,
                round_index + 1,
            )
            step_record["follow_up_hypothesis_task_type"] = FOLLOW_UP_HYPOTHESIS_TASK_TYPE
            step_record["follow_up_hypothesis"] = asdict(current_hypothesis)

            follow_up_queries = _dedupe_keep_order(
                build_follow_up_hypothesis_queries(question, current_hypothesis)
                + build_missing_slot_queries(current_hypothesis, conclusion.missing_slots)
            )[:10]
            if not follow_up_queries:
                raise ModelOutputError("follow-up hypothesis produced no retrieval queries")
            pending_queries = follow_up_queries + [build_retrieval_query(current_hypothesis)]
            step_record["next_round_queries"] = pending_queries
            current_hypothesis_task_type = FOLLOW_UP_HYPOTHESIS_TASK_TYPE

        retrieval_query = "\n\n".join(
            [
                f"[round {step['round']}]"
                + "\n"
                + "\n".join(step["queries"])
                for step in retrieval_trace
                if step.get("queries")
            ]
        )

        simplified_evidence = []
        for item in evidence:
            doc = item["document"]
            simplified_evidence.append(
                {
                    "id": doc["id"],
                    "activity_name": doc.get("activity_name"),
                    "story_name": doc.get("story_name"),
                    "stage_code": doc.get("stage_code"),
                    "avg_tag": doc.get("avg_tag"),
                    "source_path": doc.get("source_path"),
                    "fusion_score": item.get("fusion_score"),
                    "rerank_score": item.get("rerank_score"),
                    "dense_score": item.get("dense_score"),
                    "sparse_score": item.get("sparse_score"),
                    "clean_text": doc["clean_text"],
                }
            )

        return InferenceResult(
            question=question,
            intent=current_hypothesis.intent,
            hypothesis=asdict(current_hypothesis),
            model_runtime=self.generator.describe_runtime(),
            retrieval_query=retrieval_query,
            retrieval_trace=retrieval_trace,
            evidence=simplified_evidence,
            answer=final_answer,
        )
