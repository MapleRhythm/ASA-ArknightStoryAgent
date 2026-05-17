from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from asa_arknight_story_agent.config import OPERATOR_ALIAS_MAP_PATH, QueryConfig
from asa_arknight_story_agent.data.alias_map import load_operator_alias_map
from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever


QUESTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CHINESE_TOKEN_SPLIT_RE = re.compile(r"[的是和与及或为在把被让给从向对将]")
LINE_SPLIT_RE = re.compile(r"[\n\r。！？；]+")
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
CAUSAL_ANSWER_HINTS = {
    "为了",
    "因为",
    "因此",
    "所以",
    "目的",
    "动机",
    "原因",
    "选择",
    "放弃",
    "离队",
    "背叛",
    "回归",
    "必须",
    "不能",
    "才有",
    "才是",
    "只会",
    "只为",
    "机会",
    "真相",
    "活路",
    "阻止",
    "挽救",
    "避免",
    "毁灭",
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
REAL_NAME_RE = re.compile(r"(?:原名|本名|真名)[为叫是：:\s]*([\u4e00-\u9fff]{2,8}(?:·[\u4e00-\u9fff]{1,8})?)")
CONSPIRACY_ANCHOR_RE = re.compile(r"(?:撞破|发现|曝光|阻止)?([\u4e00-\u9fff]{2,4})城议员的阴谋")
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
QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
RETRIEVAL_ACTIONS_ORDER = (
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
)
INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"
INITIAL_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "intent",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
    "reflect_tokens",
)
FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
    "reflect_tokens",
)
CONCLUSION_SCHEMA_FIELDS = (
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
    "reflect_tokens",
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
PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS = 5000
PROMPT_EVIDENCE_MAX_CHARS_PER_DOC = 520
MULTI_QUERY_MERGE_RRF_K = 60


@dataclass(slots=True)
class HypothesisDocument:
    question: str
    intent: str
    query_type: str
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
    follow_up_hypothesis: HypothesisDocument | None


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
    if any(token in question for token in ("是什么", "本质", "来历")) and any(
        token in question for token in ("危机", "祸", "患", "威胁", "为什么", "为何")
    ):
        return "plot_reasoning", "概念定义/危机原因"
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


def infer_query_type(question: str, intent: str, expected_answer_type: str) -> str:
    if intent == "character_relation" or any(token in expected_answer_type for token in ("身份关系", "关系")):
        return "relation"
    if any(token in question for token in ("阴谋", "真相", "秘密", "识破", "揭穿", "曝光", "暴露", "幕后", "主使", "黑幕", "骗局", "诡计")):
        return "reveal"
    if any(token in question for token in ("谜", "怎么回事", "究竟", "到底")):
        return "mystery"
    if any(token in expected_answer_type for token in ("概念定义/危机原因", "answerability")):
        return "answerability"
    if intent == "plot_reasoning" or any(token in expected_answer_type for token in ("原因", "动机", "过程", "解释")):
        return "causality" if any(token in question for token in ("为什么", "为何", "原因", "导致", "造成")) else "reasoning"
    if intent in {"plot_fact", "timeline", "compare"}:
        return "fact"
    if any(token in expected_answer_type for token in ("事实", "时间线", "对比")):
        return "fact"
    return "reasoning"


def extract_entities(question: str, dialogue_context: str = "") -> list[str]:
    question_entities = [token for token in _extract_content_tokens(question) if _is_entity_candidate(token)]
    if any(pronoun in question for pronoun in PRONOUN_REFERENCES):
        return _dedupe_keep_order(_extract_context_entities(dialogue_context) + question_entities)[:12]
    return _dedupe_keep_order(_extract_context_entities(dialogue_context) + question_entities)[:12]


def _expand_entities_with_aliases(entities: list[str], existing_keywords: list[str]) -> list[str]:
    alias_map = load_operator_alias_map(OPERATOR_ALIAS_MAP_PATH)
    if not alias_map:
        return []
    return [alias for alias in alias_map.expand(entities) if alias not in existing_keywords]


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
    if answer_type == "概念定义/危机原因":
        concept_reasoning_keywords = [
            "本质",
            "原本",
            "一体",
            "苏醒",
            "消灭",
            "代价",
            "动乱",
            "灭顶之灾",
            "开战",
            "平息",
            "解决",
        ]
        keywords = _dedupe_keep_order(keywords + concept_reasoning_keywords)[:24]
    alias_keywords = _expand_entities_with_aliases(entities, keywords)
    if alias_keywords:
        keywords = _dedupe_keep_order(keywords + alias_keywords)[:24]
    return HypothesisDocument(
        question=question,
        intent=intent,
        query_type=infer_query_type(question, intent, answer_type),
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
            '- "query_type": 从以下集合中选择一个: fact, relation, causality, reasoning, reveal, mystery, answerability',
            "  - fact：事实细节；relation：人物关系；causality：事件前因后果；reasoning：剧情发展推理；reveal/mystery：真相、阴谋、身份揭示；answerability：同时需要定义与原因的可回答性问题",
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
            "6. 输出必须是合法 JSON，且字段严格限制为 question、intent、query_type、entities、keywords、expected_answer_type、dialogue_context。",
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
    prompt_evidence: list[dict[str, Any]] | None = None,
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
                prompt_evidence
                if prompt_evidence is not None
                else select_prompt_evidence(
                    question,
                    current_hypothesis,
                    evidence,
                    prompt_evidence_top_k=prompt_evidence_top_k,
                ),
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
            '- "query_type": 继承或修正为以下集合之一: fact, relation, causality, reasoning, reveal, mystery, answerability',
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
            "8. 输出必须是合法 JSON，且字段严格限制为 question, query_type, entities, keywords, expected_answer_type, dialogue_context。",
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
    prompt_evidence: list[dict[str, Any]] | None = None,
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
                prompt_evidence
                if prompt_evidence is not None
                else select_prompt_evidence(
                    question,
                    current_hypothesis,
                    evidence,
                    prompt_evidence_top_k=prompt_evidence_top_k,
                ),
                max_chars_per_doc=PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
                max_total_chars=PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
            ),
            "",
            "输出要求:",
            '1. 必须输出 JSON，字段严格包含 "question"、"next_action"、"answer"、"missing_slots"、"clarification_question"、"follow_up_hypothesis"。',
            '2. next_action 只能是 "answer_directly"、"retrieve_more"、"clarify_user"、"abstain"。',
            '3. 当 next_action = "answer_directly" 或 "abstain" 时，answer 必须非空，follow_up_hypothesis 必须为 null。',
            '4. 当 next_action = "clarify_user" 时，clarification_question 必须非空，follow_up_hypothesis 必须为 null。',
            '5. 当 next_action = "retrieve_more" 时，answer 必须为空字符串，missing_slots 应给出具体可检索缺口，follow_up_hypothesis 必须为非空 JSON 对象。',
            '6. follow_up_hypothesis 只能包含 "question"、"query_type"、"entities"、"keywords"、"expected_answer_type"、"dialogue_context"，不能包含 intent。',
            "7. 如果现有证据已经足够，请选择 answer_directly，不要为了流程强行继续检索。",
            "8. 如果问题本身歧义很大，请选择 clarify_user。",
            "9. 如果继续检索仍然缺乏明确方向，或已接近轮次上限，可选择 abstain。",
            "10. 如果问题属于“是谁 / 什么身份 / 来历 / 真相 / 关系”类，answer_directly 的第一句必须先直接回答核心判断，优先使用“X是Y”或“现有证据不足以确认X是谁/身份是什么”格式。",
            "11. 如果证据只支持侧面经历、行动轨迹、见闻或计划，不能把这些内容当作“是谁”的答案；此时应选择 retrieve_more 或 abstain。",
            "12. 禁止用“他去过哪里、见过谁、做过什么”替代核心身份判断。",
            "13. 若答案只能确认到部分事实，也要先明确标注“已确认部分 / 无法确认部分”，不要把推测包装成确定事实。",
            "14. 身份、种族、职业、阵营等标签必须由证据明确绑定到对应人物；不要把证据里属于其他人的标签转移给目标人物。",
            "15. 如果证据中只出现了“黎博利/萨科塔/萨卡兹/菲林”等词，但没有明确说明该人物就是这个种族，禁止写成“该人物是某种族”。",
            "16. 如果问题同时包含“是什么/本质”和“为什么成为危机/祸患/威胁”，answer 必须分两部分：先说明概念定义，再说明危机原因；每个原因都必须能在证据中找到对应表述。",
            "17. 不要把后续解决方案、结局、个人情感线当成“成为危机的原因”，除非证据明确说明它导致危机。",
            "18. 对概念定义/危机原因题，答案应保持最小充分：定义 1 句，危机原因 2-4 点；禁止把不同结局、肉鸽分支、设定传闻混写成确定主线事实。",
            "19. 如果证据来自不同活动、结局或分支，必须使用“现有证据显示/在这些证据中”限定，不要写成唯一官方全貌。",
            "20. 你的输出第一字符必须是 {",
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
        query_type=hypothesis.query_type,
        entities=_dedupe_keep_order(hypothesis.entities + extra_entities)[:12],
        keywords=_dedupe_keep_order(hypothesis.keywords + extra_keywords)[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )


def merge_hypotheses(base: HypothesisDocument, follow_up: HypothesisDocument) -> HypothesisDocument:
    return HypothesisDocument(
        question=base.question,
        intent=base.intent,
        query_type=follow_up.query_type or base.query_type,
        entities=_dedupe_keep_order(base.entities + follow_up.entities)[:12],
        keywords=_dedupe_keep_order(base.keywords + follow_up.keywords)[:20],
        expected_answer_type=follow_up.expected_answer_type or base.expected_answer_type,
        dialogue_context=base.dialogue_context,
    )


def enrich_follow_up_with_evidence_terms(
    hypothesis: HypothesisDocument,
    *,
    question: str,
    evidence: list[dict[str, Any]],
    missing_slots: list[str],
) -> HypothesisDocument:
    context_text = "\n".join(
        [question, hypothesis.question, *missing_slots]
        + [str(item["document"].get("clean_text") or "") for item in evidence[:4]]
    )
    if "阴谋" not in context_text and "具体" not in context_text:
        return hypothesis

    bridge_entities: list[str] = []
    bridge_keywords: list[str] = []
    for item in evidence[:4]:
        text = str(item["document"].get("clean_text") or "")
        for match in REAL_NAME_RE.finditer(text):
            full_name = match.group(1).strip()
            short_name = full_name.split("·", 1)[0].strip()
            bridge_entities.extend([short_name, full_name])
            bridge_keywords.extend([short_name, full_name])
        for match in CONSPIRACY_ANCHOR_RE.finditer(text):
            location = match.group(1).strip()
            bridge_entities.extend([location, f"{location}城议员", "城议员"])
            bridge_keywords.extend([location, f"{location}城议员", "城议员", "阴谋"])

    bridge_entities = _dedupe_keep_order(
        [
            term
            for term in bridge_entities
            if term and term not in hypothesis.entities and term not in COMMON_NON_ENTITY_WORDS
        ]
    )
    bridge_keywords = _dedupe_keep_order(
        [
            term
            for term in bridge_keywords
            if term and term not in COMMON_NON_ENTITY_WORDS
        ]
    )
    if not bridge_entities and not bridge_keywords:
        return hypothesis

    return HypothesisDocument(
        question=hypothesis.question,
        intent=hypothesis.intent,
        query_type=hypothesis.query_type,
        entities=_dedupe_keep_order(hypothesis.entities[:1] + bridge_entities + hypothesis.entities[1:])[:12],
        keywords=_dedupe_keep_order(bridge_keywords + hypothesis.keywords)[:20],
        expected_answer_type=hypothesis.expected_answer_type,
        dialogue_context=hypothesis.dialogue_context,
    )


def build_follow_up_hypothesis_queries(
    question: str,
    hypothesis: HypothesisDocument,
) -> list[str]:
    queries: list[str] = []
    primary_entity = hypothesis.entities[0] if hypothesis.entities else ""
    if "阴谋" in hypothesis.keywords or "阴谋" in question:
        bridge_entities = [
            entity
            for entity in hypothesis.entities[:6]
            if entity != primary_entity and entity not in COMMON_NON_ENTITY_WORDS
        ]
        for entity in bridge_entities[:4]:
            queries.append(f"{entity} 阴谋")
        if any("卡拉顿" in term for term in hypothesis.entities + hypothesis.keywords):
            for entity in bridge_entities[:3]:
                queries.append(f"{entity} 卡拉顿 阴谋")

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
        slot_parts = [
            part.strip(" ：:，,。；;")
            for part in re.split(r"[|/；;。]+", slot)
            if part.strip(" ：:，,。；;")
        ]
        for slot_part in slot_parts[:3]:
            compact_slot = _truncate_text(slot_part, 32)
            queries.append(compact_slot)
            if primary_entity:
                queries.append(f"{primary_entity} {compact_slot}")
    deduped_queries: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:8]


def _hit_raw_score(item: dict[str, Any]) -> float:
    for key in ("score", "dense_score", "sparse_score", "minirag_score", "fusion_score"):
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def merge_ranked_hits(*ranked_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            doc_index = int(item["doc_index"])
            raw_score = _hit_raw_score(item)
            payload = merged.setdefault(
                doc_index,
                {
                    "doc_index": doc_index,
                    "document": item["document"],
                    "score": raw_score,
                    "multi_query_rank_score": 0.0,
                    "multi_query_match_count": 0,
                    "best_query_rank": rank,
                },
            )
            payload["score"] = max(float(payload.get("score") or 0.0), raw_score)
            if item.get("minirag_score") is not None:
                payload["minirag_score"] = max(
                    float(payload.get("minirag_score") or 0.0),
                    float(item["minirag_score"]),
                )
            payload["multi_query_rank_score"] = float(payload.get("multi_query_rank_score") or 0.0) + (
                1.0 / (MULTI_QUERY_MERGE_RRF_K + rank + 1)
            )
            payload["multi_query_match_count"] = int(payload.get("multi_query_match_count") or 0) + 1
            previous_best_rank = payload.get("best_query_rank")
            payload["best_query_rank"] = min(
                int(previous_best_rank) if previous_best_rank is not None else rank,
                rank,
            )

    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("multi_query_rank_score") or 0.0),
            int(item.get("multi_query_match_count") or 0),
            -int(item.get("best_query_rank") or 0),
            float(item.get("score") or 0.0),
        ),
        reverse=True,
    )


def rerank_hits(
    retriever: ArknightsHybridRetriever,
    rerank_query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    batch_size: int,
    query_mode: str | None = None,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    if hasattr(retriever, "rerank_with_evidence_chains"):
        return retriever.rerank_with_evidence_chains(
            rerank_query,
            hits,
            top_k=top_k,
            batch_size=batch_size,
            query_mode=query_mode,
            fallback_to_document_rerank=True,
        )
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


def classify_retrieval_query_mode(hypothesis: HypothesisDocument) -> str:
    if hypothesis.query_type in QUERY_TYPES:
        return hypothesis.query_type
    answer_type = hypothesis.expected_answer_type
    question = hypothesis.question
    if hypothesis.intent == "character_relation" or any(token in answer_type for token in ("身份关系", "关系")):
        return "relation"
    if any(token in question for token in ("阴谋", "真相", "秘密", "识破", "揭穿", "曝光", "暴露", "幕后", "主使", "黑幕", "骗局", "诡计")):
        return "reveal"
    if any(token in question for token in ("谜", "怎么回事", "究竟", "到底")):
        return "mystery"
    if hypothesis.intent == "plot_reasoning" or any(token in answer_type for token in ("原因", "动机", "过程", "解释")):
        return "causality" if any(token in question for token in ("为什么", "为何", "原因", "导致", "造成")) else "reasoning"
    if any(token in answer_type for token in ("概念定义/危机原因", "answerability")):
        return "answerability"
    if hypothesis.intent in {"plot_fact", "timeline", "compare"}:
        return "fact"
    if any(token in answer_type for token in ("事实", "时间线", "对比")):
        return "fact"
    return "reasoning"


def render_evidence_blocks(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    blocks = []
    total_chars = 0
    seen_chain_texts: set[str] = set()
    seen_doc_ids: set[str] = set()
    for index, item in enumerate(evidence, start=1):
        doc = item["document"]
        chain_text = str(item.get("evidence_chain_text") or "").strip()
        if chain_text:
            if chain_text in seen_chain_texts:
                continue
            seen_chain_texts.add(chain_text)
            clean_text = chain_text
        else:
            doc_id = str(doc.get("id") or "")
            if doc_id and doc_id in seen_doc_ids:
                continue
            if doc_id:
                seen_doc_ids.add(doc_id)
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
            f"chain_roles: {','.join(item.get('evidence_chain_roles') or [])}",
            "clean_text:",
            clean_text,
        ]
        rendered_block = "\n".join(block)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def _evidence_score(item: dict[str, Any]) -> float:
    for key in (
        "evidence_chain_score",
        "evidence_chain_model_score",
        "rerank_score",
        "fusion_score",
        "dense_score",
        "sparse_score",
        "score",
    ):
        value = item.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _text_similarity_tokens(text: str) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    cjk_chars = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    cjk_bigrams = {
        cjk_chars[index] + cjk_chars[index + 1]
        for index in range(len(cjk_chars) - 1)
    }
    ascii_tokens = set(re.findall(r"[a-z0-9_]{2,}", normalized, flags=re.IGNORECASE))
    return cjk_bigrams | ascii_tokens


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union_size = len(left | right)
    if union_size == 0:
        return 0.0
    return len(left & right) / union_size


def select_prompt_evidence_mmr(
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
    lambda_mult: float,
) -> list[dict[str, Any]]:
    if prompt_evidence_top_k <= 0 or not evidence:
        return []
    if len(evidence) <= prompt_evidence_top_k:
        return evidence[:prompt_evidence_top_k]

    candidates = evidence[:]
    scores = [_evidence_score(item) for item in candidates]
    score_min = min(scores)
    score_max = max(scores)
    score_span = score_max - score_min
    normalized_scores = [
        1.0 if score_span <= 1e-9 else (score - score_min) / score_span
        for score in scores
    ]
    token_sets = [
        _text_similarity_tokens(str(item.get("document", {}).get("clean_text") or ""))
        for item in candidates
    ]

    selected_indices: list[int] = []
    remaining_indices = set(range(len(candidates)))
    while remaining_indices and len(selected_indices) < prompt_evidence_top_k:
        best_index = None
        best_score = float("-inf")
        for index in remaining_indices:
            diversity_penalty = 0.0
            if selected_indices:
                diversity_penalty = max(
                    _jaccard_similarity(token_sets[index], token_sets[selected_index])
                    for selected_index in selected_indices
                )
            mmr_score = lambda_mult * normalized_scores[index] - (1.0 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index
        if best_index is None:
            break
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    return [candidates[index] for index in selected_indices]


def apply_pyramid_evidence_order(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(evidence) <= 2:
        return evidence
    return [evidence[0], *evidence[2:], evidence[1]]


def split_evidence_strips(text: str, *, max_strips: int) -> list[str]:
    strips = [
        re.sub(r"\s+", " ", item).strip()
        for item in LINE_SPLIT_RE.split(text)
        if re.sub(r"\s+", " ", item).strip()
    ]
    return strips[:max_strips]


def select_prompt_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
) -> list[dict[str, Any]]:
    return evidence[:prompt_evidence_top_k]


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
    return _dedupe_keep_order([item for item in previous_missing_slots if item.strip()])[:8]


def build_answer_prompt(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    prompt_evidence_top_k: int,
    prompt_evidence: list[dict[str, Any]] | None = None,
) -> str:
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
            "检索证据:\n"
            + render_evidence_blocks(
                prompt_evidence
                if prompt_evidence is not None
                else select_prompt_evidence(
                    question,
                    hypothesis,
                    evidence,
                    prompt_evidence_top_k=prompt_evidence_top_k,
                ),
                max_chars_per_doc=PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
                max_total_chars=PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
            ),
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


def repair_json_like_output(text: str) -> str:
    candidate = text.lstrip()
    if not candidate:
        return text
    if candidate.startswith('"'):
        return "{" + candidate
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*"\s*:', candidate):
        return '{"' + candidate
    if re.match(r'^"[A-Za-z_][A-Za-z0-9_]*"\s*:', candidate):
        return "{" + candidate
    return text


def repair_common_json_syntax(text: str) -> str:
    repaired = text
    # Common 4B error: ["a",b", "c"] where the opening quote after comma is missing.
    repaired = re.sub(
        r'([,\[]\s*)([\u4e00-\u9fffA-Za-z_][^"\[\]\{\}:,\n\r]*?)"\s*(?=[,\]])',
        r'\1"\2"',
        repaired,
    )
    # Same issue for object values: "key":value", followed by comma or closing brace.
    repaired = re.sub(
        r'(:\s*)([\u4e00-\u9fffA-Za-z_][^"\[\]\{\}:,\n\r]*?)"\s*(?=[,\}])',
        r'\1"\2"',
        repaired,
    )
    # Missing value for optional nullable fields is safer as null than invalid JSON.
    repaired = re.sub(r'(:\s*)(?=[,\}])', r'\1null', repaired)
    return repaired


def extract_json_object(text: str) -> dict[str, Any] | None:
    fenced_match = JSON_BLOCK_RE.search(text)
    candidate = fenced_match.group(1) if fenced_match else text.strip()
    for candidate_variant in (candidate, repair_common_json_syntax(candidate)):
        try:
            parsed = json.loads(candidate_variant)
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
                object_candidate = candidate[start : index + 1]
                for candidate_variant in (object_candidate, repair_common_json_syntax(object_candidate)):
                    try:
                        parsed = json.loads(candidate_variant)
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        pass
                return None

    # Tolerate truncated JSON that is otherwise structurally valid except for
    # missing closing braces at the end of generation.
    if depth > 0 and not in_string:
        repaired_candidate = candidate[start:] + ("}" * depth)
        for candidate_variant in (repaired_candidate, repair_common_json_syntax(repaired_candidate)):
            try:
                parsed = json.loads(candidate_variant)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
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
    optional_missing_fields = {"dialogue_context", "query_type", "reflect_tokens"}
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
    query_type = str(payload.get("query_type", "")).strip()
    if query_type not in QUERY_TYPES:
        query_type = infer_query_type(question, intent, expected_answer_type)

    alias_keywords = _expand_entities_with_aliases(entities, keywords)
    if alias_keywords:
        keywords = _dedupe_keep_order(keywords + alias_keywords)[:24]

    return HypothesisDocument(
        question=question,
        intent=intent,
        query_type=query_type,
        entities=entities,
        keywords=keywords,
        expected_answer_type=expected_answer_type,
        dialogue_context=dialogue_context.strip(),
    )


def normalize_conclusion_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    max_round_reached: bool = False,
) -> ConclusionResult:
    if set(payload).issubset(set(INITIAL_HYPOTHESIS_SCHEMA_FIELDS)) or set(payload).issubset(
        set(FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS) | {"intent"}
    ):
        follow_up_hypothesis = normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
        )
        next_action = "abstain" if max_round_reached else "retrieve_more"
        answer = "现有检索证据不足以确认，且已达到检索轮次上限。" if max_round_reached else ""
        return ConclusionResult(
            next_action=next_action,
            answer=answer,
            missing_slots=["需要补充更直接的桥接证据"],
            clarification_question="",
            follow_up_hypothesis=None if max_round_reached else follow_up_hypothesis,
        )

    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        raise ModelOutputError(f"unexpected conclusion fields: {sorted(extra_keys)}")
    optional_missing_fields = {"clarification_question", "follow_up_hypothesis", "reflect_tokens"}
    missing_fields = [
        field for field in CONCLUSION_SCHEMA_FIELDS if field not in payload and field not in optional_missing_fields
    ]
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
    follow_up_hypothesis_payload = payload.get("follow_up_hypothesis")
    follow_up_hypothesis: HypothesisDocument | None = None
    if next_action in {"answer_directly", "abstain"} and not answer:
        raise ModelOutputError(f"{next_action} requires non-empty answer")
    if next_action == "clarify_user" and not clarification_question:
        raise ModelOutputError("clarify_user requires clarification_question")
    if next_action == "retrieve_more":
        if answer:
            raise ModelOutputError("retrieve_more requires empty answer")
        if not missing_slots:
            raise ModelOutputError("retrieve_more requires non-empty missing_slots")
        if not isinstance(follow_up_hypothesis_payload, dict):
            raise ModelOutputError("retrieve_more requires non-empty follow_up_hypothesis")
        follow_up_hypothesis = normalize_hypothesis_payload(
            follow_up_hypothesis_payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
        )
        if max_round_reached:
            next_action = "abstain"
            answer = "现有检索证据不足以确认，且已达到检索轮次上限。"
            follow_up_hypothesis = None
    else:
        if follow_up_hypothesis_payload not in (None, {}):
            raise ModelOutputError(f"{next_action} requires follow_up_hypothesis to be null")
    return ConclusionResult(
        next_action=next_action,
        answer=answer,
        missing_slots=missing_slots,
        clarification_question=clarification_question,
        follow_up_hypothesis=follow_up_hypothesis,
    )


GROUNDING_LONG_TOKEN_MIN_LEN = 3
GROUNDING_HIT_RATE_THRESHOLD = 0.5
GROUNDING_MIN_MISSED_LONG_TOKENS = 2
GROUNDING_EVIDENCE_POOL_TOP_K = 12


def _grounding_extract_answer_tokens(answer: str, question: str) -> list[str]:
    answer_tokens = [
        token
        for token in _extract_content_tokens(answer)
        if _is_entity_candidate(token)
        and token not in COMMON_NON_ENTITY_WORDS
        and token not in NOISY_RETRIEVAL_TOKENS
        and token not in PRONOUN_REFERENCES
    ]
    question_tokens = set(_extract_content_tokens(question))
    return [token for token in answer_tokens if token not in question_tokens]


def _grounding_evidence_pool(evidence: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in evidence[:GROUNDING_EVIDENCE_POOL_TOP_K]:
        document = item.get("document") or {}
        text = str(document.get("clean_text") or document.get("search_text") or "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def validate_conclusion_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
) -> ConclusionResult:
    if conclusion.next_action != "answer_directly":
        return conclusion
    if not conclusion.answer:
        return conclusion

    answer_tokens = _grounding_extract_answer_tokens(conclusion.answer, question)
    long_tokens = [token for token in answer_tokens if len(token) >= GROUNDING_LONG_TOKEN_MIN_LEN]
    if not long_tokens:
        return conclusion

    evidence_pool = _grounding_evidence_pool(evidence)
    if not evidence_pool:
        return conclusion

    missing_tokens = [token for token in long_tokens if token not in evidence_pool]
    hit_count = len(long_tokens) - len(missing_tokens)
    hit_rate = hit_count / len(long_tokens) if long_tokens else 1.0

    if (
        hit_rate < GROUNDING_HIT_RATE_THRESHOLD
        and len(missing_tokens) >= GROUNDING_MIN_MISSED_LONG_TOKENS
    ):
        downgraded_answer = (
            "现有检索证据不足以确认（grounding 校验未通过：答案中出现 "
            + "、".join(missing_tokens[:5])
            + " 等表述无法在已检索证据中找到对应支撑）。"
        )
        if max_round_reached:
            return ConclusionResult(
                next_action="abstain",
                answer=downgraded_answer,
                missing_slots=conclusion.missing_slots or ["grounding 校验未通过的关键词"],
                clarification_question="",
                follow_up_hypothesis=None,
            )
        return ConclusionResult(
            next_action="abstain",
            answer=downgraded_answer,
            missing_slots=conclusion.missing_slots or list(missing_tokens[:6]),
            clarification_question="",
            follow_up_hypothesis=None,
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
            "adapter_artifact": "model/lora/asa-arknightstoryagent-4b-lora",
            "adapter_artifact_type": "LoRA adapter",
            "recommended_runtime_model": "model/gguf/qwen3.5-4b-q4_k_m.gguf",
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
                "`model/gguf/qwen3.5-4b-q4_k_m.gguf`."
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
                "`model/gguf/qwen3.5-4b-q4_k_m.gguf`."
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
            "tokenizer_path": str(self.lora_path)
            if self.lora_path and (self.lora_path / "tokenizer_config.json").exists()
            else str(self.base_model_path),
            "adapter_artifact": "model/lora/asa-arknightstoryagent-4b-lora",
            "adapter_artifact_type": "LoRA adapter",
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
                "Install vLLM in your inference environment first, or use the llama.cpp backend."
            ) from exc

        try:
            tokenizer_path = (
                self.lora_path
                if self.lora_path and (self.lora_path / "tokenizer_config.json").exists()
                else self.base_model_path
            )
            self._llm = LLM(
                model=str(self.base_model_path),
                tokenizer=str(tokenizer_path),
                trust_remote_code=True,
                enable_lora=self.lora_path is not None,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                disable_log_stats=True,
            )
            if self.lora_path is not None:
                self._lora_request = LoRARequest("asa_sft", 1, str(self.lora_path))
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
                "Install vLLM in your inference environment first, or use the llama.cpp backend."
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
        enable_mmr: bool = False,
        mmr_lambda: float = 0.72,
        enable_pyramid_order: bool = False,
        enable_crag_refinement: bool = False,
        crag_refine_top_sentences: int = 4,
        crag_refine_max_sentences: int = 24,
        self_consistency_samples: int = 1,
        self_consistency_temperature: float = 0.7,
        max_follow_up_rounds: int | None = None,
        use_model_hypothesis: bool = True,
        use_model_conclusion_generation: bool = True,
        use_model_retrieval_planner: bool | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_config = query_config or QueryConfig()
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
        self.enable_mmr = enable_mmr
        self.mmr_lambda = min(1.0, max(0.0, mmr_lambda))
        self.enable_pyramid_order = enable_pyramid_order
        self.enable_crag_refinement = enable_crag_refinement
        self.crag_refine_top_sentences = max(1, crag_refine_top_sentences)
        self.crag_refine_max_sentences = max(self.crag_refine_top_sentences, crag_refine_max_sentences)
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.self_consistency_temperature = max(0.0, self_consistency_temperature)

    def prepare_prompt_evidence(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.enable_mmr:
            selected = select_prompt_evidence_mmr(
                evidence,
                prompt_evidence_top_k=self.prompt_evidence_top_k,
                lambda_mult=self.mmr_lambda,
            )
        else:
            selected = select_prompt_evidence(
                question,
                hypothesis,
                evidence,
                prompt_evidence_top_k=self.prompt_evidence_top_k,
            )
        if self.enable_crag_refinement:
            selected = self.refine_evidence_strips(question, hypothesis, selected)
        if self.enable_pyramid_order:
            selected = apply_pyramid_evidence_order(selected)
        return selected

    def refine_evidence_strips(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        reranker = getattr(self.retriever, "reranker", None)
        if reranker is None or not evidence:
            return evidence

        query = question
        if hypothesis.keywords:
            query = question + "\n检索线索: " + " ".join(hypothesis.keywords[:10])

        refined: list[dict[str, Any]] = []
        for item in evidence:
            doc = item.get("document") or {}
            clean_text = str(doc.get("clean_text") or "")
            strips = split_evidence_strips(clean_text, max_strips=self.crag_refine_max_sentences)
            if len(strips) <= self.crag_refine_top_sentences:
                refined.append(item)
                continue

            scores = reranker.score(
                query=query,
                documents=strips,
                batch_size=self.query_config.rerank_batch_size,
            )
            ranked = sorted(
                enumerate(zip(strips, scores)),
                key=lambda pair: float(pair[1][1]),
                reverse=True,
            )[: self.crag_refine_top_sentences]
            selected_indices = sorted(index for index, _ in ranked)
            selected_strips = [strips[index] for index in selected_indices]
            refined_doc = dict(doc)
            refined_doc["original_clean_text"] = clean_text
            refined_doc["clean_text"] = "\n".join(selected_strips)
            refined_doc["search_text"] = refined_doc["clean_text"]
            refined_item = dict(item)
            refined_item["document"] = refined_doc
            refined_item["crag_refinement"] = {
                "enabled": True,
                "original_sentence_count": len(strips),
                "kept_sentence_count": len(selected_strips),
                "kept_sentence_indices": selected_indices,
                "max_sentence_score": max(float(score) for score in scores) if scores else None,
            }
            refined.append(refined_item)
        return refined

    def build_hypothesis(self, question: str, dialogue_context: str = "") -> HypothesisDocument:
        prompt = build_hypothesis_prompt(question, dialogue_context)
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(384, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
        raw_output = repair_json_like_output(raw_output)
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
            prompt_evidence=self.prepare_prompt_evidence(question, current_hypothesis, evidence),
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=max(768, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
        raw_output = repair_json_like_output(raw_output)
        payload = extract_json_object(raw_output)
        if not payload:
            raise ModelOutputError(f"invalid follow-up hypothesis json: {raw_output}")
        follow_up_hypothesis = normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=current_hypothesis.dialogue_context,
            current_intent=current_hypothesis.intent,
        )
        return follow_up_hypothesis

    def generate_conclusion(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        current_round: int,
    ) -> ConclusionResult:
        prompt_evidence = self.prepare_prompt_evidence(question, current_hypothesis, evidence)
        prompt = build_conclusion_prompt(
            question,
            current_hypothesis,
            evidence,
            retrieval_trace,
            current_round,
            self.max_retrieval_rounds,
            self.prompt_evidence_top_k,
            prompt_evidence=prompt_evidence,
        )
        conclusions: list[ConclusionResult] = []
        errors: list[Exception] = []
        sample_count = self.self_consistency_samples
        for _ in range(sample_count):
            try:
                raw_output = self.generator.generate(
                    prompt,
                    max_tokens=min(512, self.generator.max_tokens),
                    temperature=self.self_consistency_temperature if sample_count > 1 else 0.1,
                    top_p=0.9 if sample_count > 1 else 0.8,
                    repeat_penalty=1.0,
                )
                raw_output = repair_json_like_output(raw_output)
                payload = extract_json_object(raw_output)
                if not payload:
                    raise ModelOutputError(f"invalid conclusion json: {raw_output}")
                conclusion = normalize_conclusion_payload(
                    payload,
                    question=question,
                    dialogue_context=current_hypothesis.dialogue_context,
                    current_intent=current_hypothesis.intent,
                    max_round_reached=current_round >= self.max_retrieval_rounds,
                )
                conclusion = validate_conclusion_grounding(
                    question=question,
                    hypothesis=current_hypothesis,
                    evidence=prompt_evidence,
                    conclusion=conclusion,
                    max_round_reached=current_round >= self.max_retrieval_rounds,
                )
                conclusions.append(conclusion)
            except Exception as exc:
                errors.append(exc)
                if sample_count == 1:
                    raise
                continue

        if not conclusions:
            first_error = errors[0] if errors else ModelOutputError("no valid conclusion samples")
            raise ModelOutputError("self-consistency produced no valid conclusion samples") from first_error

        action_counts: dict[str, int] = {}
        for conclusion in conclusions:
            action_counts[conclusion.next_action] = action_counts.get(conclusion.next_action, 0) + 1
        winning_action = max(
            action_counts,
            key=lambda action: (action_counts[action], -RETRIEVAL_ACTIONS_ORDER.index(action)),
        )
        return next(conclusion for conclusion in conclusions if conclusion.next_action == winning_action)

    def generate_direct_answer(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> ConclusionResult:
        prompt_evidence = self.prepare_prompt_evidence(question, current_hypothesis, evidence)
        prompt = build_answer_prompt(
            question,
            current_hypothesis,
            evidence,
            prompt_evidence_top_k=self.prompt_evidence_top_k,
            prompt_evidence=prompt_evidence,
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(512, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.0,
        )
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
        )

    def _search_queries(
        self,
        queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        dense_ranked_lists: list[list[dict[str, Any]]] = []
        sparse_ranked_lists: list[list[dict[str, Any]]] = []
        for query in queries:
            dense_ranked_lists.append(self.retriever.dense_search(query, top_k=self.query_config.dense_top_k))
            sparse_ranked_lists.append(self.retriever.sparse_search(query, top_k=self.query_config.sparse_top_k))
            minirag_search = getattr(self.retriever, "minirag_search", None)
            if minirag_search is not None:
                minirag_hits = minirag_search(query, top_k=self.query_config.sparse_top_k)
                if minirag_hits:
                    sparse_ranked_lists.append(minirag_hits)
        return merge_ranked_hits(*dense_ranked_lists), merge_ranked_hits(*sparse_ranked_lists)

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
        if self.query_config.enable_neighbor_expansion:
            fused_hits = self._expand_fused_hits_with_neighbors(fused_hits)

        rerank_query = resolved_question
        if hypothesis.keywords:
            rerank_query = resolved_question + "\n检索线索: " + " ".join(hypothesis.keywords[:10])
        return rerank_hits(
            self.retriever,
            rerank_query,
            fused_hits,
            top_k=self.query_config.rerank_top_k,
            batch_size=self.query_config.rerank_batch_size,
            query_mode=classify_retrieval_query_mode(hypothesis),
        )

    def _expand_fused_hits_with_neighbors(self, fused_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        collect_neighbors = getattr(self.retriever, "_collect_story_and_stage_neighbors", None)
        if not fused_hits or collect_neighbors is None:
            return fused_hits

        expanded_by_doc_index: dict[int, dict[str, Any]] = {
            int(item["doc_index"]): item
            for item in fused_hits
        }
        neighbor_doc_indices = collect_neighbors(
            fused_hits,
            max_seed_docs=min(self.query_config.neighbor_max_seed_docs, len(fused_hits)),
            story_window=self.query_config.neighbor_story_window,
            activity_story_sort_window=self.query_config.neighbor_activity_story_sort_window,
        )
        max_candidates = max(
            self.query_config.reranker_candidate_top_k,
            self.query_config.fusion_top_k,
            self.query_config.rerank_top_k,
        )
        for doc_index in neighbor_doc_indices:
            if doc_index in expanded_by_doc_index:
                continue
            expanded_by_doc_index[doc_index] = {
                "doc_index": doc_index,
                "document": self.retriever.documents[doc_index],
                "dense_score": None,
                "sparse_score": None,
                "fusion_score": 0.0,
                "supplemental_source": "neighbor",
            }
            if len(expanded_by_doc_index) >= max_candidates:
                break
        return list(expanded_by_doc_index.values())

    def _retrieve_round(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        dense_hits, sparse_hits = self._search_queries(queries)
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

            if conclusion.follow_up_hypothesis is not None:
                current_hypothesis = merge_hypotheses(current_hypothesis, conclusion.follow_up_hypothesis)
            else:
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

            pending_queries = [build_retrieval_query(current_hypothesis)]
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
                    "evidence_chain_score": item.get("evidence_chain_score"),
                    "evidence_chain_model_score": item.get("evidence_chain_model_score"),
                    "evidence_chain_roles": item.get("evidence_chain_roles"),
                    "evidence_chain_text": item.get("evidence_chain_text"),
                    "dense_score": item.get("dense_score"),
                    "sparse_score": item.get("sparse_score"),
                    "minirag_score": item.get("minirag_score"),
                    "clean_text": doc["clean_text"],
                }
            )

        return InferenceResult(
            question=question,
            intent=current_hypothesis.intent,
            hypothesis=asdict(current_hypothesis),
            model_runtime={
                **self.generator.describe_runtime(),
                "prompt_evidence_strategy": {
                    "top_k": self.prompt_evidence_top_k,
                    "mmr_enabled": self.enable_mmr,
                    "mmr_lambda": self.mmr_lambda,
                    "pyramid_order_enabled": self.enable_pyramid_order,
                    "crag_refinement_enabled": self.enable_crag_refinement,
                    "crag_refine_top_sentences": self.crag_refine_top_sentences,
                    "crag_refine_max_sentences": self.crag_refine_max_sentences,
                },
                "conclusion_self_consistency": {
                    "samples": self.self_consistency_samples,
                    "temperature": self.self_consistency_temperature,
                },
            },
            retrieval_query=retrieval_query,
            retrieval_trace=retrieval_trace,
            evidence=simplified_evidence,
            answer=final_answer,
        )
