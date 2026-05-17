from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenglow.config import DOCUMENTS_PATH, EXCEL_ROOT, OPERATOR_ALIAS_MAP_PATH, STORY_ROOT
from goldenglow.data.alias_map import load_operator_alias_map
from goldenglow.data.story_parser import build_corpus_documents


INITIAL_HYPOTHESIS_TASK_TYPE = "user_question_hypothesis_generation"
FOLLOW_UP_HYPOTHESIS_TASK_TYPE = "follow_up_hypothesis_generation"
CONCLUSION_TASK_TYPE = "conclusion_generation"

LEGACY_TOOL_TASK_TYPE_ALIASES = {
    "intent_hypothesis_rag": INITIAL_HYPOTHESIS_TASK_TYPE,
    "tool_calling_rag": FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    "unknown_rag_negative": CONCLUSION_TASK_TYPE,
}

SUPPORTED_TASK_TYPES = {
    "canon_qa",
    "worldbuilding_qa",
    "persona_grounded_qa",
    "multi_turn_dialogue",
    INITIAL_HYPOTHESIS_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    CONCLUSION_TASK_TYPE,
}

ACCEPTED_TASK_TYPES = SUPPORTED_TASK_TYPES | set(LEGACY_TOOL_TASK_TYPE_ALIASES)

TASK_BUCKET_MAP = {
    "canon_qa": "knowledge",
    "worldbuilding_qa": "knowledge",
    "persona_grounded_qa": "style",
    "multi_turn_dialogue": "tool",
    INITIAL_HYPOTHESIS_TASK_TYPE: "tool",
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE: "tool",
    CONCLUSION_TASK_TYPE: "tool",
    "intent_hypothesis_rag": "tool",
    "tool_calling_rag": "tool",
    "unknown_rag_negative": "tool",
}

INSUFFICIENT_EVIDENCE_MARKERS = (
    "证据不足",
    "无法确认",
    "不能确认",
    "不确定",
    "检索结果不足",
    "现有检索",
    "没有足够证据",
    "无法支持",
)

FORBIDDEN_FINAL_ANSWER_MARKERS = (
    "根据检索",
    "检索到的剧情证据",
    "根据剧情证据",
    "根据证据",
    "检索结果显示",
    "从检索结果来看",
)

INTENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_intent",
        "description": "识别用户问题的剧情问答意图、是否需要澄清以及是否需要检索。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "dialogue_context": {"type": "string"},
            },
            "required": ["question"],
        },
    },
}

HYPOTHESIS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "build_hypothesis",
        "description": "基于用户问题、意图和多轮上下文构造服务于检索的结构化假设文档。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "intent": {"type": "string"},
                "dialogue_context": {"type": "string"},
                "resolved_references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["question", "intent"],
        },
    },
}

RAG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_story_context",
        "description": "根据用户问题生成假设文档并检索明日方舟剧情证据，用于剧情问答。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "hypothesis": {"type": "string"},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "top_k": {"type": "integer"},
            },
            "required": ["question", "hypothesis", "keywords", "top_k"],
        },
    },
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PRIMARY_ENTITY_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")
CODE_NAME_RE = re.compile(r"【代号】\s*([\u4e00-\u9fff]{2,8})")
REAL_NAME_RE = re.compile(r"本名([\u4e00-\u9fff]{2,8})")
OPERATOR_NAME_RE = re.compile(r"干员([\u4e00-\u9fff]{2,8})")
PRIMARY_ENTITY_STOP_WORDS = {
    "用户问题",
    "多轮上下文",
    "当前假设",
    "当前证据",
    "历史生成",
    "历史检索",
    "检索轮次",
    "行动前",
    "行动后",
    "幕间",
    "档案",
    "语音",
    "剧情",
    "故事",
    "问题",
    "关系",
    "身份",
    "来历",
    "真相",
    "原因",
    "动机",
}
INITIAL_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
    "intent",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
    "reflect_tokens",
)
FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS = (
    "question",
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
    "reflect_tokens",
)

# Self-RAG reflect token enums
REFLECT_RETRIEVE_VALUES = frozenset({"Yes", "No", "Continue"})
REFLECT_RELEVANT_VALUES = frozenset({"Relevant", "Irrelevant", "Partial"})
REFLECT_SUPPORTED_VALUES = frozenset({"Fully", "Partially", "NoSupport"})
REFLECT_USEFUL_VALUES = frozenset({"Useful", "PartiallyUseful", "Useless"})


def _normalize_reflect_tokens(value: Any) -> dict[str, str] | None:
    """Coerce reflect_tokens into the canonical 4-field dict, returning None if invalid."""
    if not isinstance(value, dict):
        return None
    retrieve = str(value.get("Retrieve") or "").strip()
    relevant = str(value.get("Relevant") or "").strip()
    supported = str(value.get("Supported") or "").strip()
    useful = str(value.get("Useful") or "").strip()
    if (
        retrieve not in REFLECT_RETRIEVE_VALUES
        or relevant not in REFLECT_RELEVANT_VALUES
        or supported not in REFLECT_SUPPORTED_VALUES
        or useful not in REFLECT_USEFUL_VALUES
    ):
        return None
    return {
        "Retrieve": retrieve,
        "Relevant": relevant,
        "Supported": supported,
        "Useful": useful,
    }


def _default_reflect_tokens_for_action(next_action: str) -> dict[str, str]:
    """Provide deterministic defaults when teacher forgets reflect_tokens."""
    if next_action == "answer_directly":
        return {"Retrieve": "No", "Relevant": "Relevant", "Supported": "Fully", "Useful": "Useful"}
    if next_action == "abstain":
        return {"Retrieve": "No", "Relevant": "Irrelevant", "Supported": "NoSupport", "Useful": "Useless"}
    if next_action == "clarify_user":
        return {"Retrieve": "No", "Relevant": "Partial", "Supported": "Partially", "Useful": "PartiallyUseful"}
    # retrieve_more / unknown
    return {"Retrieve": "Yes", "Relevant": "Partial", "Supported": "Partially", "Useful": "PartiallyUseful"}
RETRIEVAL_ACTIONS = {
    "answer_directly",
    "retrieve_more",
    "clarify_user",
    "abstain",
}
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
LEGACY_INTENT_MAP = {
    "plot_inference": "plot_reasoning",
    "plot_motivation": "plot_reasoning",
    "character_motivation": "plot_reasoning",
    "identity_relationship": "character_relation",
    "character_relationship": "character_relation",
    "relationship_inference": "character_relation",
    "character_identity": "plot_fact",
    "role_identification": "plot_fact",
    "plot_item": "plot_fact",
    "plot_explanation": "plot_reasoning",
    "plot_qa": "plot_fact",
    "follow_up": "plot_fact",
    "clarification_needed": "out_of_scope",
}
LEGACY_PROMPT_HYPOTHESIS_MARKERS = (
    '"character_name"',
    '"appearances"',
    '"known_info"',
    '"relationship_hints"',
    '"bridging_objects"',
    '"bridge_objects"',
    '"upstream_titles"',
    '"upper_titles"',
    '"relationship_clues"',
)


@dataclass(slots=True)
class TeacherApiConfig:
    api_type: str
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: int = 120
    temperature: float = 0.8
    max_output_tokens: int = 4000
    json_mode: bool = True
    extra_headers: dict[str, str] | None = None
    auth_header: str = "bearer"
    anthropic_disable_thinking: bool = False


def load_generation_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_story_documents(documents_path: Path | None = None) -> list[dict]:
    path = documents_path or DOCUMENTS_PATH
    if path.exists():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return build_corpus_documents(STORY_ROOT, EXCEL_ROOT)


def normalize_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def _is_noisy_term(value: str) -> bool:
    lowered = value.lower()
    return (
        "{@" in value
        or "{" in value
        or "}" in value
        or "@nickname" in lowered
        or lowered.startswith("dr.")
        or lowered.startswith("doctor ")
    )


# Hypothesis keyword 黑名单：纯问句词没有检索锚定价值，过滤掉
_HYPOTHESIS_KEYWORD_BLACKLIST = frozenset(
    {
        "什么", "为什么", "为何", "怎么", "如何", "原因", "动机", "目的",
        "关系", "身份", "来历", "真相", "故事", "经历", "情况", "情节",
        "内容", "台词", "说了什么", "做了什么", "发生", "时候", "经过",
        "讲了什么", "讲什么",
    }
)


def _is_blacklisted_hypothesis_keyword(value: str) -> bool:
    token = value.strip()
    if not token:
        return True
    if token in _HYPOTHESIS_KEYWORD_BLACKLIST:
        return True
    # 单字中文（除主实体外信息量低）
    if len(token) == 1 and "一" <= token <= "鿿":
        return True
    # 纯英文且 < 3 个字符
    if token.isascii() and len(token) < 3:
        return True
    return False


# Pronouns / 问句残片：禁止出现在 entities
_ENTITY_PRONOUN_BLACKLIST = frozenset(
    {
        "她", "他", "它", "她们", "他们", "它们",
        "这位", "那位", "这个人", "那个人", "这件事", "那件事",
        "我", "你", "我们", "你们", "自己",
    }
)


def _is_invalid_entity(value: str) -> bool:
    token = value.strip()
    if not token or len(token) > 12:
        return True
    if token in _ENTITY_PRONOUN_BLACKLIST:
        return True
    # 问句残片：包含问句词
    if any(qw in token for qw in ("什么", "为什么", "怎么", "如何", "为何", "哪")):
        return True
    # 描述性短语而非命名实体（含动词/虚词）
    if any(suffix in token for suffix in ("吗", "呢", "啊", "吧", "了", "的", "之间", "之后", "之前")):
        return True
    return False


def _filter_hypothesis_entities(entities: list[str]) -> list[str]:
    """Drop pronouns, question fragments, descriptive phrases."""
    return _dedupe_keep_order([e for e in entities if not _is_invalid_entity(e)])[:12]


def _filter_hypothesis_keywords(
    keywords: list[str],
    *,
    entities: list[str],
) -> list[str]:
    """Drop blacklisted question words & noise; ensure main entity is present."""
    cleaned = [
        kw
        for kw in keywords
        if not _is_blacklisted_hypothesis_keyword(kw)
    ]
    # 确保主实体在 keywords 中（线上检索需要锚定）
    if entities:
        main_entity = entities[0]
        if main_entity and main_entity not in cleaned:
            cleaned = [main_entity, *cleaned]
    return _dedupe_keep_order(cleaned)[:20]


def _normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [item for item in value if isinstance(item, (str, int, float))]
    else:
        return []
    items = [
        str(item).strip()
        for item in raw_items
        if str(item).strip() and not _is_noisy_term(str(item).strip())
    ]
    return _dedupe_keep_order(items)[:limit]


def _normalize_intent(value: Any) -> str:
    intent = str(value or "").strip()
    intent = LEGACY_INTENT_MAP.get(intent, intent)
    return intent if intent in HYPOTHESIS_INTENTS else ""


def normalize_task_type(task_type: Any) -> str:
    normalized = str(task_type or "").strip()
    return LEGACY_TOOL_TASK_TYPE_ALIASES.get(normalized, normalized)


def normalize_task_mix(task_mix: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for task_type, weight in task_mix.items():
        canonical = normalize_task_type(task_type)
        if canonical not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"Unsupported task type in task_mix: {task_type}")
        normalized[canonical] = normalized.get(canonical, 0.0) + float(weight)
    return normalized


def _extract_expected_answer_type(payload: dict[str, Any]) -> str:
    direct_value = str(payload.get("expected_answer_type") or "").strip()
    if direct_value:
        return direct_value
    constraints = payload.get("constraints")
    if isinstance(constraints, dict):
        return str(constraints.get("expected_answer_type") or "").strip()
    return ""


def _normalize_hypothesis_tool_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    question = str(payload.get("question") or "").strip()
    intent = _normalize_intent(payload.get("intent"))
    entities = _filter_hypothesis_entities(_normalize_string_list(payload.get("entities"), limit=12))
    keywords = _filter_hypothesis_keywords(
        _normalize_string_list(payload.get("keywords"), limit=20),
        entities=entities,
    )
    expected_answer_type = _extract_expected_answer_type(payload)
    dialogue_context = str(payload.get("dialogue_context") or "").strip()

    if not question or not intent or not entities or not keywords or not expected_answer_type:
        return None

    normalized = {
        "question": question,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
    }
    if dialogue_context:
        normalized["dialogue_context"] = dialogue_context
    return normalized


def _parse_json_content(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    match = JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _contains_empty_current_hypothesis(user_text: str) -> bool:
    normalized = re.sub(r"\s+", "", user_text)
    return "当前假设文档(JSON):{}" in normalized or "当前假设文档:{}" in normalized


def _contains_legacy_prompt_hypothesis_schema(user_text: str) -> bool:
    return any(marker in user_text for marker in LEGACY_PROMPT_HYPOTHESIS_MARKERS)


def _normalize_initial_hypothesis_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    required_fields = tuple(
        field
        for field in INITIAL_HYPOTHESIS_SCHEMA_FIELDS
        if field not in {"dialogue_context", "reflect_tokens"}
    )
    if any(field not in payload for field in required_fields):
        return None
    extra_keys = set(payload) - set(INITIAL_HYPOTHESIS_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    intent = _normalize_intent(payload.get("intent"))
    entities = _filter_hypothesis_entities(_normalize_string_list(payload.get("entities"), limit=12))
    keywords = _filter_hypothesis_keywords(
        _normalize_string_list(payload.get("keywords"), limit=20),
        entities=entities,
    )
    expected_answer_type = _extract_expected_answer_type(payload)
    dialogue_context = str(payload.get("dialogue_context") or "").strip()

    if not question or not intent or not entities or not keywords or not expected_answer_type:
        return None

    reflect_tokens = _normalize_reflect_tokens(payload.get("reflect_tokens"))
    if reflect_tokens is None:
        # Teacher omitted or produced invalid reflect_tokens; fall back to retrieval-friendly default.
        reflect_tokens = _default_reflect_tokens_for_action("retrieve_more")

    return {
        "question": question,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
        "reflect_tokens": reflect_tokens,
    }


def _normalize_follow_up_hypothesis_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    required_fields = tuple(
        field
        for field in FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS
        if field not in {"dialogue_context", "reflect_tokens"}
    )
    if any(field not in payload for field in required_fields):
        return None
    extra_keys = set(payload) - set(FOLLOW_UP_HYPOTHESIS_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    entities = _filter_hypothesis_entities(_normalize_string_list(payload.get("entities"), limit=12))
    keywords = _filter_hypothesis_keywords(
        _normalize_string_list(payload.get("keywords"), limit=20),
        entities=entities,
    )
    expected_answer_type = _extract_expected_answer_type(payload)
    dialogue_context = str(payload.get("dialogue_context") or "").strip()

    if not question or not entities or not keywords or not expected_answer_type:
        return None

    reflect_tokens = _normalize_reflect_tokens(payload.get("reflect_tokens"))
    if reflect_tokens is None:
        reflect_tokens = _default_reflect_tokens_for_action("retrieve_more")

    return {
        "question": question,
        "entities": entities,
        "keywords": keywords,
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
        "reflect_tokens": reflect_tokens,
    }


_GENERIC_MISSING_SLOTS = frozenset(
    {
        "更多信息", "更多细节", "背景信息", "相关信息", "详细背景",
        "详细资料", "完整剧情", "相关内容", "其他信息", "更多内容",
        "详细信息", "更多背景",
    }
)

_ANSWER_EXPOSURE_MARKERS = (
    "根据证据", "根据剧情", "根据检索", "基于证据", "基于检索",
    "从证据中", "根据上面", "根据以上", "检索到的", "根据剧情证据",
    "根据剧情片段",
)


def _filter_missing_slots(slots: list[str]) -> list[str]:
    """Drop generic / non-actionable slots like 更多信息."""
    cleaned: list[str] = []
    for slot in slots:
        token = slot.strip()
        if not token or token in _GENERIC_MISSING_SLOTS:
            continue
        # 太短的也丢
        if len(token) < 4:
            continue
        cleaned.append(token)
    return cleaned[:8]


def _normalize_conclusion_assistant_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    required_fields = tuple(field for field in CONCLUSION_SCHEMA_FIELDS if field != "reflect_tokens")
    if any(field not in payload for field in required_fields):
        return None
    extra_keys = set(payload) - set(CONCLUSION_SCHEMA_FIELDS)
    if extra_keys:
        return None
    question = str(payload.get("question") or "").strip()
    next_action = str(payload.get("next_action") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    missing_slots = _normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()

    if not question or next_action not in RETRIEVAL_ACTIONS:
        return None
    if next_action in {"answer_directly", "abstain"} and not answer:
        return None
    if next_action == "clarify_user" and not clarification_question:
        return None
    if next_action == "retrieve_more":
        if answer:
            return None
        missing_slots = _filter_missing_slots(missing_slots)
        # retrieve_more 必须有具体可检索缺口
        if not missing_slots:
            return None
    else:
        missing_slots = []
        if next_action != "clarify_user":
            clarification_question = ""

    # answer_directly / abstain：禁止暴露检索过程
    if next_action in {"answer_directly", "abstain"} and any(
        marker in answer for marker in _ANSWER_EXPOSURE_MARKERS
    ):
        return None

    # clarify_user：clarification_question 必须列出候选解读
    if next_action == "clarify_user":
        # 最低限度地要求长度 ≥ 10 且包含至少 1 个候选分隔符
        if len(clarification_question) < 10:
            return None
        has_options = any(sep in clarification_question for sep in ("？", "?", "、", "还是", "/", "："))
        if not has_options:
            return None

    reflect_tokens = _normalize_reflect_tokens(payload.get("reflect_tokens"))
    if reflect_tokens is None:
        reflect_tokens = _default_reflect_tokens_for_action(next_action)

    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
        "reflect_tokens": reflect_tokens,
    }


def make_sample_fingerprint(sample: dict) -> str:
    normalized_messages = []
    for message in sample.get("messages", []):
        normalized_messages.append(
            {
                "role": message.get("role"),
                "name": message.get("name"),
                "content": normalize_message_content(message.get("content")),
                "tool_calls": message.get("tool_calls"),
            }
        )
    payload = {
        "task_type": sample.get("task_type"),
        "messages": normalized_messages,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build_normalized_sample_id(
    *,
    request_id: str,
    task_type: str,
    index: int,
) -> str:
    return f"{request_id}-{task_type}-{index:04d}"


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(weight for _, weight in items)
    cursor = rng.random() * total
    seen = 0.0
    for name, weight in items:
        seen += weight
        if cursor <= seen:
            return name
    return items[-1][0]


def sample_evidence_documents(
    documents: list[dict],
    rng: random.Random,
    max_docs: int,
) -> list[dict]:
    count = max(1, rng.randint(1, max_docs))
    return rng.sample(documents, min(count, len(documents)))


def sample_worldbuilding_topic(
    topics: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    if not topics:
        raise ValueError("Worldbuilding generation requires at least one topic.")
    return dict(rng.choice(topics))


def format_evidence_pack(evidence_docs: list[dict]) -> str:
    blocks = []
    for index, doc in enumerate(evidence_docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or ''}",
                    f"stage_code: {doc.get('stage_code') or ''}",
                    f"avg_tag: {doc.get('avg_tag') or ''}",
                    f"source_path: {doc.get('source_path') or ''}",
                    "clean_text:",
                    doc.get("clean_text", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


HYPOTHESIS_KEYWORD_BLACKLIST = (
    "什么", "为什么", "为何", "怎么", "如何", "原因", "动机", "目的",
    "关系", "身份", "来历", "真相", "故事", "经历", "情况", "情节",
    "内容", "台词", "说了什么", "做了什么", "发生", "时候",
)

HYPOTHESIS_QUALITY_RUBRIC = """\
keywords 与 entities 的硬性约束（违反任一条该样本就要被丢弃）：

【entities 约束】
- 第一个元素必须是问题的主实体（人物/组织/地点/事件名），不允许是代词或描述性短语。
- 至少包含 1 个主实体；当问题涉及"X 与 Y 的关系/共同经历/对话"等多实体情景时，
  entities 必须 ≥ 2，并把所有相关实体都列出（如"凯尔希、阿米娅"而非只写"凯尔希"）。
- 若问题中只识别出 1 个实体（典型场景：问某干员的台词/语音/经历），entities 可以只 1 个，
  但 keywords 必须从证据中补 1 个共现高频的桥接实体或活动名。
- 禁止把代词（她/他/她们/他们/这位/那位/这个人/那个人）写入 entities。
- 禁止把"她们之间有什么故"、"什么要启动"、"么关系"这类问句残片当作实体。

【keywords 约束】
- 必须包含主实体本身。
- **黑名单（禁止出现以下整词）**：什么、为什么、为何、怎么、如何、原因、动机、目的、关系、身份、
  来历、真相、故事、经历、情况、情节、内容、台词、说了什么、做了什么、发生、时候。
  这些是问句词，对检索没有锚定价值。
- 长度 < 2 的 token 必须丢弃；纯英文 token 长度 < 3 也要丢弃。
- 推荐结构：[主实体, 别名1, 别名2, 同活动名/章节名, 上位类别词, 桥接实体, 关键事件词]，
  至少 5 个、最多 12 个。
- 不要重复堆叠近义词（如已经有"语音"就不要再写"台词内容"）。

【别名展开】
- 若主实体在下面的"已知干员别名"列表中出现，必须把 1-3 个高质量别名加入 keywords，
  例如「凯尔希 → 凯尔希医生」「W → 维什戴尔」「阿米娅 → 兔兔」。
- 列表里没出现的实体，不要凭空编造别名。
"""


def format_alias_hints(evidence_docs: list[dict], *, limit: int = 12) -> str:
    """Build a deterministic "candidate aliases" hint block for the teacher prompt."""
    try:
        alias_map = load_operator_alias_map(OPERATOR_ALIAS_MAP_PATH)
    except Exception:  # pragma: no cover - defensive
        alias_map = None
    if not alias_map:
        return "（未加载到别名表，本次不提示别名）"
    candidates = extract_primary_entity_candidates(evidence_docs, limit=limit)
    lines: list[str] = []
    seen: set[str] = set()
    for ent in candidates:
        aliases = alias_map.lookup(ent)
        if not aliases:
            continue
        key = ent.strip()
        if key in seen:
            continue
        seen.add(key)
        shown = "、".join(aliases[:4])
        lines.append(f"- {ent} → {shown}")
        if len(lines) >= limit:
            break
    if not lines:
        return "（证据中未匹配到已知干员别名）"
    return "\n".join(lines)


def extract_primary_entity_candidates(
    evidence_docs: list[dict],
    *,
    limit: int = 8,
) -> list[str]:
    counts: dict[str, int] = {}
    for doc in evidence_docs:
        for segment in doc.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            speaker = str(segment.get("speaker") or "").strip()
            if speaker and speaker not in PRIMARY_ENTITY_STOP_WORDS:
                counts[speaker] = counts.get(speaker, 0) + 4

        clean_text = str(doc.get("clean_text") or "")[:200]
        for pattern in (CODE_NAME_RE, REAL_NAME_RE, OPERATOR_NAME_RE):
            for match in pattern.findall(clean_text):
                token = str(match).strip()
                if token and token not in PRIMARY_ENTITY_STOP_WORDS:
                    counts[token] = counts.get(token, 0) + 2

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:limit]]


def format_worldbuilding_topic(topic: dict[str, Any]) -> str:
    lines = [
        f"主题: {topic.get('topic') or ''}",
        f"目标: {topic.get('goal') or ''}",
    ]

    keywords = topic.get("keywords") or []
    if keywords:
        lines.append("关键词: " + ", ".join(str(item) for item in keywords))

    focus_points = topic.get("focus_points") or []
    if focus_points:
        lines.append("重点展开:")
        lines.extend(f"- {item}" for item in focus_points)

    avoid = topic.get("avoid") or []
    if avoid:
        lines.append("禁止触及:")
        lines.extend(f"- {item}" for item in avoid)

    return "\n".join(lines)


def build_latest_hypothesis_schema_example(
    *,
    question: str = "【格式示例，禁止复用】某角色的身份是什么？",
    intent: str = "plot_fact",
    entities: list[str] | None = None,
    keywords: list[str] | None = None,
    expected_answer_type: str = "身份关系",
    dialogue_context: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "intent": intent,
        "entities": entities or ["格式示例实体"],
        "keywords": keywords or ["格式示例实体", "身份", "来历"],
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
        "reflect_tokens": _default_reflect_tokens_for_action("retrieve_more"),
    }


def build_follow_up_hypothesis_schema_example(
    *,
    question: str = "【格式示例，禁止复用】某角色的身份是什么？",
    entities: list[str] | None = None,
    keywords: list[str] | None = None,
    expected_answer_type: str = "身份关系",
    dialogue_context: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "entities": entities or ["格式示例实体", "格式示例桥接实体"],
        "keywords": keywords or ["格式示例实体", "格式示例桥接实体", "身份", "关系"],
        "expected_answer_type": expected_answer_type,
        "dialogue_context": dialogue_context,
        "reflect_tokens": _default_reflect_tokens_for_action("retrieve_more"),
    }


def build_conclusion_schema_example(
    *,
    question: str = "【格式示例，禁止复用】某角色的身份是什么？",
    next_action: str = "retrieve_more",
    answer: str = "",
    missing_slots: list[str] | None = None,
    clarification_question: str = "",
) -> dict[str, Any]:
    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots or ["格式示例实体的身份线索", "格式示例实体的关系线索"],
        "clarification_question": clarification_question,
        "reflect_tokens": _default_reflect_tokens_for_action(next_action),
    }


def _schema_block(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_initial_hypothesis_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题，不要改写成别的问题。",
            "- `intent`: 问题的语义类型，只表示问题类型，不表示流程状态。可选值仅限 plot_fact、plot_reasoning、timeline、character_relation、event_summary、compare、persona_chat、out_of_scope。",
            "- `entities`: 当前问题里最核心、最值得用于检索的实体，一般是角色、组织、地点、事件名。第一个元素必须是主实体。",
            "- `keywords`: 比 entities 更宽的检索词，可以包含原词、同义改写、关系词、短语化检索扩展；必须包含主实体。",
            "- `expected_answer_type`: 期望最终答案的形态，例如事实问答、身份关系、原因/动机、时间线、过程解释。",
            "- `dialogue_context`: 多轮对话上下文，用于补指代和追问背景；如果没有多轮上下文，可以省略，系统会按空字符串处理。",
        ]
    )


def build_follow_up_hypothesis_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题，保持不变。",
            "- `entities`: 在上一轮实体基础上，补充本轮证据里出现的关键桥接对象；第一个元素必须保留主实体。",
            "- `keywords`: 面向下一轮检索的缩小范围关键词，优先加入关系词、称谓、桥接短语；必须包含主实体。",
            "- `expected_answer_type`: 继续沿用当前问题所需的答案形态，例如身份关系、原因/动机、时间线、过程解释。",
            "- `dialogue_context`: 多轮对话上下文；通常沿用上一轮上下文，没有则可为空字符串。",
            "- `intent`: 不在 assistant 输出中出现，默认继承上一轮 hypothesis 的 intent。",
        ]
    )


def build_conclusion_field_explanations() -> str:
    return "\n".join(
        [
            "- `question`: 用户当前原问题。",
            "- `next_action`: 当前证据下的下一步动作，只能是 answer_directly、retrieve_more、clarify_user、abstain。",
            "- `answer`: 当前阶段结论文本。answer_directly 或 abstain 时必须非空；retrieve_more 时必须为空字符串。",
            "- `missing_slots`: 当前证据还缺哪些具体可检索的信息缺口，主要在 retrieve_more 时使用。",
            "- `clarification_question`: 当问题本身有歧义时，向用户发出的澄清问题；仅 clarify_user 时必须非空。",
        ]
    )


def build_initial_hypothesis_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    primary_entity_candidates = extract_primary_entity_candidates(evidence_docs)
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 中间推理步骤的高质量 SFT 样本。"
        "当前只生成“初始假设文档生成”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )
    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 hypothesis_builder。"},
        {
            "role": "user",
            "content": "用户问题: 【格式示例，禁止复用】某角色的身份是什么？\n多轮上下文: 无\n请生成初始假设文档 JSON。",
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_latest_hypothesis_schema_example(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": INITIAL_HYPOTHESIS_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }
    requirements = [
        f"1. 只生成 `{INITIAL_HYPOTHESIS_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用初始 hypothesis schema：question、intent、entities、keywords、expected_answer_type、dialogue_context、reflect_tokens。",
        "5. assistant JSON 不允许出现任何额外字段；`reflect_tokens` 是 Self-RAG 反思 token，详见下方规则。",
        "6. intent 只能从以下集合中选择：" + "、".join(sorted(HYPOTHESIS_INTENTS)) + "。",
        "7. user prompt 应围绕“用户问题 + 多轮上下文 -> 初始假设文档 JSON”。",
        "8. assistant JSON 目标是服务检索，不是直接回答问题；不得把最终结论当作既定事实写死。",
        "9. 不要编造英文别名、Dr. 前缀、代号扩写、罗德岛职位推断或跨角色别名污染。",
        "10. 所有文本使用中文。",
        "11. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "12. 每条样本都必须围绕 1 到 2 个主实体构造问题；assistant JSON 的 `entities` 第一个元素必须是主实体。",
        "13. `keywords` 必须包含主实体，且必须遵守下方“keywords 与 entities 的硬性约束”（黑名单 + 别名展开）。",
        "14. 本批次中至少 30% 的样本必须在 user prompt 的“多轮上下文”里写入 1-2 轮真实的假对话（user/assistant 各一条），让训练分布覆盖多轮追问场景；其余可保留“无”。",
        "15. 当样本是多实体关系/共同经历类问题时，assistant JSON 的 entities 必须 ≥ 2。",
        "16. schema 和返回格式中的“【格式示例，禁止复用】”“格式示例实体”只是占位说明；实际样本必须自行基于证据包生成用户问题、entities 和 keywords，严禁复用占位文本。",
    ]
    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“初始假设文档生成”训练样本。\n\n"
        "建议主实体候选（优先围绕这些角色/称谓出题）:\n"
        + ("、".join(primary_entity_candidates) if primary_entity_candidates else "无")
        + "\n\n已知干员别名（请在 keywords 中适度展开 1-3 个）:\n"
        + format_alias_hints(evidence_docs)
        + "\n\n当前项目唯一合法的初始 hypothesis schema（仅展示字段结构，字段值是格式占位，禁止复用到样本中）:\n"
        + _schema_block(build_latest_hypothesis_schema_example())
        + "\n\n字段含义说明:\n"
        + build_initial_hypothesis_field_explanations()
        + "\n\n"
        + HYPOTHESIS_QUALITY_RUBRIC
        + "\n"
        + SELF_RAG_REFLECT_TOKEN_RUBRIC
        + "\n要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例（仅展示 JSON 包装结构，示例字段值禁止复用；实际 samples 必须自行生成 query/hypothesis）：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_follow_up_hypothesis_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    primary_entity_candidates = extract_primary_entity_candidates(evidence_docs)
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 多轮补充检索步骤的高质量 SFT 样本。"
        "当前只生成“多轮补充假设文档生成”样本。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )
    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 follow_up_hypothesis_builder。"},
        {
            "role": "user",
            "content": (
                "用户问题: 【格式示例，禁止复用】某角色的身份是什么？\n多轮上下文: 无\n当前假设文档(JSON): "
                + json.dumps(
                    build_latest_hypothesis_schema_example(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                '上一轮结论生成结果(JSON): {"question":"【格式示例，禁止复用】某角色的身份是什么？","next_action":"retrieve_more","answer":"","missing_slots":["格式示例实体的身份线索","格式示例实体的关系线索"],"clarification_question":""}\n'
                "历史生成结果: [第1轮 hypothesis 已定位到格式示例实体身份问题，但尚未补出桥接线索]\n"
                "历史检索上下文: [第1轮检索已使用格式示例实体相关查询，但仍缺关键关系信息]\n"
                "当前检索轮次: 第2轮 / 最多3轮\n当前证据: [...]\n"
                "当前未解点: 还不知道格式示例实体的关键身份与关系。\n请生成补充检索假设文档 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                build_follow_up_hypothesis_schema_example(),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }
    requirements = [
        f"1. 只生成 `{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用补充 hypothesis schema：question、entities、keywords、expected_answer_type、dialogue_context、reflect_tokens。",
        "5. assistant JSON 不允许出现 intent，follow-up hypothesis 必须继承上一轮 intent；`reflect_tokens` 是 Self-RAG 反思 token，详见下方规则。",
        "6. user prompt 必须包含：用户问题、多轮上下文、当前假设文档(JSON)、上一轮结论生成结果(JSON)、历史生成结果、历史检索上下文、当前证据、当前检索轮次。",
        "7. user prompt 里的 `当前假设文档(JSON)` 必须严格使用初始 hypothesis schema。",
        '8. 不要在 user prompt 或 assistant JSON 中使用 `character_name`、`appearances`、`known_info`、`relationship_hints`、`bridging_objects`、`constraints`、`aliases` 等旧字段或衍生字段。',
        "9. assistant JSON 应只生成更强的检索线索，不直接回答问题。",
        "10. assistant JSON 的 keywords 应体现缩小范围后的二次检索查询，并遵守下方“keywords 与 entities 的硬性约束”。",
        "11. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
        "12. follow-up hypothesis 必须保留上一轮主实体；assistant JSON 的 `entities` 第一个元素必须仍然是主实体；如果证据揭示了新的桥接实体（关键人物/组织/称谓），必须把它加入 entities，使 entities ≥ 2。",
        "13. `keywords` 必须保留主实体，并在其基础上补充：(a) 证据中出现的桥接实体或称谓；(b) 上一轮 missing_slots 里的关键短语；(c) 1-2 个干员别名（若主实体在“已知干员别名”列表中）。",
        "14. user prompt 的“当前证据”字段必须使用证据包中至少 6 段证据，保留 `[证据 N]` 结构以贴近线上多轮分布。",
        "15. schema 和返回格式中的“【格式示例，禁止复用】”“格式示例实体”只是占位说明；实际样本必须自行基于证据包生成用户问题、entities 和 keywords，严禁复用占位文本。",
    ]
    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“多轮补充假设文档生成”训练样本。\n\n"
        "建议主实体候选（优先围绕这些角色/称谓缩小检索范围）:\n"
        + ("、".join(primary_entity_candidates) if primary_entity_candidates else "无")
        + "\n\n已知干员别名（请在 keywords 中适度展开 1-3 个）:\n"
        + format_alias_hints(evidence_docs)
        + "\n\n当前项目唯一合法的初始 hypothesis schema（仅展示字段结构，字段值是格式占位，禁止复用到样本中）:\n"
        + _schema_block(build_latest_hypothesis_schema_example())
        + "\n\n当前项目唯一合法的 follow-up hypothesis schema（仅展示字段结构，字段值是格式占位，禁止复用到样本中）:\n"
        + _schema_block(build_follow_up_hypothesis_schema_example())
        + "\n\n字段含义说明:\n"
        + build_follow_up_hypothesis_field_explanations()
        + "\n\n"
        + HYPOTHESIS_QUALITY_RUBRIC
        + "\n"
        + SELF_RAG_REFLECT_TOKEN_RUBRIC
        + "\n要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例（仅展示 JSON 包装结构，示例字段值禁止复用；实际 samples 必须自行生成 query/hypothesis）：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包（共 "
        + str(len(evidence_docs))
        + " 段；user prompt 必须把其中至少 6 段写入“当前证据”字段以贴近线上分布）：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


CRAG_KNOWLEDGE_REFINEMENT_RUBRIC = """\
CRAG knowledge refinement 模拟（必须遵守）：

线上推理时会先把检索到的 chunk 按句切片、用 reranker 重新打分、丢掉低分句、得到"精炼证据包"，
然后再交给 conclusion_generator 决策。为了让训练分布与之对齐，user prompt 必须按下列结构组织
"当前证据"字段：

[原始证据](来自检索 top-k，可能含噪)：
[证据 1] ...
[证据 2] ...
...（至少 6 段）

[精炼证据](由 evidence quality 评估器打分后保留的高相关句子）：
[证据 N.句子a] ...
[证据 M.句子b] ...
...（3-6 条最相关的句子，每条都要标注来自哪条原始证据）

要求：
1. 精炼证据必须只来自原始证据中的句子，不能新增内容、不能改写。
2. 精炼证据应当覆盖 hypothesis 的核心实体、动机、关键事件；如果原始证据完全没有这些内容，
   精炼证据可以为空（即 "[精炼证据]: 无相关高分句子"），用于训练 abstain 行为。
3. 精炼证据必须保留它来自的原始证据编号（`[证据 N.句子k]`），便于反查。
4. 当样本是 answer_directly 时，精炼证据必须包含 answer 中的关键事实句；
   当样本是 retrieve_more / abstain 时，精炼证据可以为空或只含外围线索；
   当样本是 clarify_user 时，精炼证据可以保留多个互相冲突的候选解读。
"""


SELF_RAG_REFLECT_TOKEN_RUBRIC = """\
Self-RAG 反思 token（必须遵守）：

assistant JSON 必须额外输出 `reflect_tokens` 字段，是一个长度为 4 的对象，
分别评估当前轮检索/证据/答案的可靠性，用于训练 4B 学会主动反思：

{
  "Retrieve": "Yes" | "No" | "Continue",
  "Relevant": "Relevant" | "Irrelevant" | "Partial",
  "Supported": "Fully" | "Partially" | "NoSupport",
  "Useful": "Useful" | "PartiallyUseful" | "Useless"
}

判定规则：
- Retrieve：本轮是否需要继续检索。retrieve_more → "Yes"；answer_directly → "No"；clarify_user → "No"；abstain → "No"。
- Relevant：当前证据与问题相关度。证据中含主实体且能回答关键子问题 → "Relevant"；
  含主实体但答案缺失 → "Partial"；几乎无关 → "Irrelevant"。
- Supported：answer 中的关键事实在证据中的覆盖度。answer_directly 必须 "Fully"；
  abstain / retrieve_more / clarify_user 通常 "Partially" 或 "NoSupport"。
- Useful：当前证据对最终回答用户的实用度。answer_directly 通常 "Useful"；
  其它情况按实际证据贴合度判断。

`reflect_tokens` 是 assistant JSON 的顶层字段，禁止嵌套到其他字段里。
"""


CONCLUSION_DECISION_RUBRIC = """\
四种 next_action 的判定规则（必须严格遵守，每种都要有训练样本）：

【answer_directly】（推荐占比约 30%）
- 触发条件：证据包中至少存在 1 段 evidence 的 clean_text 明确含有问题核心实体，
  且能用 1-3 句话直接给出答案；不存在指代歧义；不需要补充其他子问题。
- answer 字段：先给结论（"X 是 Y" / "原因是…"），再 1-2 句简短补充证据要点。
- answer 必须忠于证据：如果证据只说"疑似/暗示/有人认为"，answer 也要保留不确定性，禁止把推测写成确定事实。
- 答案中出现的关键实体、关系词、动作词，必须都能在证据 clean_text 中找到字面或近义对应。

【retrieve_more】（推荐占比约 40%）
- 触发条件：证据与问题主题相关，但缺少关键桥接信息：缺人物身份、缺动机、缺结果、缺时间、
  缺关键道具、缺事件起因，且这些缺口是可以被后续检索补上的（有明确实体可查）。
- answer 必须为空字符串。
- missing_slots：必须是 2-5 个具体的可检索语义缺口短语，例如「某主实体的真实身份」「某主实体与桥接实体的关系」；
  禁止写"更多信息""相关背景""详细资料"这种空泛短语。
- 每个 missing_slot 必须以具体实体或具体事件命名，能直接转化成下一轮检索 query。

【clarify_user】（推荐占比约 15%，**必须主动构造**）
- 触发条件（任一即可）：
  1. 用户问题里含代词（她/他/她们/他们/它/这位/那位/这个人/那个人/这件事/那件事），
     且多轮上下文不足以唯一确定指代对象。
  2. 用户问题含同名/重名实体（如"博士"、"魔王"、"戈尔丁"、"陈"等可能指多人）。
  3. 用户问题范围过宽（问"她的故事"但未限定哪一段经历）或语义有多种合理解读。
- clarification_question 必须列出 2-4 个候选解读供用户选择，禁止只说"请澄清"。
- answer 必须为空字符串；missing_slots 必须为空数组。

【abstain】（推荐占比约 15%，**必须主动构造**）
- 触发条件：证据与问题核心实体几乎完全不相关（证据中没有出现问题中的核心实体或其等价别名），
  或多轮重试后仍无任何桥接线索，或问题超出剧情可知范围。
- answer 字段：必须明确说"现有检索证据不足以确认…"，并说明缺失的关键事实是什么，
  禁止用"根据证据""根据剧情"这种暴露过程的说法。
- missing_slots 可为空数组，clarification_question 必须为空。
- 不要把"伪 abstain"当成真 abstain：如果证据明显含答案，但你只是没好好读，不能 abstain。
"""


def build_conclusion_prompt_bundle(
    *,
    evidence_docs: list[dict[str, Any]],
    samples_per_request: int,
) -> tuple[str, str]:
    system_prompt = (
        "你是一个严格的中文教师模型数据合成器。"
        "你的任务是生成专门训练《明日方舟》剧情问答 Agent 结论生成步骤的高质量 SFT 样本。"
        "当前只生成“结论生成”样本。"
        "你必须主动覆盖 4 种 next_action（answer_directly / retrieve_more / clarify_user / abstain），"
        "禁止全部偏向 retrieve_more。"
        "assistant 输出必须是严格 JSON，不要输出任何额外说明。"
    )
    example_hypothesis = build_latest_hypothesis_schema_example(
        question="【格式示例，禁止复用】某角色的身份是什么？",
        intent="plot_fact",
        entities=["格式示例实体"],
        keywords=["格式示例实体", "身份", "来历"],
    )
    example_conclusion = build_conclusion_schema_example(
        question="【格式示例，禁止复用】某角色的身份是什么？",
        next_action="retrieve_more",
        answer="",
        missing_slots=["格式示例实体的身份线索", "格式示例实体的关系线索"],
        clarification_question="",
    )
    message_example = [
        {"role": "system", "content": "你是《明日方舟》剧情问答系统中的 conclusion_generator。"},
        {
            "role": "user",
            "content": (
                "用户问题: 【格式示例，禁止复用】某角色的身份是什么？\n多轮上下文: 无\n当前假设文档(JSON): "
                + json.dumps(
                    example_hypothesis,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
                "历史检索上下文: [第1轮已检索格式示例实体相关片段，但仍缺关键桥接信息]\n"
                "当前检索轮次: 第2轮 / 最多3轮\n当前证据: [...]\n"
                "请基于证据生成当前阶段结论 JSON。"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(
                example_conclusion,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]
    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": CONCLUSION_TASK_TYPE,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "decision_case": "answer_directly|retrieve_more|clarify_user|abstain",
                    "notes": "string",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                },
            }
        ]
    }
    requirements = [
        f"1. 只生成 `{CONCLUSION_TASK_TYPE}` 类型样本。",
        "2. 每条样本都只允许出现 `system`、`user`、`assistant` 三种 role，不要使用 tool_calls。",
        "3. assistant 的 content 必须是单个 JSON 对象，不要带 markdown 代码块，不要输出解释。",
        "4. assistant JSON 必须严格使用：question、next_action、answer、missing_slots、clarification_question、reflect_tokens。",
        "5. next_action 只能是 `answer_directly`、`retrieve_more`、`clarify_user`、`abstain`。",
        f"6. 本次必须生成 {samples_per_request} 条样本，**4 种 next_action 都要有**：尽量按 answer_directly ≈ 30%、retrieve_more ≈ 40%、clarify_user ≈ 15%、abstain ≈ 15% 的比例覆盖（单次只生成 1 条则按需挑选最贴合证据的类型，但整体多次调用必须均衡）。",
        "7. 每条样本必须**自行生成新的用户问题 query**，并同步生成匹配该 query 的当前假设文档；query 必须来自本次证据包中的真实实体、事件、地点、组织、道具或关系，不能复用下面 schema/返回格式示例里的任何字段值。",
        "8. 严禁复用格式示例内容：不得输出“【格式示例，禁止复用】”、不得照抄“某角色的身份是什么？”、“格式示例实体”、“格式示例实体的身份线索”等占位文本；如果样本中出现这些占位文本，该样本视为无效。",
        "9. 同一次返回的多条样本中，用户问题 question 必须互不相同；不要只替换代词或标点来制造伪变化。",
        "10. user prompt 必须包含：用户问题、多轮上下文、当前假设文档(JSON)、当前证据；当前假设文档不能是空对象 {}。",
        "11. conclusion prompt 必须显式带出当前检索轮次与历史检索上下文。",
        "12. user prompt 的“当前证据”字段必须按 CRAG 双段结构输出（原始证据 + 精炼证据），原始段至少 6 段且不超过 12 段，精炼段为 3-6 个高分句子；保留 `[证据 N]` / `[证据 N.句子k]` 编号便于反查。",
        "13. 若要构造 clarify_user 样本，必须主动在用户问题里写入代词（她/他/这位 等）或同名实体（如\"博士\"、\"戈尔丁\"、\"魔王\"、\"陈\"等），并把多轮上下文留空或刻意只给少量信息使指代无法唯一确定。",
        "14. 若要构造 abstain 样本，必须挑选证据中**核心实体与用户问题完全不重叠**的子集（例如证据是另一活动的片段、或只剩进驻设施语音之类的无关片段），让结论无法成立。",
        "15. answer_directly / abstain 的 answer 文本绝对不允许出现\"根据证据\"\"根据剧情\"\"根据检索\"\"基于证据\"\"从证据中\"\"根据上面\"等暴露检索过程的措辞。",
        "16. answer_directly 时 answer 中出现的所有关键实体名、关系词、地点、事件，都必须能在“精炼证据”原文中找到字面对应或显著近义；不允许只在 hypothesis.keywords 或 raw 段中出现却不在精炼证据中的内容。",
        "17. retrieve_more 时 missing_slots 必须列出 2-5 个具体可检索缺口；禁止使用\"更多信息\"\"相关背景\"\"详细资料\"\"完整剧情\"等空泛词。",
        "18. clarify_user 时 clarification_question 必须列出 2-4 个候选解读，例如 \"您指的是 A、B 还是 C？\"；不允许只说\"请澄清\"。",
        "19. abstain 时 answer 第一句必须是 \"现有检索证据不足以确认…\" 模板的变体；必须显式指出**缺哪个关键事实**，禁止笼统说\"不知道\"。",
        "20. 结论生成样本必须显式体现“基于当前证据是否足够作答”的判断；不要随意把 answer_directly 与 retrieve_more 互换。",
        "21. assistant JSON 必须额外输出 `reflect_tokens` 对象（Self-RAG 反思 token），字段值参考下方 Self-RAG 规则。",
        "22. 所有文本使用中文，不要编造英文别名或 Dr. 前缀。",
        "23. 顶层返回格式必须是一个 JSON 对象，且只有 `samples` 字段。",
    ]
    user_prompt = (
        f"请基于下面证据生成 {samples_per_request} 条“结论生成”训练样本。\n\n"
        "当前项目唯一合法的初始 hypothesis schema（仅展示字段结构，字段值是格式占位，禁止复用到样本中）:\n"
        + _schema_block(example_hypothesis)
        + "\n\n当前项目唯一合法的 conclusion schema（仅展示字段结构，字段值是格式占位，禁止复用到样本中）:\n"
        + _schema_block(example_conclusion)
        + "\n\n字段含义说明:\n"
        + build_conclusion_field_explanations()
        + "\n\n"
        + CONCLUSION_DECISION_RUBRIC
        + "\n"
        + CRAG_KNOWLEDGE_REFINEMENT_RUBRIC
        + "\n"
        + SELF_RAG_REFLECT_TOKEN_RUBRIC
        + "\n要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例（仅展示 JSON 包装结构，示例字段值禁止复用；实际 samples 必须自行生成 query/hypothesis/conclusion）：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n证据包（共 "
        + str(len(evidence_docs))
        + " 段；user prompt 必须把其中至少 6 段写入“当前证据·原始证据”字段，并基于这些原始证据派生 3-6 条精炼句进入“当前证据·精炼证据”字段以贴近线上 CRAG 分布）：\n"
        + format_evidence_pack(evidence_docs)
    )
    return system_prompt, user_prompt


def build_teacher_prompts(
    *,
    task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    samples_per_request: int,
) -> tuple[str, str]:
    task_type = normalize_task_type(task_type)
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        return build_initial_hypothesis_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )
    if task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
        return build_follow_up_hypothesis_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )
    if task_type == CONCLUSION_TASK_TYPE:
        return build_conclusion_prompt_bundle(
            evidence_docs=evidence_docs,
            samples_per_request=samples_per_request,
        )

    if task_type == "worldbuilding_qa":
        system_prompt = (
            "你是一个严格的中文教师模型数据合成器。"
            "你的任务是基于给定的《明日方舟》世界观主题生成高质量 SFT 训练样本。"
            "这些样本用于通用世界观注入，而不是复述具体剧情桥段。"
            "允许使用稳定、通行、跨剧情复用的官方世界观常识，但不要引入冷门细节、争议推断、二创设定或未经确认的数字。"
            "必须返回严格 JSON，不要输出任何额外说明。"
        )
    else:
        system_prompt = (
            "你是一个严格的中文教师模型数据合成器。"
            "你的任务是基于给定的《明日方舟》剧情证据生成高质量 SFT 训练样本。"
            "只能使用证据中明确出现或可直接归纳的信息，不得编造设定。"
            "必须返回严格 JSON，不要输出任何额外说明。"
        )

    task_specific_rules = {
        "canon_qa": "生成剧情事实问答，答案必须能从证据直接支持。",
        "worldbuilding_qa": "生成通用世界观问答，重点是概念解释、制度背景、社会结构或通用设定，不要依赖具体剧情片段。",
        "persona_grounded_qa": "生成带有澄闪语气但事实严格受证据约束的问答，语气轻柔、克制、礼貌，不要卖萌过度。",
        "multi_turn_dialogue": "生成 2 到 4 轮多轮对话，包含追问、指代消解或澄清，至少出现 2 次 user 发言和 2 次 assistant 发言。",
    }[task_type]

    if task_type == "worldbuilding_qa":
        message_example = [
            {"role": "system", "content": "optional string"},
            {"role": "user", "content": "请先用一句话解释这个设定。"},
            {"role": "assistant", "content": "不超过100字的简短回答"},
            {"role": "user", "content": "那它通常会怎样影响社会或地区？"},
            {"role": "assistant", "content": "不超过100字的简短追问回答"},
            {"role": "user", "content": "再补一句最容易混淆的点。"},
            {"role": "assistant", "content": "不超过100字的简短澄清回答"},
        ]
    else:
        message_example = [
            {"role": "system", "content": "optional string"},
            {"role": "user", "content": "string"},
            {"role": "assistant", "content": "string"},
        ]

    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": task_type,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                    "worldbuilding_topic": "string or null",
                    "notes": "string",
                },
            }
        ]
    }

    requirements = [
        f"1. {task_specific_rules}",
        "2. `messages` 里的 role 只允许是 `system`、`user`、`assistant`。",
        "3. 所有文本使用中文。",
        "4. 输出必须是一个 JSON 对象，顶层只有 `samples` 字段。",
    ]

    if task_type == "worldbuilding_qa":
        requirements.extend(
            [
                "5. 不要引用具体剧情桥段、对白原句、关卡编号、活动标题或角色私密动机。",
                "6. 优先生成可跨剧情复用的定义、解释、比较、概念辨析、常识性制度描述等样本。",
                "7. 如果某个细节不够稳定，就使用保守表达，例如“通常”“一般而言”“常被视为”，不要编造精确设定。",
                "8. 可以生成“这个主题需要区分地区差异或组织差异”的问答，但不要虚构未被广泛接受的细节。",
                "9. 每条样本都必须是多轮对话，至少包含 3 次 `user` 发言和 3 次 `assistant` 发言，且后续轮次必须基于前一轮追问展开。",
                "10. 每次 `assistant` 回答都必须是短答，不超过 100 个中文字符。",
                "11. 每条样本只围绕当前主题包里的一个核心设定展开，不要把多个国家、多个生物体系混在同一条样本里。",
                "12. 不要使用 `tool_calls`，也不要输出 `tool` 角色。",
            ]
        )
        source_block = "世界观主题包：\n" + format_worldbuilding_topic(worldbuilding_topic or {})
    else:
        requirements.extend(
            [
                "5. 只允许使用证据中的信息；如果证据不足，就生成“需要澄清”或“证据不足”的样本。",
                "6. 不要使用 `tool_calls`，也不要输出 `tool` role。",
                "7. `multi_turn_dialogue` 必须体现上下文延续，不要把多个单轮问答拼在一起。",
                "8. `canon_qa` 与 `persona_grounded_qa` 的答案必须能被证据直接支持，不要把推测写成事实。",
                "9. 最终 assistant 不要出现“根据检索到的剧情证据”“根据检索结果”“根据证据”等暴露检索过程的措辞。",
            ]
        )
        source_block = "证据包：\n" + format_evidence_pack(evidence_docs)

    user_prompt = (
        f"请基于下面给定的材料生成 {samples_per_request} 条 `{task_type}` 类型训练样本。\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n"
        + source_block
    )

    return system_prompt, user_prompt


def call_teacher_api(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict]:
    api_key = os.environ.get(api_config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing teacher API key env var: {api_config.api_key_env}"
        )

    headers = {"Content-Type": "application/json"}
    if api_config.auth_header in {"bearer", "both"}:
        headers["Authorization"] = f"Bearer {api_key}"
    if api_config.auth_header in {"x-api-key", "both"}:
        headers["x-api-key"] = api_key
    if api_config.extra_headers:
        headers.update(api_config.extra_headers)
    for header_name, header_value in headers.items():
        try:
            str(header_value).encode("latin-1")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                f"HTTP header {header_name!r} contains non-latin-1 characters. "
                "Check the API key/env value and auth header configuration; do not use Chinese placeholders."
            ) from exc

    if api_config.api_type == "chat_completions":
        url = api_config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": api_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": api_config.temperature,
            "max_tokens": api_config.max_output_tokens,
        }
        if api_config.json_mode:
            payload["response_format"] = {"type": "json_object"}
    elif api_config.api_type == "anthropic_messages":
        url = api_config.base_url.rstrip("/") + "/messages"
        headers.setdefault("anthropic-version", "2023-06-01")
        payload = {
            "model": api_config.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            "temperature": api_config.temperature,
            "max_tokens": api_config.max_output_tokens,
        }
        if api_config.anthropic_disable_thinking:
            payload["thinking"] = {"type": "disabled"}
    elif api_config.api_type == "responses":
        url = api_config.base_url.rstrip("/") + "/responses"
        payload = {
            "model": api_config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": api_config.temperature,
            "max_output_tokens": api_config.max_output_tokens,
        }
    else:
        raise ValueError(f"Unsupported api_type: {api_config.api_type}")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=api_config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Teacher API HTTPError {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Teacher API URLError: {exc}") from exc

    decoded = json.loads(raw)
    try:
        text = extract_response_text(api_config.api_type, decoded)
    except Exception as exc:
        preview = json.dumps(decoded, ensure_ascii=False)[:2000]
        raise ValueError(
            f"Could not extract text content from API response: {exc}; "
            f"payload_preview={preview}"
        ) from exc
    return text, decoded


def _extract_jsonish_from_text(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    match = JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not ("entities" in text and "relations" in text):
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text if text.startswith("{") else ""


def extract_response_text(api_type: str, payload: dict) -> str:
    if api_type == "chat_completions":
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("No choices in chat completion payload")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get("text") or item.get("content") or "")
            return "\n".join(texts).strip()
        raise ValueError("Unsupported message content shape")

    if api_type == "anthropic_messages":
        if isinstance(payload.get("completion"), str) and payload["completion"].strip():
            return payload["completion"]
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"]
        content = payload.get("content") or []
        texts: list[str] = []
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or ""
            if text:
                return str(text)
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    texts.append(str(text))
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts).strip()
        thinking_blocks = [
            str(item.get("thinking") or "")
            for item in content
            if isinstance(item, dict) and item.get("thinking")
        ]
        if thinking_blocks:
            for thinking in thinking_blocks:
                jsonish = _extract_jsonish_from_text(thinking)
                if jsonish:
                    return jsonish
            raise ValueError(
                "Anthropic payload contains only thinking blocks and no final text. "
                "Disable thinking for JSON extraction or increase max_tokens."
            )
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            message_content = message.get("content")
            if isinstance(message_content, str) and message_content.strip():
                return message_content
            if isinstance(message_content, list):
                for item in message_content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content") or ""
                        if text:
                            texts.append(str(text))
                if texts:
                    return "\n".join(texts).strip()
        raise ValueError("No text in anthropic messages payload")

    if api_type == "responses":
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"]
        output = payload.get("output") or []
        texts: list[str] = []
        for item in output:
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict):
                    text = (
                        part.get("text")
                        or part.get("output_text")
                        or part.get("content")
                        or ""
                    )
                    if text:
                        texts.append(text)
        if texts:
            return "\n".join(texts).strip()
        raise ValueError("No text in responses payload")

    raise ValueError(f"Unsupported api_type: {api_type}")


def parse_teacher_json(text: str) -> dict:
    candidate = text.strip()
    match = JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    return json.loads(candidate)


def categorize_task_type(task_type: str) -> str:
    return TASK_BUCKET_MAP.get(task_type, "knowledge")


def _normalize_tool_calls(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    clean_calls: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name") or ""
        arguments = function.get("arguments") or ""
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = None
            if isinstance(parsed_arguments, dict):
                if name == "build_hypothesis" and "intent" in parsed_arguments:
                    parsed_arguments["intent"] = _normalize_intent(parsed_arguments.get("intent"))
                    if not parsed_arguments["intent"]:
                        continue
                if name == "detect_intent" and "intent" in parsed_arguments:
                    parsed_arguments["intent"] = _normalize_intent(parsed_arguments.get("intent"))
                    if not parsed_arguments["intent"]:
                        parsed_arguments.pop("intent", None)
                arguments = json.dumps(parsed_arguments, ensure_ascii=False, separators=(",", ":"))
        clean_calls.append(
            {
                "id": item.get("id") or "call_1",
                "type": item.get("type") or "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return clean_calls


def _has_non_empty_content(message: dict) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return content is not None


def _validate_multi_turn_messages(messages: list[dict]) -> bool:
    user_turns = sum(1 for message in messages if message["role"] == "user")
    assistant_turns = sum(1 for message in messages if message["role"] == "assistant")
    if user_turns < 2 or assistant_turns < 2:
        return False
    if any(message["role"] == "tool" for message in messages):
        return False
    return True


def _validate_worldbuilding_messages(messages: list[dict]) -> bool:
    user_turns = sum(1 for message in messages if message["role"] == "user")
    assistant_messages = [
        message
        for message in messages
        if message["role"] == "assistant" and _has_non_empty_content(message)
    ]
    if user_turns < 3 or len(assistant_messages) < 3:
        return False
    if any(message["role"] == "tool" for message in messages):
        return False
    for message in assistant_messages:
        content = normalize_message_content(message.get("content"))
        if len(content) > 100:
            return False
    return True


def _final_assistant_message(messages: list[dict]) -> dict | None:
    return next(
        (
            messages[index]
            for index in range(len(messages) - 1, -1, -1)
            if messages[index]["role"] == "assistant"
            and _has_non_empty_content(messages[index])
        ),
        None,
    )


def _final_answer_hides_retrieval(messages: list[dict]) -> bool:
    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False
    final_text = normalize_message_content(final_assistant.get("content"))
    return not any(marker in final_text for marker in FORBIDDEN_FINAL_ANSWER_MARKERS)


def _tool_call_function_name(message: dict) -> str | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    function = first_call.get("function")
    if not isinstance(function, dict):
        return None
    return function.get("name")


def _tool_call_arguments(message: dict) -> dict | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    function = first_call.get("function")
    if not isinstance(function, dict):
        return None
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def _validate_tool_chain_messages(messages: list[dict]) -> bool:
    expected_tool_names = [
        "detect_intent",
        "build_hypothesis",
        "retrieve_story_context",
    ]
    cursor = -1

    first_user_index = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        None,
    )
    if first_user_index is None:
        return False
    cursor = first_user_index

    for tool_name in expected_tool_names:
        assistant_index = next(
            (
                index
                for index in range(cursor + 1, len(messages))
                if messages[index]["role"] == "assistant"
                and _tool_call_function_name(messages[index]) == tool_name
            ),
            None,
        )
        if assistant_index is None:
            return False

        arguments = _tool_call_arguments(messages[assistant_index])
        if arguments is None:
            return False

        tool_index = next(
            (
                index
                for index in range(assistant_index + 1, len(messages))
                if messages[index]["role"] == "tool"
                and messages[index].get("name") == tool_name
            ),
            None,
        )
        if tool_index is None:
            return False

        if tool_name == "detect_intent":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
        elif tool_name == "build_hypothesis":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
            if not isinstance(arguments.get("intent"), str) or not arguments["intent"].strip():
                return False
        elif tool_name == "retrieve_story_context":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
            if not isinstance(arguments.get("hypothesis"), str) or not arguments["hypothesis"].strip():
                return False
            keywords = arguments.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                return False
            top_k = arguments.get("top_k")
            if not isinstance(top_k, int) or top_k <= 0:
                return False

        cursor = tool_index

    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False
    final_index = messages.index(final_assistant)
    if final_index <= cursor:
        return False
    return _final_answer_hides_retrieval(messages)


def _validate_unknown_negative_messages(messages: list[dict]) -> bool:
    if not _validate_tool_chain_messages(messages):
        return False

    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False

    final_text = normalize_message_content(final_assistant.get("content"))
    if not any(marker in final_text for marker in INSUFFICIENT_EVIDENCE_MARKERS):
        return False

    tool_messages = [message for message in messages if message["role"] == "tool"]
    tool_text = "\n".join(normalize_message_content(message.get("content")) for message in tool_messages)
    weak_tool_markers = (
        "[]",
        "空",
        "低相关",
        "无相关",
        "没有检索到",
        "不足以支持",
        "无法支持",
        "证据不足",
        "未找到",
    )
    return any(marker in tool_text for marker in weak_tool_markers)


def validate_and_normalize_samples(
    payload: dict,
    *,
    expected_task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    request_id: str,
) -> list[dict]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Teacher payload must contain a list field named 'samples'")

    expected_task_type = normalize_task_type(expected_task_type)

    source_story_ids = [doc.get("story_id") for doc in evidence_docs if doc.get("story_id")]
    source_stage_codes = [doc.get("stage_code") for doc in evidence_docs if doc.get("stage_code")]
    source_activity_names = [doc.get("activity_name") for doc in evidence_docs if doc.get("activity_name")]

    normalized: list[dict] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        task_type = normalize_task_type(sample.get("task_type") or expected_task_type)
        if task_type not in SUPPORTED_TASK_TYPES:
            continue
        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue

        if task_type in {
            INITIAL_HYPOTHESIS_TASK_TYPE,
            FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
            CONCLUSION_TASK_TYPE,
        }:
            if task_type != expected_task_type or len(messages) < 3:
                continue
            clean_messages: list[dict[str, Any]] = []
            valid = True
            for message in messages:
                if not isinstance(message, dict):
                    valid = False
                    break
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    valid = False
                    break
                if message.get("tool_calls") or role == "tool":
                    valid = False
                    break
                clean_messages.append(
                    {"role": role, "content": content if isinstance(content, str) else ""}
                )
            if not valid:
                continue
            if not any(message["role"] == "system" for message in clean_messages):
                continue
            user_message = next(
                (message for message in clean_messages if message["role"] == "user"),
                None,
            )
            if user_message is None:
                continue
            if task_type in {
                FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                CONCLUSION_TASK_TYPE,
            } and _contains_empty_current_hypothesis(user_message["content"]):
                continue
            if _contains_legacy_prompt_hypothesis_schema(user_message["content"]):
                continue
            final_assistant = next(
                (
                    message
                    for message in reversed(clean_messages)
                    if message["role"] == "assistant"
                    and str(message.get("content") or "").strip()
                ),
                None,
            )
            if final_assistant is None:
                continue
            assistant_payload = _parse_json_content(final_assistant.get("content"))
            if assistant_payload is None:
                continue
            normalized_assistant_payload: dict[str, Any] | None = None
            if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
                normalized_assistant_payload = _normalize_initial_hypothesis_assistant_payload(
                    assistant_payload
                )
            elif task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
                normalized_assistant_payload = _normalize_follow_up_hypothesis_assistant_payload(
                    assistant_payload
                )
            elif task_type == CONCLUSION_TASK_TYPE:
                normalized_assistant_payload = _normalize_conclusion_assistant_payload(
                    assistant_payload
                )
            if normalized_assistant_payload is None:
                continue
            final_assistant["content"] = json.dumps(
                normalized_assistant_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
            normalized.append(
                {
                    "id": build_normalized_sample_id(
                        request_id=request_id,
                        task_type=task_type,
                        index=index,
                    ),
                    "task_type": task_type,
                    "bucket": "tool",
                    "messages": clean_messages,
                    "tools": [],
                    "meta": {
                        "category": "tool",
                        "grounded": True,
                        "difficulty": meta.get("difficulty") or "medium",
                        "notes": meta.get("notes") or "",
                        "source_story_ids": meta.get("source_story_ids") or source_story_ids,
                        "source_stage_codes": meta.get("source_stage_codes") or source_stage_codes,
                        "source_activity_names": meta.get("source_activity_names")
                        or source_activity_names,
                        "task_family": (
                            "hypothesis_generation"
                            if task_type
                            in {INITIAL_HYPOTHESIS_TASK_TYPE, FOLLOW_UP_HYPOTHESIS_TASK_TYPE}
                            else "conclusion_generation"
                        ),
                        "decision_case": normalized_assistant_payload.get("next_action")
                        if task_type == CONCLUSION_TASK_TYPE
                        else None,
                        "worldbuilding_topic": None,
                        "generation_mode": "evidence_grounded",
                        "request_id": request_id,
                    },
                }
            )
            continue

        clean_messages = []
        valid = True

        for message in messages:
            if not isinstance(message, dict):
                valid = False
                break
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                valid = False
                break
            clean_message = {
                "role": role,
                "content": message.get("content", ""),
            }
            if role == "tool":
                clean_message["name"] = message.get("name") or "retrieve_story_context"
            if role == "assistant" and message.get("tool_calls"):
                clean_tool_calls = _normalize_tool_calls(message.get("tool_calls"))
                if clean_tool_calls:
                    clean_message["tool_calls"] = clean_tool_calls
            clean_messages.append(clean_message)

        if not valid:
            continue
        for clean_message in clean_messages:
            if clean_message["role"] != "tool":
                continue
            tool_name = clean_message.get("name")
            if tool_name == "build_hypothesis":
                try:
                    tool_payload = json.loads(str(clean_message.get("content") or "{}"))
                except json.JSONDecodeError:
                    valid = False
                    break
                if not isinstance(tool_payload, dict):
                    valid = False
                    break
                normalized_payload = _normalize_hypothesis_tool_payload(tool_payload)
                if normalized_payload is None:
                    valid = False
                    break
                clean_message["content"] = json.dumps(
                    normalized_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            elif tool_name == "detect_intent":
                try:
                    tool_payload = json.loads(str(clean_message.get("content") or "{}"))
                except json.JSONDecodeError:
                    valid = False
                    break
                if not isinstance(tool_payload, dict):
                    valid = False
                    break
                if "intent" in tool_payload:
                    normalized_intent = _normalize_intent(tool_payload.get("intent"))
                    if not normalized_intent:
                        valid = False
                        break
                    tool_payload["intent"] = normalized_intent
                clean_message["content"] = json.dumps(
                    tool_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
        if not valid:
            continue
        if any(msg.get("tool_calls") for msg in clean_messages):
            continue
        if any(msg["role"] == "tool" for msg in clean_messages):
            continue
        if task_type == "worldbuilding_qa" and not _validate_worldbuilding_messages(clean_messages):
            continue
        if task_type == "multi_turn_dialogue" and not _validate_multi_turn_messages(clean_messages):
            continue
        # style/knowledge 类样本不允许任何 assistant 消息出现"根据证据/根据剧情/检索到的"等检索过程暴露词
        if task_type in {"canon_qa", "persona_grounded_qa", "multi_turn_dialogue", "worldbuilding_qa"}:
            exposed = False
            for msg in clean_messages:
                if msg.get("role") != "assistant":
                    continue
                text = str(msg.get("content") or "")
                if any(marker in text for marker in _ANSWER_EXPOSURE_MARKERS):
                    exposed = True
                    break
            if exposed:
                continue

        meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
        normalized_sample = {
            "id": build_normalized_sample_id(
                request_id=request_id,
                task_type=task_type,
                index=index,
            ),
            "task_type": task_type,
            "bucket": categorize_task_type(task_type),
            "messages": clean_messages,
            "tools": [],
            "meta": {
                "category": categorize_task_type(task_type),
                "grounded": True,
                "difficulty": meta.get("difficulty") or "medium",
                "notes": meta.get("notes") or "",
                "source_story_ids": meta.get("source_story_ids") or source_story_ids,
                "source_stage_codes": meta.get("source_stage_codes") or source_stage_codes,
                "source_activity_names": meta.get("source_activity_names") or source_activity_names,
                "worldbuilding_topic": meta.get("worldbuilding_topic")
                or (worldbuilding_topic.get("topic") if worldbuilding_topic else None),
                "generation_mode": (
                    "topic_driven" if task_type == "worldbuilding_qa" else "evidence_grounded"
                ),
                "request_id": request_id,
            },
        }
        normalized.append(normalized_sample)
    return normalized


def dedupe_samples(samples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for sample in samples:
        fingerprint = make_sample_fingerprint(sample)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        sample["fingerprint"] = fingerprint
        unique.append(sample)
    return unique


def split_samples(
    samples: list[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio)
    val_cut = train_cut + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_cut],
        "val": shuffled[train_cut:val_cut],
        "test": shuffled[val_cut:],
    }


def build_request_record(
    *,
    request_id: str,
    task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    evidence_mode: str | None,
    retrieval_query: str | None,
    retrieval_seed_doc_id: str | None,
    system_prompt: str,
    user_prompt: str,
    raw_text: str | None,
    parsed_ok: bool,
    accepted_samples: int,
    latency_seconds: float,
    error: str | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "task_type": task_type,
        "bucket": categorize_task_type(task_type),
        "evidence_doc_ids": [doc["id"] for doc in evidence_docs],
        "evidence_mode": evidence_mode,
        "retrieval_query": retrieval_query,
        "retrieval_seed_doc_id": retrieval_seed_doc_id,
        "worldbuilding_topic": worldbuilding_topic,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_text": raw_text,
        "parsed_ok": parsed_ok,
        "accepted_samples": accepted_samples,
        "latency_seconds": latency_seconds,
        "error": error,
        "created_at": int(time.time()),
    }
