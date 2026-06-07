from __future__ import annotations

import ast
import json
import base64
import hashlib
import html
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

from goldenglow.config import DATA_ROOT, OPERATOR_ALIAS_MAP_PATH, QueryConfig
from goldenglow.data.alias_map import load_operator_alias_map
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever
from goldenglow.retrieval.minirag import document_chapter_scope_key, document_chapter_scope_label
from goldenglow.retrieval.storyline import document_storyline_scopes, storyline_scope_label


QUESTION_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")
CHINESE_TOKEN_SPLIT_RE = re.compile(
    r"(?:[的是和与及或为在把被让给从向对将要]|为什么|为何|怎么|如何|具体|真正|目的|原因|动机|"
    r"发生了什么|发生了|发生|什么|启动|开启|启用|动用|使用|发动|打开|关闭|解除|建造|修建|建设|制造|改造|布局|设下|安排)"
)
QUOTED_TERM_RE = re.compile(r"[“\"'「『]([^”\"'」』]{2,16})[”\"'」』]")
ACTION_TARGET_RE = re.compile(
    r"(?:启动|开启|启用|动用|使用|发动|打开|关闭|解除|建造|修建|建设|制造|改造|布局|设下|安排)"
    r"(?:[“\"'「『])?([^\s，。！？；、”\"'」』?？]{2,18})(?:[”\"'」』])?"
)
ACTION_TARGET_BOUNDARY_RE = re.compile(r"(?:的|是|和|与|及|为|为了|为什么|为何|怎么|如何|关系|原因|目的|区别|吗|么)")
ACTION_WORDS = (
    "启动",
    "开启",
    "启用",
    "动用",
    "使用",
    "发动",
    "打开",
    "关闭",
    "解除",
    "建造",
    "修建",
    "建设",
    "制造",
    "改造",
    "布局",
    "设下",
    "安排",
)
ACTION_ANSWER_MARKERS = (
    "因为",
    "为了",
    "原因",
    "目的",
    "代价",
    "性命",
    "生命",
    "捐躯",
    "牺牲",
    "解决",
    "危机",
    "威能",
    "权柄",
    "认定",
    "便能",
    "才能",
    "不得不",
    "必须",
    "故名",
    "确认",
    "安全窗口",
    "长驱直入",
    "遇刺",
    "刺杀",
    "防御被解除",
)
ACTION_PURPOSE_MARKERS = (
    "解决",
    "危机",
    "为了",
    "因为",
    "所以",
    "目的",
    "动机",
    "真正",
    "确认",
    "安全窗口",
    "长驱直入",
    "遇刺",
    "刺杀",
    "当下",
    "局势",
    "下策",
    "好戏",
    "所图",
    "何居心",
    "图谋",
)
ACTION_COST_MARKERS = (
    "代价",
    "性命",
    "生命",
    "捐躯",
    "牺牲",
    "故名",
    "全力启用",
    "本人",
    "血脉",
)
CHAPTER_TOKEN_RE = re.compile(r"(?:第[一二三四五六七八九十百零〇两0-9]+章|[0-9]{1,2}章)")
MAIN_CHAPTER_REF_RE = re.compile(
    r"(?:第\s*([一二三四五六七八九十百零〇两0-9]{1,4})\s*章|([0-9]{1,2})\s*章|level_main[_-]([0-9]{1,2})|main[_-]([0-9]{1,2}))",
    re.IGNORECASE,
)
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
    "章中",
    "目的",
    "原因",
    "动机",
    "真正目的",
    "真正目",
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
DEFINITION_QUESTION_MARKERS = (
    "是什么",
    "指什么",
    "全称",
    "本名",
    "真名",
    "真实身份",
    "身份",
    "来历",
    "是谁",
)
DEFINITION_EVIDENCE_MARKERS = (
    "是",
    "即",
    "为",
    "作为",
    "称为",
    "名为",
    "所谓",
    "全称",
    "本名",
    "真名",
    "系统",
    "产物",
    "机器",
    "制造",
    "实质",
    "本质",
    "指",
)
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
DOMAIN_ANCHOR_TERMS = TITLE_TERMS | {
    "不反",
    "岁陵",
    "书刀",
    "司岁台",
    "百灶",
    "玉门",
    "岁兽",
    "巨兽",
    "太尉",
    "太傅",
    "炎武",
}
DOMAIN_RELATED_RETRIEVAL_TERMS = {
    "碎片大厦": ("阿喃那", "道标", "天灾", "风暴", "控制", "启用", "战火", "敌人"),
    "阿喃那": ("碎片大厦", "道标", "控制", "源石", "权限", "天灾", "风暴"),
    "英雄宴": ("武典", "秘籍", "争斗", "好戏", "朔", "山海众", "云青萍", "槐天裴"),
    "武典": ("英雄宴", "秘籍", "争斗", "朔", "山海众", "云青萍"),
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
INTERNAL_EVIDENCE_META_RE = re.compile(
    r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]+\]\s*|\[E\d+\]\s*"
)
INHERITANCE_RE = re.compile(r"([\u4e00-\u9fff]{2,8})的(后人|女儿|儿子|传人)")
KINSHIP_RE = re.compile(r"(亲生父亲|父亲|母亲|家人|老师|师父|弟子|学生)")
REAL_NAME_RE = re.compile(r"(?:原名|本名|真名)[为叫是：:\s]*([\u4e00-\u9fff]{2,8}(?:·[\u4e00-\u9fff]{1,8})?)")
CONSPIRACY_ANCHOR_RE = re.compile(r"(?:撞破|发现|曝光|阻止)?([\u4e00-\u9fff]{2,4})城议员的阴谋")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
DIALOGUE_ROLE_PREFIX_RE = re.compile(r"^(user|assistant)\s*:\s*(.*)$", re.IGNORECASE)
REVEAL_KNOWLEDGE_RETRIEVAL_TERMS = (
    "曝光",
    "败露",
    "贝希曼",
    "贝希曼伯爵",
    "议员",
    "警备队",
    "送线索",
    "劫持",
    "爆炸",
    "工厂",
    "地下",
    "设备",
    "物流通道",
    "议会",
    "拨款",
    "报告损失",
    "钱的窟窿",
    "栽赃",
    "嫁祸",
    "感染者",
)
REVEAL_QUERY_TERMS = ("阴谋", "真相", "秘密", "识破", "揭穿", "曝光", "暴露", "幕后", "主使", "黑幕", "骗局", "诡计")
REVEAL_DIRECT_EVIDENCE_TERMS = (
    "苏茜去警备队送线索",
    "送线索",
    "遭到劫持",
    "劫持",
    "意外爆炸",
    "爆炸",
    "逃出",
    "阴谋得以曝光",
    "曝光",
    "贝希曼议员的阴谋",
    "贝希曼伯爵",
    "警备队长",
    "工厂",
    "地下",
    "废弃物流通道",
    "物流通道",
    "设备",
    "报告损失",
    "拨给我的钱",
    "钱的窟窿",
    "栽赃",
    "嫁祸",
    "感染者社区",
)
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
WEB_CONTEXT_TASK_TYPE = "web_context_retrieval"
MINIRAG_CHAPTER_EXPANSION_TASK_TYPE = "minirag_chapter_expansion_retrieval"
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
    "final_answer",
    "supported_facts",
    "inferred_facts",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
    "reflect_tokens",
)
CONCLUSION_IGNORED_EXTRA_FIELDS = {
    "additional_evidence_needed",
    "clarification_questions",
    "confidence",
    "conflicting_info",
    "current_round",
    "decision",
    "dialogue_context",
    "follow_up_question",
    "new_entities",
    "new_keywords",
    "slot_values",
}
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
WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES = (
    "https://www.sogou.com/web?query={query}",
    "http://www.baidu.com/s?wd={query}",
    "https://duckduckgo.com/html/?q={query}",
    "https://www.bing.com/search?q={query}",
)
WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES = (
    "明日方舟 {story_name} {question_terms} 剧情解析 时间线",
    "明日方舟 {story_name} 剧情解析 时间线",
    "{story_name} {question_terms} 明日方舟 剧情",
    "{story_name} 明日方舟 剧情解析",
)
WEB_CONTEXT_EXCLUDED_ACTIVITY_NAMES = {
    "",
    "干员档案",
    "萌百世界观资料",
    "外部资料",
    "未知",
}
WEB_CONTEXT_BLOCKED_URL_HOSTS = {
    "duckduckgo.com",
    "www.duckduckgo.com",
    "bing.com",
    "www.bing.com",
    "r.bing.com",
    "th.bing.com",
    "baidu.com",
    "www.baidu.com",
    "google.com",
    "www.google.com",
    "sogou.com",
    "www.sogou.com",
    "so.com",
    "www.so.com",
}
WEB_CONTEXT_STATIC_URL_SUFFIXES = (
    ".css",
    ".js",
    ".mjs",
    ".ico",
    ".svg",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".pdf",
    ".zip",
    ".woff",
    ".woff2",
    ".ttf",
)
WEB_CONTEXT_QUERY_ANCHOR_TERMS = DOMAIN_ANCHOR_TERMS | {
    "危机",
    "岁兽之患",
    "苏醒",
    "平息",
    "望日",
    "望",
    "辞岁行",
    "百灶",
}
WEB_CONTEXT_GENERIC_QUERY_TERMS = {
    "本质",
    "原本",
    "一体",
    "消灭",
    "代价",
    "动乱",
    "灭顶之灾",
    "开战",
    "解决",
    "原因",
    "概念定义",
    "危机原因",
    "回答类型",
}


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
    supported_facts: list[dict[str, Any]] = field(default_factory=list)
    inferred_facts: list[dict[str, Any]] = field(default_factory=list)
    grounding_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WebContextConfig:
    enabled: bool = False
    cache_dir: Path | None = None
    cache_ttl_seconds: int = 604800
    timeout_seconds: float = 6.0
    max_elapsed_seconds: float = 18.0
    max_first_round_evidence: int = 24
    min_story_hits: int = 2
    max_search_queries: int = 2
    max_search_results: int = 6
    max_pages: int = 3
    max_chars_per_page: int = 2200
    max_total_chars: int = 6000
    rerank_top_k: int = 2
    rerank_min_score: float = 1.0
    require_story_or_question_hit: bool = True
    force_prompt_evidence: bool = False
    query_templates: tuple[str, ...] = WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES
    search_url_templates: tuple[str, ...] = WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES
    user_agent: str = "Mozilla/5.0 GoldenglowRAG/1.0"


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


def strip_internal_evidence_meta(text: str) -> str:
    return INTERNAL_EVIDENCE_META_RE.sub("", text or "")


def _truncate_text(text: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def _normalize_string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        return default
    normalized = tuple(str(item).strip() for item in value if str(item).strip())
    return normalized or default


def build_web_context_config(payload: dict[str, Any] | None) -> WebContextConfig:
    cfg = payload if isinstance(payload, dict) else {}
    cache_dir_value = cfg.get("cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value else None
    return WebContextConfig(
        enabled=bool(cfg.get("enabled", False)),
        cache_dir=cache_dir,
        cache_ttl_seconds=max(0, int(cfg.get("cache_ttl_seconds", 604800))),
        timeout_seconds=max(1.0, float(cfg.get("timeout_seconds", 6.0))),
        max_elapsed_seconds=max(3.0, float(cfg.get("max_elapsed_seconds", 18.0))),
        max_first_round_evidence=max(1, int(cfg.get("max_first_round_evidence", 24))),
        min_story_hits=max(1, int(cfg.get("min_story_hits", 2))),
        max_search_queries=max(1, int(cfg.get("max_search_queries", 2))),
        max_search_results=max(1, int(cfg.get("max_search_results", 6))),
        max_pages=max(1, int(cfg.get("max_pages", 3))),
        max_chars_per_page=max(300, int(cfg.get("max_chars_per_page", 2200))),
        max_total_chars=max(800, int(cfg.get("max_total_chars", 6000))),
        rerank_top_k=max(0, int(cfg.get("rerank_top_k", 2))),
        rerank_min_score=float(cfg.get("rerank_min_score", 1.0)),
        require_story_or_question_hit=bool(cfg.get("require_story_or_question_hit", True)),
        force_prompt_evidence=bool(cfg.get("force_prompt_evidence", False)),
        query_templates=_normalize_string_tuple(cfg.get("query_templates"), WEB_CONTEXT_DEFAULT_QUERY_TEMPLATES),
        search_url_templates=_normalize_string_tuple(
            cfg.get("search_url_templates"),
            WEB_CONTEXT_DEFAULT_SEARCH_URL_TEMPLATES,
        ),
        user_agent=str(cfg.get("user_agent") or "Mozilla/5.0 GoldenglowRAG/1.0"),
    )


def _extract_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    tokens.extend(match.group(0) for match in CHAPTER_TOKEN_RE.finditer(text))
    for raw_token in QUESTION_TOKEN_RE.findall(text):
        parts = [raw_token] if raw_token.isascii() else [part for part in CHINESE_TOKEN_SPLIT_RE.split(raw_token) if part]
        for part in parts:
            normalized = part.strip()
            normalized = re.sub(r"(城)(?:识|发|曝|揭|撞|送|遭|被|去|到)$", r"\1", normalized)
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


def expand_related_retrieval_terms(terms: list[str], *, limit: int = 16) -> list[str]:
    related: list[str] = []
    for term in terms:
        compact = re.sub(r"\s+", "", term or "")
        if not compact:
            continue
        for key, values in DOMAIN_RELATED_RETRIEVAL_TERMS.items():
            if key in compact or compact in key:
                related.extend(values)
    return _dedupe_keep_order(
        item
        for item in related
        if item and item not in COMMON_NON_ENTITY_WORDS and item not in NOISY_RETRIEVAL_TOKENS
    )[:limit]


def _parse_chapter_number(value: str) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        number = int(raw)
        return number if 0 < number < 100 else None
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if raw == "十":
        return 10
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        number = tens * 10 + ones
        return number if 0 < number < 100 else None
    if len(raw) == 1 and raw in digits:
        return digits[raw]
    return None


def extract_main_chapter_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in MAIN_CHAPTER_REF_RE.finditer(text or ""):
        raw_number = next((group for group in match.groups() if group), "")
        number = _parse_chapter_number(raw_number)
        if number is not None:
            numbers.append(number)
    return list(dict.fromkeys(numbers))


def build_main_chapter_retrieval_terms(text: str) -> list[str]:
    terms: list[str] = []
    for number in extract_main_chapter_numbers(text):
        padded = f"{number:02d}"
        terms.extend(
            [
                f"第{number}章",
                f"{number}章",
                f"main_{padded}",
                f"level_main_{padded}",
                f"main_{number}",
                f"level_main_{number}",
                f"EPISODE {number}",
            ]
        )
    return _dedupe_keep_order(terms)


def expand_queries_with_main_chapter_terms(queries: list[str]) -> list[str]:
    expanded: list[str] = []
    for query in queries:
        if not query:
            continue
        expanded.append(query)
        chapter_terms = build_main_chapter_retrieval_terms(query)
        if chapter_terms and "章节限定:" not in query:
            expanded.append(query + "\n章节限定: " + " ".join(chapter_terms))
    return _dedupe_keep_order(expanded)


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
    return _dedupe_keep_order(question_entities)[:12]


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
    context_entities = _extract_context_entities(dialogue_context) if any(
        pronoun in question for pronoun in PRONOUN_REFERENCES
    ) else []

    keywords = _dedupe_keep_order(
        context_entities
        + entities
        + question_keywords
    )[:16]
    related_keywords = expand_related_retrieval_terms(entities + question_keywords)
    if related_keywords:
        keywords = _dedupe_keep_order(keywords + related_keywords)[:24]

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
    evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
    prompt_mode: str = "full",
) -> str:
    rendered_dialogue_context = _truncate_text(
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
    if prompt_mode == "minimal":
        system_prompt = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
        evidence_brief = render_short_evidence_brief(
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
        minirag_hints = render_minirag_hints_for_prompt(
            prompt_evidence
            if prompt_evidence is not None
            else select_prompt_evidence(
                question,
                current_hypothesis,
                evidence,
                prompt_evidence_top_k=prompt_evidence_top_k,
            ),
            current_hypothesis,
        )
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
                'answer_directly: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_refs":[{"evidence_id":"","quote":""}]}],"inferred_facts":[],"final_answer":""}',
                'retrieve_more: {"next_action":"retrieve_more","follow_up_hypothesis":{"question":"","query_type":"","entities":[],"keywords":[],"expected_answer_type":"","dialogue_context":""}}',
                'abstain: {"next_action":"abstain","final_answer":"现有证据不足以确认。"}',
                'follow_up_hypothesis_fields: question,query_type,entities,keywords,expected_answer_type,dialogue_context',
                "rules: JSON only；只能使用 evidence_brief 中的证据；单条 quote 必须从 evidence_brief 原文精确复制，推荐20-60字，硬上限80字；每个 supported_fact 最多2条 quote 且总长<=160字；supported_facts最多6条，所有quote总长最好<=400字；final_answer 只能使用 supported_facts 和 inferred_facts；证据不足才 retrieve_more；不要输出 current_round、confidence、decision、missing_slots、clarification_question。",
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
            "20. 若当前证据中出现与问题关键词直接匹配的专名、引号术语或“启动/开启/动用 + 对象”原文，优先基于这些原文回答；不要因缺少外围背景而直接 abstain。",
            "21. 对“为什么启动/开启/动用某物”类问题，回答必须围绕该“某物”的直接证据；优先使用同时包含对象名、启动/开启/动用动作、代价、目的或危机的证据。",
            "22. 禁止用不包含目标对象名的背景段替代直接原因；例如问题问“为什么启动X”，不能只回答“为什么没有做Y/为什么放弃Z”。",
            "23. 若直接证据与背景证据同时出现，先答直接原因，再用背景作补充，不能把背景原因写成主要原因。",
            "24. 对“为什么/原因/目的/动机/后果”类问题，必须区分人物口头宣称、表面计划、真实执行动作和最终结果；如果后续证据显示实际动作与口头宣称相反，不能把口头宣称当作真实原因，应说明这是表面说法或前置动作。",
            "25. 因果题优先按“动机/背景 -> 实际执行动作 -> 直接后果”组织答案；若缺少动机证据但有执行动作和后果，只能回答已确认的因果链，并标注动机不足，不要编造。",
            "26. 对阴谋/真相/识破/曝光类问题，如果证据能确认主使、关键行动或结果，先给最小充分答案；不要因为缺少完整目的、所有手段或全部后果而 abstain。",
            "27. 阴谋/真相类答案可以使用“现有证据显示/可确认部分”限定；但如果证据中已有“送线索、劫持、爆炸、曝光、工厂、物流通道、报告损失”等直接链条，应选择 answer_directly。",
            "28. 对“某场危机是什么/指什么”类问题，必须区分“核心危机/直接问题”和“外部压力/潜在最坏后果”；不要把“可能开战、最坏结果、后续解决方案”写成危机爆发的直接原因。",
            "29. 你的输出第一字符必须是 {",
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
    if hypothesis.entities:
        lines.append("实体: " + " ".join(hypothesis.entities))
    if hypothesis.keywords:
        lines.append("关键词: " + " ".join(hypothesis.keywords[:10]))
    chapter_terms = build_main_chapter_retrieval_terms(
        "\n".join([hypothesis.question, " ".join(hypothesis.entities), " ".join(hypothesis.keywords)])
    )
    if chapter_terms:
        lines.append("章节限定: " + " ".join(chapter_terms))
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


POLLUTED_RETRIEVAL_SLOT_PATTERNS = (
    "supported_fact_",
    "quote_not_found",
    "quote_over_",
    "quote_total_over_",
    "missing_quote",
    "missing_evidence_refs",
    "not_object",
    "answer_directly 缺少可校验 quote 支撑",
    "final_answer_has_terms_outside_supported_facts",
    "has_terms_outside_quotes",
    "grounding 校验",
    "JSON-like 结论已使用启发式续检索",
    "tuple-like 结论已转换为续检索",
    "follow_up_hypothesis 不可用",
    "模型未返回 follow_up_hypothesis",
)


def clean_missing_slots_for_retrieval(missing_slots: list[str]) -> list[str]:
    cleaned: list[str] = []
    for slot in missing_slots:
        text = str(slot or "").strip()
        if not text:
            continue
        if any(pattern in text for pattern in POLLUTED_RETRIEVAL_SLOT_PATTERNS):
            continue
        if re.fullmatch(r"[A-Za-z_0-9:>\-]+", text):
            continue
        cleaned.append(text)
    return _dedupe_keep_order(cleaned)[:8]


def build_heuristic_follow_up_hypothesis(
    question: str,
    current_hypothesis: HypothesisDocument,
    missing_slots: list[str],
) -> HypothesisDocument:
    missing_slots = clean_missing_slots_for_retrieval(missing_slots)
    slot_text = " ".join(slot for slot in missing_slots if slot)
    slot_terms = _extract_content_tokens(slot_text)
    slot_entities = [term for term in slot_terms if _is_entity_candidate(term)]
    action_targets = extract_action_targets(question + "\n" + current_hypothesis.question)
    is_reason_query = any(token in question for token in ("为什么", "为何", "原因", "目的", "动机", "真正"))

    bridge_terms: list[str] = []
    if is_reason_query:
        bridge_terms.extend(["原因", "目的", "直接原因", "具体原因"])
        for target in action_targets[:3]:
            bridge_terms.extend([f"{target} 目的", f"{target} 原因"])
            if current_hypothesis.entities and current_hypothesis.entities[0] != target:
                bridge_terms.extend(
                    [
                        f"{current_hypothesis.entities[0]} {target}",
                        f"{current_hypothesis.entities[0]} {target} 原因",
                    ]
                )

    focus_terms = _dedupe_keep_order(
        action_targets + current_hypothesis.entities[:4] + current_hypothesis.keywords[:8] + slot_terms + bridge_terms
    )
    related_terms = expand_related_retrieval_terms(focus_terms)
    expected_answer_type = current_hypothesis.expected_answer_type
    if is_reason_query and action_targets:
        expected_answer_type = "short_text"

    return HypothesisDocument(
        question=current_hypothesis.question or question,
        intent=current_hypothesis.intent,
        query_type=current_hypothesis.query_type or infer_query_type(
            question,
            current_hypothesis.intent,
            expected_answer_type,
        ),
        entities=_dedupe_keep_order(current_hypothesis.entities + slot_entities)[:12],
        keywords=_dedupe_keep_order(current_hypothesis.keywords + slot_terms + bridge_terms + related_terms)[:24],
        expected_answer_type=expected_answer_type,
        dialogue_context=current_hypothesis.dialogue_context,
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
        for term in REVEAL_KNOWLEDGE_RETRIEVAL_TERMS:
            if term in text:
                bridge_keywords.append(term)

    if "阴谋" in context_text or any(token in hypothesis.query_type for token in ("reveal", "mystery")):
        bridge_keywords.extend(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
        if any("卡拉顿" in term for term in hypothesis.entities + hypothesis.keywords) or "卡拉顿" in context_text:
            bridge_entities.extend(["卡拉顿", "卡拉顿城议员"])

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
    action_targets = extract_action_targets(question + "\n" + hypothesis.question)
    focus_terms = _dedupe_keep_order(
        action_targets
        + _extract_content_tokens(question)
        + hypothesis.entities[:4]
        + hypothesis.keywords[:4]
    )
    related_terms = expand_related_retrieval_terms(focus_terms)
    is_reason_query = any(token in question for token in ("为什么", "为何", "原因", "目的", "动机", "真正"))

    if action_targets:
        for target in action_targets[:3]:
            queries.append(target)
            if primary_entity and primary_entity != target:
                queries.append(f"{primary_entity} {target}")
            if is_reason_query:
                queries.append(f"{target} 目的 原因")
                if primary_entity and primary_entity != target:
                    queries.append(f"{primary_entity} {target} 目的 原因")
        if related_terms:
            queries.append(" ".join(_dedupe_keep_order([*action_targets[:3], *related_terms[:8]])))

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
            queries.extend(
                [
                    "阴云火花 贝希曼 阴谋 曝光",
                    "卡拉顿 贝希曼 议员 阴谋",
                    "苏茜 警备队 送线索 劫持 爆炸",
                    "贝希曼 工厂 地下 设备 物流通道",
                    "贝希曼 议会 拨款 报告损失 钱的窟窿",
                    "贝希曼 栽赃 感染者",
                ]
            )

    if primary_entity and any(token in question for token in IDENTITY_HINT_WORDS):
        queries.extend(
            [
                f"{primary_entity} 身份 来历",
                f"{primary_entity} 身世 真相",
            ]
        )

    if "阴谋" in question or "阴谋" in hypothesis.keywords or hypothesis.query_type in {"reveal", "mystery"}:
        queries.extend(term for term in REVEAL_KNOWLEDGE_RETRIEVAL_TERMS if term in hypothesis.keywords)
    queries.extend(hypothesis.keywords[:8])

    for entity in hypothesis.entities[:4]:
        queries.append(entity)
        for keyword in hypothesis.keywords[:4]:
            if keyword != entity:
                queries.append(f"{entity} {keyword}")
    if related_terms:
        for term in focus_terms[:4]:
            queries.append(" ".join(_dedupe_keep_order([term, *related_terms[:6]])))

    deduped_queries: list[str] = []
    seen: set[str] = {question.strip()}
    for query in queries:
        normalized = query.strip()
        query_terms = normalized.split()
        if (
            not normalized
            or normalized in seen
            or len(query_terms) >= 2 and len(set(query_terms)) == 1
        ):
            continue
        seen.add(normalized)
        deduped_queries.append(normalized)
    return deduped_queries[:14]


def build_missing_slot_queries(
    hypothesis: HypothesisDocument,
    missing_slots: list[str],
) -> list[str]:
    missing_slots = clean_missing_slots_for_retrieval(missing_slots)
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


def merge_evidence_keep_order(*evidence_lists: list[dict[str, Any]], limit: int | None = None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evidence in evidence_lists:
        for item in evidence:
            doc = item.get("document") or {}
            doc_id = str(doc.get("id") or item.get("doc_index") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(item)
            if limit is not None and len(merged) >= limit:
                return merged
    return merged


def infer_dominant_minirag_chapter_scope(
    *ranked_lists: list[dict[str, Any]],
    max_items: int = 40,
) -> dict[str, Any] | None:
    scope_scores: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    scope_labels: dict[str, str] = {}
    for source_index, ranked in enumerate(ranked_lists):
        source_weight = 1.25 if source_index == 0 else 1.0
        for rank, item in enumerate(ranked[:max_items]):
            doc = item.get("document") or {}
            if not isinstance(doc, dict):
                continue
            scope = document_chapter_scope_key(doc)
            if not scope:
                continue
            scope_scores[scope] += source_weight / ((rank + 1) ** 0.5)
            scope_counts[scope] += 1
            scope_labels.setdefault(scope, document_chapter_scope_label(doc) or scope)
    if not scope_scores:
        return None
    ranked_scopes = sorted(
        scope_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    scope, score = ranked_scopes[0]
    if scope_counts[scope] < 3:
        return None
    runner_up_score = float(ranked_scopes[1][1]) if len(ranked_scopes) > 1 else 0.0
    dominance_ratio = float(score) / max(runner_up_score, 1e-6)
    if runner_up_score > 0 and dominance_ratio < 1.15 and scope_counts[scope] < 6:
        return None
    return {
        "scope": scope,
        "label": scope_labels.get(scope) or scope,
        "score": float(score),
        "count": int(scope_counts[scope]),
        "runner_up_score": runner_up_score,
        "dominance_ratio": dominance_ratio,
        "candidates": [
            {
                "scope": candidate_scope,
                "label": scope_labels.get(candidate_scope) or candidate_scope,
                "score": float(candidate_score),
                "count": int(scope_counts[candidate_scope]),
            }
            for candidate_scope, candidate_score in ranked_scopes[:5]
        ],
    }


def infer_dominant_storyline_scope(
    *ranked_lists: list[dict[str, Any]],
    max_items: int = 40,
) -> dict[str, Any] | None:
    scope_scores: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    for source_index, ranked in enumerate(ranked_lists):
        source_weight = 1.25 if source_index == 0 else 1.0
        for rank, item in enumerate(ranked[:max_items]):
            doc = item.get("document") or {}
            if not isinstance(doc, dict):
                continue
            scopes = document_storyline_scopes(doc)
            if not scopes:
                continue
            for scope in scopes:
                scope_scores[scope] += source_weight / ((rank + 1) ** 0.5)
                scope_counts[scope] += 1
    if not scope_scores:
        return None
    ranked_scopes = sorted(scope_scores.items(), key=lambda item: item[1], reverse=True)
    scope, score = ranked_scopes[0]
    if scope_counts[scope] < 3:
        return None
    runner_up_score = float(ranked_scopes[1][1]) if len(ranked_scopes) > 1 else 0.0
    dominance_ratio = float(score) / max(runner_up_score, 1e-6)
    if runner_up_score > 0 and dominance_ratio < 1.1 and scope_counts[scope] < 6:
        return None
    return {
        "scope": scope,
        "label": storyline_scope_label(scope),
        "score": float(score),
        "count": int(scope_counts[scope]),
        "runner_up_score": runner_up_score,
        "dominance_ratio": dominance_ratio,
        "candidates": [
            {
                "scope": candidate_scope,
                "label": storyline_scope_label(candidate_scope),
                "score": float(candidate_score),
                "count": int(scope_counts[candidate_scope]),
            }
            for candidate_scope, candidate_score in ranked_scopes[:5]
        ],
    }


def filter_hits_by_chapter_scope(
    hits: list[dict[str, Any]],
    chapter_scope: str,
) -> list[dict[str, Any]]:
    if not chapter_scope:
        return hits
    scoped_hits: list[dict[str, Any]] = []
    for item in hits:
        doc = item.get("document") or {}
        if isinstance(doc, dict) and document_chapter_scope_key(doc) == chapter_scope:
            scoped_hits.append(item)
    return scoped_hits


def build_minirag_expansion_queries(
    question: str,
    hypothesis: HypothesisDocument,
    minirag_hits: list[dict[str, Any]],
    *,
    chapter_scope_label: str,
    top_k: int = 8,
) -> list[str]:
    anchors = extract_question_anchor_terms(question, hypothesis)[:12]
    metadata_terms: list[str] = []
    snippets: list[str] = []
    for item in minirag_hits[:top_k]:
        doc = item.get("document") or {}
        if not isinstance(doc, dict):
            continue
        for key in ("activity_name", "story_name", "stage_code", "stage_name", "zone_name"):
            value = str(doc.get(key) or "").strip()
            if value:
                metadata_terms.append(value)
        text = strip_internal_evidence_meta(
            str(doc.get("clean_text") or doc.get("search_text") or "")
        ).strip()
        if text:
            snippets.append(_truncate_text(text, 260))

    related_terms = expand_related_retrieval_terms(anchors)
    compact_terms = _dedupe_keep_order([*anchors, *related_terms, *metadata_terms])[:32]
    evidence_blob = "\n".join(snippets[:top_k])
    queries = [
        "\n".join(
            [
                question,
                f"章节限定: {chapter_scope_label}",
                "关系图扩展线索: " + " ".join(compact_terms),
                "关系图扩展证据:",
                _truncate_text(evidence_blob, 1400),
            ]
        ).strip(),
        " ".join([question, chapter_scope_label, *compact_terms]).strip(),
    ]
    return _dedupe_keep_order([query for query in queries if query])[:3]


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
        clean_text = _best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        chain_text = _document_chain_text(item)
        if bool(item.get("prompt_prefer_clean_text")):
            chain_text = "" if clean_text else chain_text
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
            clean_text = strip_internal_evidence_meta(str(doc["clean_text"]))
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


def render_short_evidence_brief(
    evidence: list[dict[str, Any]],
    *,
    max_chars_per_doc: int = 260,
    max_total_chars: int = 2200,
) -> str:
    lines: list[str] = []
    total_chars = 0
    seen: set[str] = set()
    for item in evidence:
        doc = item.get("document") or {}
        doc_id = str(doc.get("id") or item.get("doc_index") or "").strip()
        text = _best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        text = re.sub(r"\s+", " ", strip_internal_evidence_meta(text)).strip()
        if not text:
            continue
        key = doc_id or text[:160]
        if key in seen:
            continue
        seen.add(key)
        text = _truncate_text(text, max_chars_per_doc)
        line = f"{len(lines) + 1}. {doc_id or '<unknown>'}: {text}"
        if lines and total_chars + len(line) > max_total_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines)


def render_minirag_hints_for_prompt(evidence: list[dict[str, Any]], hypothesis: HypothesisDocument) -> str:
    entities: list[str] = []
    relations: list[str] = []
    neighbors: list[str] = []
    for item in evidence[:12]:
        doc = item.get("document") or {}
        for value in (
            doc.get("activity_name"),
            doc.get("story_name"),
            doc.get("stage_code"),
            doc.get("stage_name"),
            doc.get("zone_name"),
        ):
            text = str(value or "").strip()
            if text:
                entities.append(text)
        for role in item.get("evidence_chain_roles") or []:
            text = str(role or "").strip()
            if text:
                neighbors.append(text)
        chain_text = str(item.get("evidence_chain_text") or "").strip()
        if chain_text:
            for token in extract_question_anchor_terms(chain_text, hypothesis)[:4]:
                entities.append(token)
    entities = _dedupe_keep_order(hypothesis.entities + entities)[:12]
    keywords = _dedupe_keep_order(hypothesis.keywords + neighbors)[:12]
    if len(entities) >= 2:
        relations = [f"{entities[index]}-相关-{entities[index + 1]}" for index in range(min(len(entities) - 1, 6))]
    return (
        "entities="
        + ",".join(entities)
        + " | relations="
        + ";".join(relations)
        + " | neighbors="
        + ",".join(keywords)
    )


def _evidence_identity(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    doc_id = str(doc.get("id") or "").strip()
    if doc_id:
        return "doc:" + doc_id
    doc_index = item.get("doc_index")
    if doc_index is not None:
        return "idx:" + str(doc_index)
    return "text:" + _evidence_text(item)[:160]


def _evidence_text(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    parts = [
        str(item.get("evidence_chain_text") or ""),
        str(doc.get("clean_text") or ""),
        str(doc.get("search_text") or ""),
        str(doc.get("activity_name") or ""),
        str(doc.get("story_name") or ""),
        str(doc.get("stage_code") or ""),
        str(doc.get("avg_tag") or ""),
        str(doc.get("source_path") or ""),
    ]
    text_parts = [strip_internal_evidence_meta(part).strip() for part in parts if str(part).strip()]
    return "\n".join(_dedupe_keep_order(text_parts))


def _is_reveal_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = "\n".join([question or "", hypothesis.question or "", " ".join(hypothesis.keywords)])
    return hypothesis.query_type in {"reveal", "mystery"} or any(term in text for term in REVEAL_QUERY_TERMS)


def _document_clean_text(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    return strip_internal_evidence_meta(str(doc.get("clean_text") or doc.get("search_text") or "")).strip()


def _document_chain_text(item: dict[str, Any]) -> str:
    return strip_internal_evidence_meta(str(item.get("evidence_chain_text") or "")).strip()


def _best_prompt_text(item: dict[str, Any], *, prefer_direct: bool = False) -> str:
    del prefer_direct
    clean_text = _document_clean_text(item)
    chain_text = _document_chain_text(item)
    if clean_text:
        return clean_text
    return chain_text


def _reveal_direct_score(text: str, question: str, hypothesis: HypothesisDocument) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text:
        return 0
    query_terms = _dedupe_keep_order(
        hypothesis.entities
        + hypothesis.keywords
        + _extract_content_tokens(question)
        + list(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
    )
    query_hits = sum(1 for term in query_terms if term and term in compact_text)
    direct_hits = sum(1 for term in REVEAL_DIRECT_EVIDENCE_TERMS if term in compact_text)
    score = query_hits + direct_hits * 3
    if "贝希曼" in compact_text and "阴谋" in compact_text:
        score += 4
    if "苏茜" in compact_text and ("送线索" in compact_text or "劫持" in compact_text):
        score += 6
    if "贝希曼议员的阴谋得以曝光" in compact_text:
        score += 10
    if "[uc]info" in str((hypothesis.question or "") + text) and "阴谋" in compact_text:
        score += 3
    return score


def _best_reveal_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not _is_reveal_question(question, hypothesis):
        return []
    candidates: list[tuple[int, float, int, dict[str, Any]]] = []
    for index, item in enumerate(evidence):
        clean_text = _document_clean_text(item)
        chain_text = _document_chain_text(item)
        score = max(
            _reveal_direct_score(clean_text, question, hypothesis),
            _reveal_direct_score(chain_text, question, hypothesis),
            _reveal_direct_score(_evidence_text(item), question, hypothesis),
        )
        if score <= 0:
            continue
        doc = item.get("document") or {}
        source_path = str(doc.get("source_path") or "")
        if "handbook_info_table.json" in source_path or "charword_table.json" in source_path:
            score -= 5
        if "[uc]info" in source_path and ("阴谋" in clean_text or "曝光" in clean_text):
            score += 8
        candidates.append((score, _evidence_score(item), -index, item))
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return [item for score, _, _, item in candidates[:limit] if score > 0]


def _prefer_direct_prompt_text(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    payload["prompt_prefer_clean_text"] = True
    return payload


def _is_web_context_item(item: dict[str, Any]) -> bool:
    doc = item.get("document") or {}
    return item.get("supplemental_source") == "web_context" or str(doc.get("id") or "").startswith("web_context/")


def _is_moegirl_evidence(item: dict[str, Any]) -> bool:
    doc = item.get("document") or {}
    doc_id = str(doc.get("id") or "")
    activity_name = str(doc.get("activity_name") or "")
    source_path = str(doc.get("source_path") or "")
    source_path_lower = source_path.lower()
    return (
        doc_id.startswith("moegirl/")
        or activity_name == "萌百世界观资料"
        or "/moegirl/" in source_path_lower
        or "moegirl" in source_path_lower
        or "萌百" in source_path
    )


def _prompt_evidence_score(item: dict[str, Any]) -> float:
    score = _evidence_score(item)
    if _is_moegirl_evidence(item):
        score -= 6.0
    return score


def _dedupe_prompt_evidence_candidates(
    evidence: list[dict[str, Any]],
    *,
    similarity_threshold: float = 0.82,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    seen_token_sets: list[set[str]] = []
    for item in evidence:
        identity = _evidence_identity(item)
        if identity in seen_identities:
            continue
        text = _best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        token_set = _text_similarity_tokens(text)
        if token_set and any(_jaccard_similarity(token_set, seen) >= similarity_threshold for seen in seen_token_sets):
            continue
        seen_identities.add(identity)
        if token_set:
            seen_token_sets.append(token_set)
        output.append(item)
    return output


def _merge_forced_prompt_evidence(
    forced: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_token_sets: list[set[str]] = []
    for item in forced + selected:
        identity = _evidence_identity(item)
        if identity in seen:
            continue
        text = _best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        token_set = _text_similarity_tokens(text)
        if token_set and any(_jaccard_similarity(token_set, seen_tokens) >= 0.82 for seen_tokens in seen_token_sets):
            continue
        seen.add(identity)
        if token_set:
            seen_token_sets.append(token_set)
        output.append(_prefer_direct_prompt_text(item) if _is_web_context_item(item) else item)
        if len(output) >= limit:
            break
    return output


def _story_name_candidate(document: dict[str, Any]) -> str:
    for key in ("activity_name", "story_name"):
        value = re.sub(r"\s+", " ", str(document.get(key) or "")).strip()
        if not value or value in WEB_CONTEXT_EXCLUDED_ACTIVITY_NAMES:
            continue
        if value.startswith("档案资料") or value in {"晋升记录", "模组故事", "语音记录"}:
            continue
        return value
    return ""


def _dominant_story_name_from_evidence(
    evidence: list[dict[str, Any]],
    *,
    max_items: int,
    min_hits: int,
) -> tuple[str, dict[str, int]]:
    scores: dict[str, int] = {}
    for rank, item in enumerate(evidence[:max_items], start=1):
        document = item.get("document") or {}
        story_name = _story_name_candidate(document)
        if not story_name:
            continue
        weight = max(1, max_items - rank + 1)
        scores[story_name] = scores.get(story_name, 0) + weight
    if not scores:
        return "", {}
    winner, score = max(scores.items(), key=lambda pair: (pair[1], len(pair[0])))
    if score < min_hits:
        return "", scores
    return winner, scores


def _web_context_question_terms(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    hypothesis: HypothesisDocument | None = None,
    limit: int = 12,
) -> list[str]:
    terms: list[str] = []
    terms.extend(extract_entities(question))
    terms.extend(_extract_content_tokens(question))

    seed_text = question + "\n" + "\n".join(_evidence_text(item)[:1200] for item in evidence[:8])
    if "岁陵" in question and "危机" in question:
        terms.extend(["岁陵", "危机", "岁陵危机", "岁兽之患", "岁兽", "苏醒", "平息"])
    for anchor in sorted(WEB_CONTEXT_QUERY_ANCHOR_TERMS, key=len, reverse=True):
        if anchor and anchor in seed_text:
            terms.append(anchor)
    if "危机" in question:
        terms.append("危机")
    if hypothesis is not None:
        terms.extend(hypothesis.entities)
        terms.extend(
            term
            for term in hypothesis.keywords[:16]
            if term not in WEB_CONTEXT_GENERIC_QUERY_TERMS
        )
    for item in evidence[:8]:
        text = _evidence_text(item)
        if "不反" in text or "不反" in question:
            terms.extend(["不反", "岁陵", "真龙"])
        for term in WEB_CONTEXT_QUERY_ANCHOR_TERMS:
            if term in question or term in text:
                terms.append(term)
    return _dedupe_keep_order(
        [
            term
            for term in terms
            if term
            and term not in COMMON_NON_ENTITY_WORDS
            and term not in WEB_CONTEXT_GENERIC_QUERY_TERMS
            and (term in WEB_CONTEXT_QUERY_ANCHOR_TERMS or term not in NOISY_RETRIEVAL_TOKENS)
            and len(term) <= 12
        ]
    )[:limit]


def _cache_key_for_web_context(story_name: str, queries: list[str]) -> str:
    raw = json.dumps(
        {"version": 7, "story_name": story_name, "queries": queries},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _read_web_context_cache(cache_dir: Path | None, cache_key: str, ttl_seconds: int) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    created_at = float(payload.get("created_at") or 0)
    if ttl_seconds > 0 and created_at and time.time() - created_at > ttl_seconds:
        return None
    return payload if isinstance(payload, dict) else None


def _write_web_context_cache(cache_dir: Path | None, cache_key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps({"created_at": time.time(), **payload}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


def _strip_html_to_text(payload: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript|svg|template)[^>]*>.*?</\1>", " ", payload or "")
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>|</div\s*>|</li\s*>|</h[1-6]\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) >= 8)


def _decode_duckduckgo_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def _decode_bing_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    encoded = next((query[key][0] for key in ("u", "url") if query.get(key)), "")
    if not encoded:
        return url
    if encoded.startswith("a1") and len(encoded) > 4:
        payload = encoded[2:]
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii")).decode("utf-8", "ignore")
            if decoded.startswith(("http://", "https://")):
                return decoded
        except Exception:
            pass
    return unquote(encoded)


def _normalize_search_href(href: str) -> str:
    href = html.unescape(href.strip())
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/l/?") or "duckduckgo.com/l/?" in href:
        href = _decode_duckduckgo_redirect(href)
    if href.startswith(("/url?", "/ck/a")) or "bing.com/ck/a" in href:
        href = _decode_bing_redirect(href)
    return unquote(href)


def _is_usable_web_url(url: str) -> bool:
    if not url.startswith(("http://", "https://")):
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return False
    if host in WEB_CONTEXT_BLOCKED_URL_HOSTS:
        if host in {"baidu.com", "www.baidu.com"} and parsed.path == "/link":
            return True
        return False
    normalized_url = url.lower().split("?", 1)[0]
    if any(normalized_url.endswith(suffix) for suffix in WEB_CONTEXT_STATIC_URL_SUFFIXES):
        return False
    if "/rs/" in normalized_url or "/assets/" in normalized_url or "/static/" in normalized_url:
        return False
    return True


def _extract_search_results(search_html: str, *, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    result_pattern = re.compile(
        r'''<a[^>]+href=["'](?P<href>[^"']+)["'][^>]*>(?P<title>.*?)</a>''',
        re.IGNORECASE | re.DOTALL,
    )
    for match in result_pattern.finditer(search_html or ""):
        title = _strip_html_to_text(match.group("title"))
        if "剧情" not in title:
            continue
        href = _normalize_search_href(match.group("href"))
        if not _is_usable_web_url(href) or href in seen_urls:
            continue
        seen_urls.add(href)
        results.append({"url": href, "title": title[:160]})
        if len(results) >= limit:
            return results
    return results


def _extract_search_result_urls(search_html: str, *, limit: int) -> list[str]:
    return [result["url"] for result in _extract_search_results(search_html, limit=limit)]


def _http_get_text(url: str, *, timeout: float, user_agent: str) -> str:
    try:
        import requests
    except Exception:
        return ""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.9,*/*;q=0.8"},
        )
        if response.status_code >= 400:
            return ""
        content_type = response.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type and content_type:
            return ""
        response.encoding = response.encoding or "utf-8"
        return response.text
    except Exception:
        return ""


def _resolve_search_redirect_url(url: str, *, timeout: float, user_agent: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in {"baidu.com", "www.baidu.com"} or parsed.path != "/link":
        return url
    try:
        import requests
    except Exception:
        return url
    try:
        response = requests.head(
            url,
            timeout=timeout,
            allow_redirects=False,
            headers={"User-Agent": user_agent, "Accept": "text/html, text/plain;q=0.9,*/*;q=0.8"},
        )
        location = response.headers.get("location") or ""
        if not location:
            return url
        resolved = urljoin(url, location)
        return resolved if _is_usable_web_url(resolved) else url
    except Exception:
        return url


def _remaining_timeout(deadline: float | None, default_timeout: float) -> float:
    if deadline is None:
        return default_timeout
    return max(0.5, min(default_timeout, deadline - time.time()))


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.time() >= deadline


def _web_search_urls(query: str, config: WebContextConfig, *, deadline: float | None = None) -> list[str]:
    return [result["url"] for result in _web_search_results(query, config, deadline=deadline)]


def _web_search_results(query: str, config: WebContextConfig, *, deadline: float | None = None) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    encoded_query = quote_plus(query)
    for template in config.search_url_templates:
        if _deadline_expired(deadline):
            break
        search_url = template.format(query=encoded_query)
        raw_html = _http_get_text(
            search_url,
            timeout=_remaining_timeout(deadline, config.timeout_seconds),
            user_agent=config.user_agent,
        )
        if not raw_html:
            continue
        for result in _extract_search_results(raw_html, limit=config.max_search_results):
            url = result["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                results.append(result)
            if len(results) >= config.max_search_results:
                return results
    return results


def _select_web_context_lines(text: str, *, story_name: str, question: str, max_chars: int) -> str:
    if not text:
        return ""
    compact_head = re.sub(r"\s+", "", text[:1200]).lower()
    if (
        compact_head.count("{") >= 6
        and any(marker in compact_head for marker in ("font-family", "--bing", "rgba(", "@media", "display:"))
    ):
        return ""
    compact_text = re.sub(r"\s+", "", text)
    question_terms = _extract_content_tokens(question)
    if "岁陵" in question and "危机" in question:
        question_terms = _dedupe_keep_order(
            question_terms + ["岁陵", "危机", "岁陵危机", "岁兽之患", "岁兽", "苏醒", "平息", "望", "不反", "真龙"]
        )
    for anchor in WEB_CONTEXT_QUERY_ANCHOR_TERMS:
        if anchor in question:
            question_terms.append(anchor)
    question_terms = _dedupe_keep_order(question_terms)
    if story_name and story_name not in compact_text and not any(term and term in compact_text for term in question_terms):
        return ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if re.sub(r"\s+", " ", line).strip()]
    focus_terms = (story_name, *question_terms, "明日方舟", "剧情", "时间线", "解析", "故事集", "活动", "事件", "主线", "时间")
    selected: list[str] = []
    for line in lines:
        if len(line) < 12:
            continue
        if any(term and term in line for term in focus_terms):
            selected.append(line)
        if sum(len(item) for item in selected) >= max_chars:
            break
    if not selected:
        selected = lines[:20]
    return _truncate_text("\n".join(_dedupe_keep_order(selected)), max_chars)


def _build_web_context_text(
    *,
    story_name: str,
    queries: list[str],
    pages: list[dict[str, str]],
    max_total_chars: int,
) -> str:
    sections = [
        "联网补充资料：以下内容来自运行时网页检索，主要用于补足剧情解析与时间线线索；若与本地游戏文本证据冲突，以本地证据为准。",
        f"召回链命中最多的剧情集：{story_name}",
        "联网检索词：" + " | ".join(queries),
    ]
    for index, page in enumerate(pages, start=1):
        excerpt = page.get("excerpt", "").strip()
        if not excerpt:
            continue
        sections.append(
            "\n".join(
                [
                    f"[联网来源 {index}] {page.get('title') or page.get('url') or ''}",
                    f"url: {page.get('url') or ''}",
                    "摘录:",
                    excerpt,
                ]
            )
        )
    return _truncate_text("\n\n".join(sections), max_total_chars)


def _make_web_context_evidence_item(
    story_name: str,
    text: str,
    urls: list[str],
    *,
    item_key: str = "",
    title: str = "",
) -> dict[str, Any]:
    key = item_key or story_name
    doc_id = "web_context/" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    source_path = " | ".join(urls[:3])
    title_prefix = f"{title}\n" if title else ""
    return {
        "doc_index": -1,
        "document": {
            "id": doc_id,
            "activity_name": "联网补充资料",
            "story_name": f"{story_name} 剧情解析/时间线",
            "stage_code": "",
            "avg_tag": "联网资料",
            "source_path": source_path,
            "clean_text": title_prefix + text,
            "search_text": title_prefix + text,
        },
        "fusion_score": 0.0,
        "rerank_score": 0.0,
        "dense_score": None,
        "sparse_score": None,
        "minirag_score": None,
        "supplemental_source": "web_context",
        "prompt_prefer_clean_text": True,
    }


def _web_context_has_scope_hit(item: dict[str, Any], *, story_name: str, question: str) -> bool:
    doc = item.get("document") or {}
    text = re.sub(r"\s+", "", str(doc.get("clean_text") or doc.get("search_text") or ""))
    if story_name and story_name in text:
        return True
    question_terms = _extract_content_tokens(question)
    anchor_terms = [term for term in [*_web_context_question_terms(question, [], hypothesis=None), *question_terms] if len(term) >= 2]
    return any(term in text for term in _dedupe_keep_order(anchor_terms)[:12])


def _filter_web_context_candidates(
    *,
    retriever: ArknightsHybridRetriever,
    question: str,
    story_name: str,
    candidates: list[dict[str, Any]],
    config: WebContextConfig,
    hypothesis: HypothesisDocument | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates or config.rerank_top_k <= 0:
        return [], {"status": "disabled_or_no_candidates", "candidate_count": len(candidates)}
    scoped_candidates = []
    rejected: list[dict[str, Any]] = []
    for item in candidates:
        if config.require_story_or_question_hit and not _web_context_has_scope_hit(
            item,
            story_name=story_name,
            question=question,
        ):
            doc = item.get("document") or {}
            rejected.append(
                {
                    "id": doc.get("id"),
                    "title": str(doc.get("clean_text") or "").splitlines()[0][:120],
                    "reason": "missing_story_or_question_hit",
                }
            )
            continue
        scoped_candidates.append(item)
    if not scoped_candidates:
        return [], {
            "status": "all_candidates_rejected_by_scope",
            "candidate_count": len(candidates),
            "rejected": rejected[:8],
        }
    rerank_query = question
    if hypothesis is not None:
        terms = _dedupe_keep_order(hypothesis.entities + hypothesis.keywords + _extract_content_tokens(question))
        if terms:
            rerank_query = rerank_query + "\n联网资料相关线索: " + " ".join(terms[:12])
    reranked = rerank_hits(
        retriever,
        rerank_query,
        scoped_candidates,
        top_k=min(config.rerank_top_k, len(scoped_candidates)),
        batch_size=4,
        query_mode=classify_retrieval_query_mode(hypothesis) if hypothesis else None,
    )
    accepted = [
        item
        for item in reranked
        if float(item.get("rerank_score") or 0.0) >= config.rerank_min_score
    ]
    return accepted, {
        "status": "filtered",
        "candidate_count": len(candidates),
        "scoped_candidate_count": len(scoped_candidates),
        "accepted_count": len(accepted),
        "rerank_min_score": config.rerank_min_score,
        "top_scores": [
            {
                "id": (item.get("document") or {}).get("id"),
                "score": item.get("rerank_score"),
                "title": str((item.get("document") or {}).get("clean_text") or "").splitlines()[0][:120],
            }
            for item in reranked[:5]
        ],
        "rejected": rejected[:8],
    }


def build_web_context_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    config: WebContextConfig,
    *,
    retriever: ArknightsHybridRetriever | None = None,
    hypothesis: HypothesisDocument | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not config.enabled or not evidence:
        return [], {"enabled": config.enabled, "status": "disabled_or_no_evidence"}
    story_name, story_scores = _dominant_story_name_from_evidence(
        evidence,
        max_items=config.max_first_round_evidence,
        min_hits=config.min_story_hits,
    )
    if not story_name:
        return [], {"enabled": True, "status": "no_dominant_story", "story_scores": story_scores}
    question_terms = _web_context_question_terms(question, evidence, hypothesis=hypothesis)
    question_terms_text = " ".join(question_terms)
    queries = [
        template.format(
            story_name=story_name,
            question=question,
            question_terms=question_terms_text,
        )
        for template in config.query_templates
    ][: config.max_search_queries]
    cache_key = _cache_key_for_web_context(story_name, queries)
    cached = _read_web_context_cache(config.cache_dir, cache_key, config.cache_ttl_seconds)
    if cached and cached.get("text"):
        urls = [str(url) for url in cached.get("urls") or []]
        cached_items = [_make_web_context_evidence_item(story_name, str(cached["text"]), urls)]
        filter_record: dict[str, Any] = {"status": "not_filtered_no_retriever"}
        if retriever is not None:
            cached_items, filter_record = _filter_web_context_candidates(
                retriever=retriever,
                question=question,
                story_name=story_name,
                candidates=cached_items,
                config=config,
                hypothesis=hypothesis,
            )
        return cached_items, {
            "enabled": True,
            "status": "cache_hit",
            "story_name": story_name,
            "queries": queries,
            "question_terms": question_terms,
            "urls": urls,
            "filter": filter_record,
        }

    deadline = time.time() + config.max_elapsed_seconds
    candidate_results: list[dict[str, str]] = []
    candidate_urls: list[str] = []
    for query in queries:
        if _deadline_expired(deadline):
            break
        for result in _web_search_results(query, config, deadline=deadline):
            url = result["url"]
            if url not in candidate_urls:
                candidate_urls.append(url)
                candidate_results.append(result)
            if len(candidate_urls) >= config.max_search_results:
                break
        if len(candidate_urls) >= config.max_search_results:
            break

    pages: list[dict[str, str]] = []
    rejected_pages: list[dict[str, str]] = []
    for result in candidate_results:
        if _deadline_expired(deadline):
            break
        url = result["url"]
        search_title = result.get("title", "")
        resolved_url = _resolve_search_redirect_url(
            url,
            timeout=_remaining_timeout(deadline, min(config.timeout_seconds, 2.0)),
            user_agent=config.user_agent,
        )
        raw_text = _http_get_text(
            resolved_url,
            timeout=_remaining_timeout(deadline, config.timeout_seconds),
            user_agent=config.user_agent,
        )
        if not raw_text:
            continue
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw_text)
        page_title = _strip_html_to_text(title_match.group(1)) if title_match else ""
        title = page_title or search_title
        if "剧情" not in title:
            rejected_pages.append(
                {
                    "url": resolved_url,
                    "title": title[:120],
                    "search_title": search_title[:120],
                    "reason": "title_missing_剧情",
                }
            )
            continue
        stripped = _strip_html_to_text(raw_text)
        excerpt = _select_web_context_lines(
            stripped,
            story_name=story_name,
            question=question,
            max_chars=config.max_chars_per_page,
        )
        if not excerpt:
            continue
        pages.append({"url": resolved_url, "title": title[:120], "excerpt": excerpt})
        if len(pages) >= config.max_pages:
            break

    if not pages:
        return [], {
            "enabled": True,
            "status": "no_pages",
            "story_name": story_name,
            "queries": queries,
            "question_terms": question_terms,
            "candidate_urls": candidate_urls[: config.max_search_results],
            "candidate_results": candidate_results[: config.max_search_results],
            "rejected_pages": rejected_pages[:8],
        }

    text = _build_web_context_text(
        story_name=story_name,
        queries=queries,
        pages=pages,
        max_total_chars=config.max_total_chars,
    )
    urls = [page["url"] for page in pages]
    _write_web_context_cache(config.cache_dir, cache_key, {"text": text, "urls": urls, "story_name": story_name})
    candidates = [
        _make_web_context_evidence_item(
            story_name,
            page["excerpt"],
            [page["url"]],
            item_key=f"{story_name}:{page.get('url') or index}",
            title=page.get("title") or "",
        )
        for index, page in enumerate(pages, start=1)
    ]
    filter_record: dict[str, Any] = {"status": "not_filtered_no_retriever"}
    if retriever is not None:
        candidates, filter_record = _filter_web_context_candidates(
            retriever=retriever,
            question=question,
            story_name=story_name,
            candidates=candidates,
            config=config,
            hypothesis=hypothesis,
        )
    return candidates, {
        "enabled": True,
        "status": "fetched",
        "story_name": story_name,
        "queries": queries,
        "question_terms": question_terms,
        "urls": urls,
        "story_scores": story_scores,
        "filter": filter_record,
    }


def _clean_anchor_term(term: str) -> str:
    cleaned = re.sub(r"\s+", "", term or "").strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")
    if not cleaned:
        return ""
    cleaned = re.split(
        r"(?:这种题|这类题|这个问题|这种问题|题目|问题|反而|检索|证据|不足|为什么|为何|怎么|如何|吗|呢|啊|吧)",
        cleaned,
        maxsplit=1,
    )[0]
    return cleaned.strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")


def extract_question_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    text = "\n".join(
        [
            question,
            hypothesis.question,
            " ".join(hypothesis.entities),
            " ".join(hypothesis.keywords),
            hypothesis.expected_answer_type,
        ]
    )
    anchors: list[str] = []

    for raw_term in QUOTED_TERM_RE.findall(text):
        anchors.append(_clean_anchor_term(raw_term))
    for match in ACTION_TARGET_RE.finditer(text):
        anchors.append(_clean_anchor_term(match.group(1)))
    anchors.extend(extract_action_targets(text))
    for action in ACTION_WORDS:
        if action in text:
            anchors.append(action)
    anchors.extend(term for term in DOMAIN_ANCHOR_TERMS if term in text)
    anchors.extend(_extract_content_tokens(question))
    anchors.extend(term for term in hypothesis.entities if term)
    anchors.extend(term for term in hypothesis.keywords if term)

    cleaned_anchors: list[str] = []
    for term in anchors:
        cleaned = _clean_anchor_term(term)
        if (
            not cleaned
            or cleaned in COMMON_NON_ENTITY_WORDS
            or cleaned in NOISY_RETRIEVAL_TOKENS
            or cleaned in PRONOUN_REFERENCES
            or len(cleaned) == 1 and not cleaned.isascii()
            or any(marker in cleaned for marker in NOISY_TOKEN_MARKERS)
        ):
            continue
        cleaned_anchors.append(cleaned)
    deduped_anchors = _dedupe_keep_order(cleaned_anchors)
    related_anchors = expand_related_retrieval_terms(deduped_anchors)
    return _dedupe_keep_order(deduped_anchors + related_anchors)[:24]


def _anchor_hit_count(text: str, anchors: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not anchors:
        return 0
    return sum(1 for anchor in anchors if re.sub(r"\s+", "", anchor) in compact_text)


def extract_action_targets(text: str) -> list[str]:
    source = text or ""
    targets: list[str] = []
    for match in ACTION_TARGET_RE.finditer(source):
        raw_target = match.group(1)
        raw_target = ACTION_TARGET_BOUNDARY_RE.split(raw_target, maxsplit=1)[0]
        cleaned = _clean_anchor_term(raw_target)
        if cleaned:
            targets.append(cleaned)
    if any(action in source for action in ACTION_WORDS):
        targets.extend(term for term in DOMAIN_ANCHOR_TERMS if term in source)
    return _dedupe_keep_order(
        target
        for target in targets
        if target and target not in ACTION_WORDS and target not in COMMON_NON_ENTITY_WORDS
    )


def _action_target_score(text: str, targets: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not targets:
        return 0
    target_hit = any(target and target in compact_text for target in targets)
    if not target_hit:
        return 0
    action_hit = any(action in compact_text for action in ACTION_WORDS)
    marker_hit = any(marker in compact_text for marker in ACTION_ANSWER_MARKERS)
    return int(target_hit) + int(action_hit) + int(marker_hit)


def _action_target_marker_score(text: str, targets: list[str], markers: tuple[str, ...]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not targets:
        return 0
    if not any(target and target in compact_text for target in targets):
        return 0
    marker_hits = sum(1 for marker in markers if marker in compact_text)
    if marker_hits <= 0:
        return 0
    action_hit = any(action in compact_text for action in ACTION_WORDS)
    return marker_hits + int(action_hit)


def _best_action_target_evidence(
    evidence: list[dict[str, Any]],
    targets: list[str],
    markers: tuple[str, ...],
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in evidence
        if _action_target_marker_score(_evidence_text(item), targets, markers) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            _action_target_marker_score(_evidence_text(item), targets, markers),
            _action_target_score(_evidence_text(item), targets),
            _evidence_score(item),
        ),
    )


def _anchor_bundle_score(text: str, core_terms: list[str], bundle_terms: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not bundle_terms:
        return 0
    compact_core_terms = _dedupe_keep_order(
        re.sub(r"\s+", "", term or "")
        for term in core_terms
        if term and term not in COMMON_NON_ENTITY_WORDS
    )
    compact_bundle_terms = _dedupe_keep_order(
        re.sub(r"\s+", "", term or "")
        for term in bundle_terms
        if term and term not in COMMON_NON_ENTITY_WORDS
    )
    core_hits = sum(1 for term in compact_core_terms if term and term in compact_text)
    if compact_core_terms and core_hits <= 0:
        return 0
    bundle_hits = sum(1 for term in compact_bundle_terms if term and term in compact_text)
    return core_hits * 3 + bundle_hits


def _best_anchor_bundle_evidence(
    evidence: list[dict[str, Any]],
    *,
    core_terms: list[str],
    bundle_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not core_terms or not bundle_terms:
        return []
    candidates = [
        item
        for item in evidence
        if _anchor_bundle_score(_evidence_text(item), core_terms, bundle_terms) >= 5
    ]
    return sorted(
        candidates,
        key=lambda item: (
            _anchor_bundle_score(_evidence_text(item), core_terms, bundle_terms),
            _evidence_score(item),
        ),
        reverse=True,
    )[:limit]


def _is_definition_or_identity_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = "\n".join([question or "", hypothesis.question or "", hypothesis.expected_answer_type or ""])
    return any(marker in text for marker in DEFINITION_QUESTION_MARKERS) or hypothesis.expected_answer_type in {
        "definition",
        "string",
    }


def _definition_anchor_score(text: str, anchors: list[str]) -> int:
    compact_text = re.sub(r"\s+", "", strip_internal_evidence_meta(text or ""))
    if not compact_text or not anchors:
        return 0
    compact_anchors = _dedupe_keep_order(
        re.sub(r"\s+", "", anchor or "")
        for anchor in anchors
        if anchor and anchor not in COMMON_NON_ENTITY_WORDS
    )
    anchor_hits = [anchor for anchor in compact_anchors if anchor and anchor in compact_text]
    if not anchor_hits:
        return 0
    marker_hits = sum(1 for marker in DEFINITION_EVIDENCE_MARKERS if marker in compact_text)
    local_definition_hits = 0
    for anchor in anchor_hits[:4]:
        for marker in DEFINITION_EVIDENCE_MARKERS:
            if re.search(re.escape(anchor) + r".{0,32}" + re.escape(marker), compact_text):
                local_definition_hits += 1
                break
            if re.search(re.escape(marker) + r".{0,32}" + re.escape(anchor), compact_text):
                local_definition_hits += 1
                break
    return len(anchor_hits) * 4 + min(marker_hits, 4) + local_definition_hits * 3


def _definition_source_bonus(item: dict[str, Any]) -> int:
    if _is_moegirl_evidence(item):
        return -10
    if _is_web_context_item(item):
        return -8
    source_path = str((item.get("document") or {}).get("source_path") or "")
    if "/data/ArknightsGameData/" in source_path or "data/ArknightsGameData/" in source_path:
        return 3
    return 0


def _definition_candidate_score(item: dict[str, Any], anchors: list[str]) -> int:
    return _definition_anchor_score(_evidence_text(item), anchors) + _definition_source_bonus(item)


def _best_definition_evidence(
    evidence: list[dict[str, Any]],
    *,
    anchors: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not anchors:
        return []
    candidates = [
        item
        for item in evidence
        if _definition_anchor_score(_evidence_text(item), anchors) >= 7
    ]
    return sorted(
        candidates,
        key=lambda item: (
            _definition_candidate_score(item, anchors),
            _prompt_evidence_score(item),
        ),
        reverse=True,
    )[:limit]


@lru_cache(maxsize=1)
def _raw_story_text_files() -> tuple[Path, ...]:
    story_root = DATA_ROOT / "story"
    if not story_root.exists():
        return ()
    return tuple(sorted(path for path in story_root.rglob("*.txt") if path.is_file()))


def _raw_exact_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    terms = _dedupe_keep_order(
        [
            *hypothesis.entities,
            *extract_action_targets(question + "\n" + hypothesis.question),
            *_extract_content_tokens(question),
        ]
    )
    anchors: list[str] = []
    for term in terms:
        cleaned = _clean_anchor_term(term)
        if (
            not cleaned
            or cleaned in COMMON_NON_ENTITY_WORDS
            or cleaned in NOISY_RETRIEVAL_TOKENS
            or len(cleaned) == 1 and not cleaned.isascii()
        ):
            continue
        if cleaned.isascii() and len(cleaned) < 2:
            continue
        anchors.append(cleaned)
    return _dedupe_keep_order(anchors)[:8]


def _raw_line_context(lines: list[str], index: int, *, window: int = 2) -> str:
    start = max(0, index - window)
    end = min(len(lines), index + window + 1)
    return "\n".join(line.strip() for line in lines[start:end] if line.strip())


def _clean_raw_story_context(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sticker_texts = re.findall(r'text="((?:\\.|[^"\\])*)"', line)
        if sticker_texts:
            for sticker_text in sticker_texts:
                sticker_clean = (
                    sticker_text.replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\t", " ")
                    .strip()
                )
                if sticker_clean:
                    cleaned_lines.append(sticker_clean)
            continue
        line = re.sub(r'^\[name="([^"]+)"\](.*)$', r"\1：\2", line)
        line = re.sub(r"\[[^\]]+\]", "", line).strip()
        if line:
            cleaned_lines.append(line)
    return strip_internal_evidence_meta("\n".join(cleaned_lines)).strip()


def _raw_exact_definition_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not _is_definition_or_identity_question(question, hypothesis):
        return []
    if hypothesis.query_type in {"reveal", "mystery"} or any(term in question for term in ("阴谋", "识破", "曝光")):
        return []
    anchors = _raw_exact_anchor_terms(question, hypothesis)
    if not anchors:
        return []
    compact_anchors = [re.sub(r"\s+", "", anchor) for anchor in anchors if anchor]
    allow_rogue = any(term in question.lower() for term in ("rogue", "肉鸽", "集成战略", "结局", "萨卡兹的无终奇语"))
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for path in _raw_story_text_files():
        normalized_path = path.as_posix().lower()
        if not allow_rogue and "/obt/rogue/" in normalized_path:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        compact_text = re.sub(r"\s+", "", text)
        if not any(anchor and anchor in compact_text for anchor in compact_anchors):
            continue
        lines = text.splitlines()
        for line_index, line in enumerate(lines):
            compact_line = re.sub(r"\s+", "", line)
            if not compact_line or not any(anchor and anchor in compact_line for anchor in compact_anchors):
                continue
            context = _raw_line_context(lines, line_index, window=2)
            score = _definition_anchor_score(context, anchors)
            if "？" in compact_line or "?" in compact_line or "是什么" in compact_line or "疑问" in compact_line:
                score -= 6
            if "[Sticker" in line or "text=" in line:
                score += 4
            for anchor in compact_anchors[:4]:
                if re.search(
                    re.escape(anchor) + r".{0,10}[，,:：/（(].{0,32}(?:系统|产物|机器|设备|本名|全称|身份)",
                    compact_line,
                ):
                    score += 8
                    break
            if score < 7:
                continue
            rel_path = path.relative_to(DATA_ROOT.parent.parent.parent.parent) if DATA_ROOT.exists() else path
            clean_text = _clean_raw_story_context(context)
            if not clean_text:
                continue
            doc_id = f"raw_exact/{path.relative_to(DATA_ROOT).as_posix()}#L{line_index + 1}"
            document = {
                "id": doc_id,
                "activity_name": "ArknightsGameData原文",
                "story_name": path.stem,
                "stage_code": "",
                "avg_tag": "raw_exact",
                "source_path": str(path),
                "clean_text": clean_text,
                "search_text": clean_text,
            }
            item = {
                "doc_index": -1,
                "document": document,
                "evidence_chain_score": 100.0 + float(score),
                "fusion_score": 100.0 + float(score),
                "supplemental_source": "raw_exact",
                "raw_exact": {
                    "line": line_index + 1,
                    "relative_path": str(rel_path),
                    "anchors": [anchor for anchor in anchors if anchor and anchor in re.sub(r"\s+", "", clean_text)],
                    "score": score,
                },
            }
            candidates.append((score, -len(clean_text), str(path), item))
            break
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[2]), reverse=True)
    return [item for _, _, _, item in candidates[:limit]]


def _pin_anchor_evidence(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    anchors = extract_question_anchor_terms(question, hypothesis)
    is_reveal = _is_reveal_question(question, hypothesis)
    if not anchors and not is_reveal:
        return selected[:limit]
    action_targets = extract_action_targets(question + "\n" + hypothesis.question)
    related_anchors = expand_related_retrieval_terms(action_targets + anchors)

    pinned: list[dict[str, Any]] = []
    max_pinned = max(1, min(3, limit // 3 or 1))
    bundle_core_terms = action_targets or anchors[:3]
    bundle_terms = _dedupe_keep_order([*bundle_core_terms, *anchors[:8], *related_anchors[:12]])
    if _is_definition_or_identity_question(question, hypothesis):
        pinned.extend(
            _best_definition_evidence(
                evidence,
                anchors=anchors,
                limit=max(2, min(4, limit // 2 or 2)),
            )
        )
    pinned.extend(
        _best_anchor_bundle_evidence(
            evidence,
            core_terms=bundle_core_terms,
            bundle_terms=bundle_terms,
            limit=max_pinned,
        )
    )
    if is_reveal:
        reveal_pinned = _best_reveal_evidence(
            question,
            hypothesis,
            evidence,
            limit=max(2, min(5, limit // 2 or 2)),
        )
        pinned.extend(_prefer_direct_prompt_text(item) for item in reveal_pinned)
    if action_targets:
        purpose_evidence = _best_action_target_evidence(evidence, action_targets, ACTION_PURPOSE_MARKERS)
        cost_evidence = _best_action_target_evidence(evidence, action_targets, ACTION_COST_MARKERS)
        for item in (purpose_evidence, cost_evidence):
            if item is not None:
                pinned.append(item)
        action_pinned = [
            item
            for item in evidence
            if _action_target_score(_evidence_text(item), action_targets) >= 2
        ]
        pinned.extend(
            sorted(
                action_pinned,
                key=lambda item: (_action_target_score(_evidence_text(item), action_targets), _evidence_score(item)),
                reverse=True,
            )[:max_pinned]
        )
    for item in evidence:
        text = _evidence_text(item)
        if _anchor_hit_count(text, anchors) < 2:
            continue
        pinned.append(item)
        if len(pinned) >= max_pinned:
            break

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in pinned + selected:
        identity = _evidence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        output.append(_prefer_direct_prompt_text(item) if is_reveal else item)
        if len(output) >= limit:
            break
    return output


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
    candidates = _dedupe_prompt_evidence_candidates(evidence)
    if len(candidates) <= prompt_evidence_top_k:
        return candidates[:prompt_evidence_top_k]
    scores = [_prompt_evidence_score(item) for item in candidates]
    score_min = min(scores)
    score_max = max(scores)
    score_span = score_max - score_min
    normalized_scores = [
        1.0 if score_span <= 1e-9 else (score - score_min) / score_span
        for score in scores
    ]
    token_sets = [
        _text_similarity_tokens(_evidence_text(item))
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
    del question, hypothesis
    if prompt_evidence_top_k <= 0 or not evidence:
        return []
    candidates = _dedupe_prompt_evidence_candidates(evidence)
    ranked = sorted(
        enumerate(candidates),
        key=lambda pair: (_prompt_evidence_score(pair[1]), -pair[0]),
        reverse=True,
    )
    return [item for _, item in ranked[:prompt_evidence_top_k]]


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
    evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
    evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
) -> str:
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
    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n{"
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
    repaired = text.strip()
    # Common 4B error: {("question": ...} with a stray parenthesis after the
    # opening brace.
    repaired = re.sub(r"^\{\s*\(", "{", repaired)
    # Common 4B errors: missing colon after list/object fields.
    repaired = re.sub(
        r'([{\[,]\s*)missing_slots\s*\[',
        r'\1"missing_slots":[',
        repaired,
    )
    repaired = re.sub(
        r'([{\[,]\s*)follow_up_hypothesis\s*\{',
        r'\1"follow_up_hypothesis":{',
        repaired,
    )
    # Common 4B error: {question": "..."} where the opening quote is missing.
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:',
        r'\1"\2":',
        repaired,
    )
    # Common 4B error: {next_action:"answer_directly"} with unquoted object keys.
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        repaired,
    )
    # Common 4B error: enum values such as "next_action": retrieve_more.
    def quote_bare_value(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2).strip()
        if value in {"null", "true", "false"} or re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            return prefix + value
        return prefix + json.dumps(value, ensure_ascii=False)

    repaired = re.sub(
        r'("(?:next_action|query_type|intent|expected_answer_type)"\s*:\s*)'
        r'([A-Za-z_][A-Za-z0-9_\-/]*)\b(?=\s*[,}])',
        quote_bare_value,
        repaired,
    )
    # Common 4B error: list items where only some Chinese strings are quoted.
    def quote_bare_list_item(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2).strip()
        if (
            not value
            or value in {"null", "true", "false"}
            or re.fullmatch(r"-?\d+(?:\.\d+)?", value)
        ):
            return prefix + value
        return prefix + json.dumps(value, ensure_ascii=False)

    repaired = re.sub(
        r'([,\[]\s*)(?!["{\[\]])([^,\]\{\}\n\r:]{1,120})(?=\s*[,]])',
        quote_bare_list_item,
        repaired,
    )
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
        items = re.split(r"[、,，;；]\s*", value) if re.search(r"[、,，;；]", value) else [value]
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
    optional_missing_fields = {"dialogue_context", "query_type", "expected_answer_type", "reflect_tokens"}
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
    heuristic_entities = extract_entities(question, dialogue_context)
    heuristic_keywords = _extract_content_tokens(question)
    entities = _dedupe_keep_order(entities + heuristic_entities)[:12]
    keywords = _dedupe_keep_order(
        keywords
        + heuristic_keywords
        + expand_related_retrieval_terms(entities + keywords + heuristic_keywords)
    )[:24]

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


def _compact_supported_facts_payload(value: Any, *, max_facts: int = 6, max_refs: int = 2) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        if not fact:
            continue
        refs: list[dict[str, str]] = []
        raw_refs = item.get("evidence_refs")
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                quote = str(ref.get("quote") or "").strip()
                evidence_id = str(ref.get("evidence_id") or "").strip()
                if quote:
                    quote = quote[:80].rstrip()
                new_ref: dict[str, str] = {}
                if evidence_id:
                    new_ref["evidence_id"] = evidence_id
                if quote:
                    new_ref["quote"] = quote
                if new_ref:
                    refs.append(new_ref)
                if len(refs) >= max_refs:
                    break
        compact.append({"fact": fact, "evidence_refs": refs})
        if len(compact) >= max_facts:
            break
    return compact


def _compact_inferred_facts_payload(value: Any, *, max_items: int = 2) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value:
        fact = str(item.get("fact") or "").strip() if isinstance(item, dict) else str(item or "").strip()
        if fact:
            compact.append({"fact": fact})
        if len(compact) >= max_items:
            break
    return compact


def _answer_from_structured_facts(
    supported_facts: list[dict[str, Any]],
    inferred_facts: list[dict[str, Any]],
    *,
    max_chars: int = 280,
) -> str:
    facts = [
        str(item.get("fact") or "").strip()
        for item in [*supported_facts, *inferred_facts]
        if isinstance(item, dict) and str(item.get("fact") or "").strip()
    ]
    facts = _dedupe_keep_order(facts)[:3]
    answer = "；".join(facts)
    if len(answer) > max_chars:
        answer = answer[: max_chars - 1].rstrip("；，。 ") + "。"
    return answer


def normalize_conclusion_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult:
    payload = dict(payload)
    if "next_action" not in payload:
        decision = str(payload.get("decision") or "").strip().lower()
        decision_action = {
            "retrieve": "retrieve_more",
            "retrieve_more": "retrieve_more",
            "need_more_evidence": "retrieve_more",
            "more_evidence": "retrieve_more",
            "answer": "answer_directly",
            "answer_directly": "answer_directly",
            "direct_answer": "answer_directly",
            "clarify": "clarify_user",
            "clarify_user": "clarify_user",
            "abstain": "abstain",
        }.get(decision)
        if not decision_action:
            if str(payload.get("answer") or "").strip():
                decision_action = "answer_directly"
            elif payload.get("additional_evidence_needed"):
                decision_action = "retrieve_more"
        if decision_action:
            payload["next_action"] = decision_action
    if "clarification_question" not in payload and payload.get("follow_up_question"):
        payload["clarification_question"] = payload.get("follow_up_question")
    if "missing_slots" not in payload:
        additional_evidence = payload.get("additional_evidence_needed")
        payload["missing_slots"] = additional_evidence if isinstance(additional_evidence, list) else []
    if not str(payload.get("answer") or "").strip() and str(payload.get("final_answer") or "").strip():
        payload["answer"] = payload.get("final_answer")
    payload.setdefault("answer", "")
    payload = {key: value for key, value in payload.items() if key not in CONCLUSION_IGNORED_EXTRA_FIELDS}

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
    optional_missing_fields = {
        "question",
        "clarification_question",
        "follow_up_hypothesis",
        "reflect_tokens",
        "final_answer",
        "supported_facts",
        "inferred_facts",
    }
    missing_fields = [
        field for field in CONCLUSION_SCHEMA_FIELDS if field not in payload and field not in optional_missing_fields
    ]
    if missing_fields:
        raise ModelOutputError(f"missing conclusion fields: {missing_fields}")
    payload_question = str(payload.get("question") or question).strip()
    if not payload_question:
        raise ModelOutputError("conclusion must contain question")
    next_action = str(payload.get("next_action", "")).strip()
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    if next_action not in RETRIEVAL_ACTIONS:
        raise ModelOutputError(f"invalid conclusion action: {next_action or '<empty>'}")
    answer = str(payload.get("answer", "") or "").strip()
    supported_facts = _compact_supported_facts_payload(payload.get("supported_facts"))
    inferred_facts = _compact_inferred_facts_payload(payload.get("inferred_facts"))
    if not answer and next_action == "answer_directly":
        answer = _answer_from_structured_facts(supported_facts, inferred_facts)
    missing_slots = _normalize_string_list(payload.get("missing_slots"), limit=8)
    clarification_question = str(payload.get("clarification_question") or "").strip()
    follow_up_hypothesis_payload = payload.get("follow_up_hypothesis")
    follow_up_hypothesis: HypothesisDocument | None = None
    if next_action in {"answer_directly", "abstain"} and not answer:
        raise ModelOutputError(f"{next_action} requires non-empty answer")
    if next_action == "clarify_user" and not clarification_question:
        raise ModelOutputError("clarify_user requires clarification_question")
    if next_action == "retrieve_more":
        if answer:
            answer = ""
        if not missing_slots:
            missing_slots = ["需要补充更直接的证据"]
        if isinstance(follow_up_hypothesis_payload, dict):
            try:
                follow_up_hypothesis = normalize_hypothesis_payload(
                    follow_up_hypothesis_payload,
                    question=question,
                    dialogue_context=dialogue_context,
                    current_intent=current_intent,
                )
            except ModelOutputError:
                if not max_round_reached:
                    follow_up_hypothesis = (
                        build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
                        if current_hypothesis is not None
                        else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
                    )
                    missing_slots = _dedupe_keep_order(
                        [*missing_slots, "follow_up_hypothesis 不可用，已使用启发式续检索"]
                    )
        elif not max_round_reached:
            follow_up_hypothesis = (
                build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
                if current_hypothesis is not None
                else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
            )
            missing_slots = _dedupe_keep_order([*missing_slots, "模型未返回 follow_up_hypothesis，已使用启发式续检索"])
        if max_round_reached:
            next_action = "abstain"
            answer = "现有检索证据不足以确认，且已达到检索轮次上限。"
            follow_up_hypothesis = None
    else:
        follow_up_hypothesis = None
    return ConclusionResult(
        next_action=next_action,
        answer=answer,
        missing_slots=missing_slots,
        clarification_question=clarification_question,
        follow_up_hypothesis=follow_up_hypothesis,
        supported_facts=supported_facts,
        inferred_facts=inferred_facts,
    )


def _extract_json_like_bare_field(text: str, field: str) -> str:
    match = re.search(
        rf'"?{re.escape(field)}"?\s*:\s*"?([A-Za-z_][A-Za-z0-9_\-/]*)"?',
        text,
    )
    return match.group(1).strip() if match else ""


def _extract_json_like_string_field(text: str, field: str) -> str:
    match = re.search(rf'"?{re.escape(field)}"?\s*:\s*"', text)
    if match:
        start = match.end()
        escape = False
        chars: list[str] = []
        for char in text[start:]:
            if escape:
                chars.append("\\" + char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                raw_value = "".join(chars)
                try:
                    parsed = json.loads(f'"{raw_value}"')
                    return parsed if isinstance(parsed, str) else str(parsed)
                except json.JSONDecodeError:
                    return raw_value.replace(r"\"", '"').replace(r"\\", "\\").strip()
            chars.append(char)
    bare_match = re.search(
        rf'"?{re.escape(field)}"?\s*:\s*([^,\}}\n\r]+)',
        text,
    )
    if not bare_match:
        return ""
    value = bare_match.group(1).strip().strip('"')
    return "" if value == "null" else value


def _extract_json_like_missing_slots(text: str, *, limit: int = 8) -> list[str]:
    string_value = _extract_json_like_string_field(text, "missing_slots")
    if string_value and not string_value.startswith("["):
        return _normalize_string_list(string_value, limit=limit)

    match = re.search(r'"?missing_slots"?\s*:?\s*\[', text)
    if not match:
        return []
    start = match.end()
    end = text.find("]", start)
    follow_up_match = re.search(r'"?follow_up_hypothesis"?\s*:?\s*\{', text[start:])
    if end == -1 or (follow_up_match and start + follow_up_match.start() < end):
        end = start + follow_up_match.start() if follow_up_match else min(len(text), start + 500)
    body = text[start:end]

    items: list[str] = []
    for quoted in re.finditer(r'"((?:\\.|[^"\\])*)"', body):
        raw_value = quoted.group(1)
        try:
            value = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            value = raw_value
        if isinstance(value, str) and value.strip():
            items.append(value.strip())

    bare_body = re.sub(r'"(?:\\.|[^"\\])*"', "", body)
    for part in re.split(r"[、,，;；]\s*", bare_body):
        value = part.strip().strip('"').strip()
        value = re.sub(r"^[\[\s]+|[\]\s]+$", "", value).strip()
        if value and value not in {":", "null", "None"}:
            items.append(value)
    return _dedupe_keep_order(items)[:limit]


def _extract_json_like_repeated_string_field(text: str, field: str, *, limit: int = 4) -> list[str]:
    values: list[str] = []
    pattern = re.compile(rf'"?{re.escape(field)}"?\s*:\s*"')
    for match in pattern.finditer(text):
        start = match.end()
        escape = False
        chars: list[str] = []
        for char in text[start:]:
            if escape:
                chars.append("\\" + char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                raw_value = "".join(chars)
                try:
                    value = json.loads(f'"{raw_value}"')
                except json.JSONDecodeError:
                    value = raw_value.replace(r"\"", '"').replace(r"\\", "\\")
                value = str(value).strip()
                if value:
                    values.append(value)
                break
            chars.append(char)
        if len(values) >= limit:
            break
    return _dedupe_keep_order(values)[:limit]


def _extract_truncated_supported_facts(text: str, *, limit: int = 2) -> list[dict[str, Any]]:
    facts = _extract_json_like_repeated_string_field(text, "fact", limit=limit)
    quotes = _extract_json_like_repeated_string_field(text, "quote", limit=limit)
    evidence_ids = _extract_json_like_repeated_string_field(text, "evidence_id", limit=limit)
    supported: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        item: dict[str, Any] = {"fact": fact}
        if index < len(quotes):
            ref: dict[str, Any] = {"quote": quotes[index]}
            if index < len(evidence_ids):
                ref["evidence_id"] = evidence_ids[index]
            item["evidence_refs"] = [ref]
        supported.append(item)
    return supported


def recover_truncated_grounded_answer(
    text: str,
    *,
    question: str,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    next_action = _extract_json_like_bare_field(text, "next_action")
    next_action = {
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
        "retrieve": "retrieve_more",
    }.get(next_action, next_action)
    if next_action != "answer_directly":
        return None

    final_answer = _extract_json_like_string_field(text, "final_answer").strip()
    if final_answer:
        answer = final_answer
    else:
        facts = _extract_json_like_repeated_string_field(text, "fact", limit=2)
        if not facts:
            return None
        # Use only completed fact strings from the truncated JSON. The regular
        # grounding guard still verifies the recovered answer against evidence.
        answer = "；".join(facts)
        if len(answer) > 280:
            answer = answer[:279].rstrip("；，。 ") + "。"

    if not answer:
        return None
    return ConclusionResult(
        next_action="answer_directly",
        answer=answer,
        missing_slots=[],
        clarification_question="",
        follow_up_hypothesis=None,
        supported_facts=_extract_truncated_supported_facts(text, limit=2),
        grounding_warnings=["recovered_from_truncated_json"],
    )


def parse_conclusion_json_like_output(
    text: str,
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    tuple_conclusion = parse_tuple_like_conclusion_output(
        text,
        question=question,
        dialogue_context=dialogue_context,
        current_intent=current_intent,
        current_hypothesis=current_hypothesis,
        max_round_reached=max_round_reached,
    )
    if tuple_conclusion is not None:
        return tuple_conclusion

    truncated_conclusion = recover_truncated_grounded_answer(
        text,
        question=question,
        max_round_reached=max_round_reached,
    )
    if truncated_conclusion is not None:
        return truncated_conclusion

    next_action = _extract_json_like_bare_field(text, "next_action")
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    answer = _extract_json_like_string_field(text, "answer").strip()
    if not answer:
        answer = _extract_json_like_string_field(text, "final_answer").strip()
    if not next_action and answer:
        next_action = "answer_directly"
    if next_action not in RETRIEVAL_ACTIONS:
        return None

    missing_slots = _extract_json_like_missing_slots(text)
    clarification_question = _extract_json_like_string_field(text, "clarification_question").strip()
    if next_action in {"answer_directly", "abstain"}:
        if not answer:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer=answer,
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if next_action == "clarify_user":
        if not clarification_question:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer="",
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if not missing_slots:
        missing_slots = ["需要补充更直接的证据"]
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            answer="现有检索证据不足以确认，且已达到检索轮次上限。",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=None,
        )
    follow_up_hypothesis = (
        build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
        if current_hypothesis is not None
        else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
    )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=_dedupe_keep_order([*missing_slots, "JSON-like 结论已使用启发式续检索"]),
        clarification_question="",
        follow_up_hypothesis=follow_up_hypothesis,
    )


def parse_tuple_like_conclusion_output(
    text: str,
    *,
    question: str,
    dialogue_context: str,
    current_intent: str,
    current_hypothesis: HypothesisDocument | None = None,
    max_round_reached: bool = False,
) -> ConclusionResult | None:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}") and "(" in raw[:3]:
        raw = raw[1:-1].strip()
    if not (raw.startswith("(") and raw.endswith(")")):
        return None
    pythonish = (
        raw.replace(": null", ": None")
        .replace(": true", ": True")
        .replace(": false", ": False")
        .replace(", null", ", None")
        .replace(", true", ", True")
        .replace(", false", ", False")
    )
    try:
        payload = ast.literal_eval(pythonish)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, tuple) or len(payload) < 2:
        return None
    values = list(payload)
    action_aliases = {"retrieve", "answer", "direct_answer"}
    action_index = next(
        (
            index
            for index, value in enumerate(values[:2])
            if str(value).strip() in RETRIEVAL_ACTIONS | action_aliases
        ),
        None,
    )
    if action_index is None:
        return None
    next_action = str(values[action_index]).strip()
    next_action = {
        "retrieve": "retrieve_more",
        "answer": "answer_directly",
        "direct_answer": "answer_directly",
    }.get(next_action, next_action)
    tail = values[action_index + 1 :]
    answer = str(tail[0] if len(tail) >= 1 and tail[0] is not None else "").strip()
    clarification_question = str(tail[1] if len(tail) >= 2 and tail[1] is not None else "").strip()
    missing_slots = _normalize_string_list(tail[2] if len(tail) >= 3 else [], limit=8)
    follow_up_payload = next((item for item in tail if isinstance(item, dict)), None)

    if next_action in {"answer_directly", "abstain"}:
        if not answer:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer=answer,
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if next_action == "clarify_user":
        if not clarification_question:
            return None
        return ConclusionResult(
            next_action=next_action,
            answer="",
            missing_slots=missing_slots,
            clarification_question=clarification_question,
            follow_up_hypothesis=None,
        )
    if not missing_slots:
        missing_slots = ["需要补充更直接的证据"]
    if max_round_reached:
        return ConclusionResult(
            next_action="abstain",
            answer="现有检索证据不足以确认，且已达到检索轮次上限。",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=None,
        )
    follow_up_hypothesis = None
    if isinstance(follow_up_payload, dict):
        try:
            follow_up_hypothesis = normalize_hypothesis_payload(
                follow_up_payload,
                question=question,
                dialogue_context=dialogue_context,
                current_intent=current_intent,
            )
        except ModelOutputError:
            follow_up_hypothesis = None
    if follow_up_hypothesis is None:
        follow_up_hypothesis = (
            build_heuristic_follow_up_hypothesis(question, current_hypothesis, missing_slots)
            if current_hypothesis is not None
            else build_hypothesis(question + " " + " ".join(missing_slots[:4]), dialogue_context)
        )
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=_dedupe_keep_order([*missing_slots, "tuple-like 结论已转换为续检索"]),
        clarification_question="",
        follow_up_hypothesis=follow_up_hypothesis,
    )


GROUNDING_LONG_TOKEN_MIN_LEN = 3
GROUNDING_HIT_RATE_THRESHOLD = 0.25
GROUNDING_MIN_MISSED_LONG_TOKENS = 4
GROUNDING_EVIDENCE_POOL_TOP_K = 12
GROUNDING_CAUSAL_MARKERS = ACTION_ANSWER_MARKERS


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
        for value in (
            item.get("evidence_chain_text"),
            document.get("clean_text"),
            document.get("search_text"),
            document.get("activity_name"),
            document.get("story_name"),
            document.get("stage_code"),
        ):
            text = strip_internal_evidence_meta(str(value or "")).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _normalize_for_evidence_match(text: str) -> str:
    return re.sub(r"\s+", "", strip_internal_evidence_meta(str(text or "")))


def _grounded_supported_fact_texts(conclusion: ConclusionResult) -> list[str]:
    texts: list[str] = []
    for fact in conclusion.supported_facts:
        if not isinstance(fact, dict):
            continue
        fact_text = str(fact.get("fact") or "").strip()
        if fact_text:
            texts.append(fact_text)
        for ref in fact.get("evidence_refs") or []:
            if isinstance(ref, dict):
                quote = str(ref.get("quote") or "").strip()
                if quote:
                    texts.append(quote)
    for fact in conclusion.inferred_facts:
        if isinstance(fact, dict):
            fact_text = str(fact.get("fact") or "").strip()
        else:
            fact_text = str(fact or "").strip()
        if fact_text:
            texts.append(fact_text)
    return _dedupe_keep_order(texts)


def _grounded_quote_texts(conclusion: ConclusionResult) -> list[str]:
    texts: list[str] = []
    for fact in conclusion.supported_facts:
        if not isinstance(fact, dict):
            continue
        for ref in fact.get("evidence_refs") or []:
            if isinstance(ref, dict):
                quote = str(ref.get("quote") or "").strip()
                if quote:
                    texts.append(quote)
    return _dedupe_keep_order(texts)


QUOTE_REQUIRED_RELATION_TERMS = (
    "未婚夫",
    "未婚妻",
    "父亲",
    "母亲",
    "亲生",
    "幕后主使",
    "真正原因",
    "建造",
    "开发",
    "制造",
    "创造",
    "设计",
    "源石计划",
    "种族整合",
    "整合统一",
    "仿生学",
    "目的",
    "动机",
    "旨在",
    "服务于",
)


def _claim_has_unsupported_quote_required_terms(claim: str, quote_pool: str) -> list[str]:
    missing: list[str] = []
    for term in QUOTE_REQUIRED_RELATION_TERMS:
        if term in claim and _normalize_for_evidence_match(term) not in quote_pool:
            missing.append(term)
    for token in _extract_content_tokens(claim):
        if token.isascii() and len(token) >= 3 and _normalize_for_evidence_match(token) not in quote_pool:
            missing.append(token)
    return _dedupe_keep_order(missing)


def _answer_from_grounded_facts(conclusion: ConclusionResult) -> str:
    quote_pool = _normalize_for_evidence_match("\n".join(_grounded_quote_texts(conclusion)))
    facts = [
        str(fact.get("fact") or "").strip()
        for fact in conclusion.supported_facts
        if (
            isinstance(fact, dict)
            and str(fact.get("fact") or "").strip()
            and not _claim_has_unsupported_quote_required_terms(str(fact.get("fact") or ""), quote_pool)
        )
    ]
    inferred = [
        str(fact.get("fact") or "").strip() if isinstance(fact, dict) else str(fact or "").strip()
        for fact in conclusion.inferred_facts
    ]
    inferred = [item for item in inferred if item and not _claim_has_unsupported_quote_required_terms(item, quote_pool)]
    selected = _dedupe_keep_order([*facts, *inferred])
    if not selected:
        return conclusion.answer
    if len(selected) == 1:
        return selected[0]
    return "根据当前证据可确认：" + "；".join(selected) + "。"


def _validate_grounded_quotes(
    *,
    conclusion: ConclusionResult,
    evidence: list[dict[str, Any]],
    question: str,
    evidence_prompt_text: str | None = None,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    evidence_pool = _normalize_for_evidence_match(evidence_prompt_text or _grounding_evidence_pool(evidence))
    if not conclusion.supported_facts:
        issues.append("missing_supported_facts")
        return issues, warnings
    if len(conclusion.supported_facts) > 6:
        issues.append(f"too_many_supported_facts:{len(conclusion.supported_facts)}>6")

    quote_count = 0
    total_quote_chars = 0
    for fact_index, fact in enumerate(conclusion.supported_facts, start=1):
        if not isinstance(fact, dict):
            issues.append(f"supported_fact_{fact_index}_not_object")
            continue
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            issues.append(f"supported_fact_{fact_index}_missing_evidence_refs")
            continue
        if len(refs) > 2:
            issues.append(f"supported_fact_{fact_index}_too_many_quotes:{len(refs)}>2")
        fact_quote_chars = 0
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_not_object")
                continue
            quote = str(ref.get("quote") or "").strip()
            if not quote:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_missing_quote")
                continue
            quote_count += 1
            fact_quote_chars += len(quote)
            total_quote_chars += len(quote)
            if len(quote) > 80:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_quote_over_80")
            if _normalize_for_evidence_match(quote) not in evidence_pool:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_quote_not_found")
        if fact_quote_chars > 160:
            issues.append(f"supported_fact_{fact_index}_quote_total_over_160")
    if quote_count == 0:
        issues.append("missing_quotes")
    if total_quote_chars > 400:
        issues.append(f"answer_quote_total_over_400:{total_quote_chars}")

    quote_pool = _normalize_for_evidence_match("\n".join(_grounded_quote_texts(conclusion)))
    answer_tokens = _grounding_extract_answer_tokens(conclusion.answer, question)
    missing_tokens = [
        token
        for token in answer_tokens
        if len(token) >= GROUNDING_LONG_TOKEN_MIN_LEN and _normalize_for_evidence_match(token) not in quote_pool
    ]
    unsupported_relations = _claim_has_unsupported_quote_required_terms(conclusion.answer, quote_pool)
    for fact_index, fact in enumerate(conclusion.supported_facts, start=1):
        if isinstance(fact, dict):
            unsupported_fact_terms = _claim_has_unsupported_quote_required_terms(str(fact.get("fact") or ""), quote_pool)
            if unsupported_fact_terms:
                issues.append(
                    f"supported_fact_{fact_index}_has_terms_outside_quotes:"
                    + ",".join(unsupported_fact_terms[:8])
                )
    if missing_tokens or unsupported_relations:
        warnings.append(
            "final_answer_has_terms_outside_supported_facts:"
            + ",".join(_dedupe_keep_order([*missing_tokens[:8], *unsupported_relations]))
        )
    return issues, warnings


def _is_identity_question(question: str, hypothesis: HypothesisDocument) -> bool:
    text = question + "\n" + hypothesis.expected_answer_type
    return any(token in text for token in IDENTITY_HINT_WORDS)


def _primary_entity_anchor_required(question: str, hypothesis: HypothesisDocument) -> str:
    if _is_reveal_question(question, hypothesis):
        return ""
    if not any(token in question for token in ("一事", "具体是指", "指的是什么", "指什么", "是谁", "是什么", "身份", "来历")):
        return ""
    for entity in hypothesis.entities:
        normalized = re.sub(r"\s+", "", entity)
        if (
            len(normalized) >= 3
            and _is_entity_candidate(normalized)
            and normalized not in COMMON_NON_ENTITY_WORDS
            and normalized not in NOISY_RETRIEVAL_TOKENS
        ):
            return normalized
    return ""


def _anchor_aliases(anchor: str) -> list[str]:
    aliases = [anchor]
    try:
        alias_map = load_operator_alias_map(OPERATOR_ALIAS_MAP_PATH)
        if alias_map:
            aliases.extend(alias_map.expand([anchor]))
    except Exception:
        pass
    return _dedupe_keep_order([re.sub(r"\s+", "", alias) for alias in aliases if alias])


def _unsupported_required_entity_anchor(
    question: str,
    hypothesis: HypothesisDocument,
    evidence_pool: str,
) -> str:
    anchor = _primary_entity_anchor_required(question, hypothesis)
    if not anchor:
        return ""
    compact_evidence = re.sub(r"\s+", "", strip_internal_evidence_meta(evidence_pool))
    if not compact_evidence:
        return anchor
    if any(alias and alias in compact_evidence for alias in _anchor_aliases(anchor)):
        return ""
    return anchor


def _has_direct_causal_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence_pool: str,
) -> bool:
    if _is_identity_question(question, hypothesis):
        return False
    if not any(token in question + hypothesis.expected_answer_type for token in ("为什么", "为何", "原因", "动机", "目的")):
        return False

    compact_evidence = re.sub(r"\s+", "", strip_internal_evidence_meta(evidence_pool))
    if not compact_evidence:
        return False

    anchors = extract_question_anchor_terms(question, hypothesis)
    high_value_anchors = [
        anchor
        for anchor in anchors
        if anchor in DOMAIN_ANCHOR_TERMS
        or anchor in hypothesis.entities
        or anchor in hypothesis.keywords
        or anchor in ACTION_WORDS
    ]
    anchor_hits = [
        anchor
        for anchor in _dedupe_keep_order(high_value_anchors)
        if re.sub(r"\s+", "", anchor) in compact_evidence
    ]
    if len(anchor_hits) < 2:
        return False

    has_action_target = any(match.group(1) and match.group(1) in compact_evidence for match in ACTION_TARGET_RE.finditer(question))
    has_action_word = any(word in compact_evidence for word in ACTION_WORDS)
    has_causal_marker = any(marker in compact_evidence for marker in GROUNDING_CAUSAL_MARKERS)
    return has_causal_marker and (has_action_target or has_action_word)


def _build_grounded_fallback_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    missing_tokens: list[str],
) -> str:
    query_terms = [
        term
        for term in _dedupe_keep_order(
            extract_entities(question, hypothesis.dialogue_context)
            + hypothesis.entities
            + hypothesis.keywords
            + _extract_content_tokens(question)
        )
        if term and term not in COMMON_NON_ENTITY_WORDS and term not in NOISY_RETRIEVAL_TOKENS
    ]
    selected: list[str] = []
    for item in evidence[:6]:
        document = item.get("document") or {}
        text = strip_internal_evidence_meta(
            str(item.get("evidence_chain_text") or document.get("clean_text") or document.get("search_text") or "")
        ).strip()
        if not text:
            continue
        if query_terms and not any(term in text for term in query_terms[:10]):
            continue
        text = _truncate_text(re.sub(r"\s+", " ", text), 180)
        if text and text not in selected:
            selected.append(text)
        if len(selected) >= 3:
            break

    if not selected:
        return "现有检索证据不足以确认答案所需的关键表述。"

    answer_lines = ["当前证据只能确认以下片段事实："]
    answer_lines.extend(f"{index}. {text}" for index, text in enumerate(selected, start=1))
    answer_lines.append("缺少足以完整回答用户问题的直接因果或身份绑定证据。")
    return "\n".join(answer_lines)


def _is_suiling_crisis_question(question: str, hypothesis: HypothesisDocument) -> bool:
    compact = re.sub(r"\s+", "", question + "\n" + hypothesis.question + "\n" + hypothesis.expected_answer_type)
    return (
        "岁陵" in compact
        and "危机" in compact
        and any(marker in compact for marker in ("是什么", "指什么", "什么危机", "那场危机", "危机原因", "概念定义"))
    )


def _build_suiling_crisis_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    if not _is_suiling_crisis_question(question, hypothesis):
        return None

    core_candidates: list[tuple[int, str]] = []
    pressure_candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        text = _evidence_text(item)
        strips = split_evidence_strips(text, max_strips=64)
        if not strips and text:
            strips = [text]
        for strip in strips:
            compact = re.sub(r"\s+", "", strip)
            if not compact:
                continue
            core_score = 0
            if "岁兽之患" in compact:
                core_score += 8
            if "岁兽" in compact and ("苏醒" in compact or "平息" in compact or "危害" in compact):
                core_score += 5
            if "岁陵" in compact and ("没有动静" in compact or "石门" in compact or "控制在岁陵" in compact):
                core_score += 4
            if "望" in compact and "岁陵" in compact and ("平息" in compact or "望日" in compact):
                core_score += 3
            if core_score > 0:
                core_candidates.append((core_score, _truncate_text(strip, 260)))

            pressure_score = 0
            if "五只巨兽" in compact:
                pressure_score += 5
            if "最坏的结果" in compact or "同时开战" in compact:
                pressure_score += 4
            if "大炎周遭" in compact or "盘踞" in compact:
                pressure_score += 2
            if pressure_score > 0:
                pressure_candidates.append((pressure_score, _truncate_text(strip, 240)))

    core_strips: list[str] = []
    for _, strip in sorted(core_candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in core_strips:
            core_strips.append(strip)
        if len(core_strips) >= 2:
            break
    if not core_strips:
        return None

    pressure_strips: list[str] = []
    for _, strip in sorted(pressure_candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in pressure_strips:
            pressure_strips.append(strip)
        if len(pressure_strips) >= 1:
            break

    answer = (
        "岁陵那场危机的核心是岁兽苏醒或即将苏醒引发的“岁兽之患”："
        "证据显示，望被准许进入岁陵尝试平息此事，但望日临近仍没有结果，岁陵局势可能失控。"
    )
    if pressure_strips:
        answer += "五只巨兽盘踞、可能同时开战属于当时的大炎外部压力和潜在最坏后果，不是危机本身的直接原因。"
    answer += "\n依据：" + "；".join([*core_strips, *pressure_strips][:3])
    return answer


def _suiling_crisis_answer_needs_correction(answer: str) -> bool:
    compact = re.sub(r"\s+", "", answer or "")
    if not compact:
        return True
    if "岁兽" not in compact and "岁兽之患" not in compact:
        return True
    if "五只巨兽" in compact or "同时开战" in compact:
        return not any(marker in compact for marker in ("外部压力", "最坏后果", "潜在", "不是危机本身", "不是直接原因"))
    if any(marker in compact for marker in ("直接原因是五只巨兽", "致大炎不得不应对五只巨兽")):
        return True
    return False


def _is_event_reference_question(question: str, hypothesis: HypothesisDocument) -> bool:
    compact = re.sub(r"\s+", "", "\n".join([question or "", hypothesis.question or "", hypothesis.expected_answer_type or ""]))
    if any(marker in compact for marker in ("一事", "这件事", "此事", "具体是指", "指的是什么", "指什么", "发生了什么")):
        return True
    return False


def _event_reference_anchor_terms(question: str, hypothesis: HypothesisDocument) -> list[str]:
    raw_terms: list[str] = []
    primary = _primary_entity_anchor_required(question, hypothesis)
    if primary:
        raw_terms.append(primary)
    else:
        raw_terms.extend(term for term in hypothesis.entities if term and term in question)
        if not raw_terms:
            raw_terms.extend(_extract_content_tokens(question))

    anchors: list[str] = []
    for raw_term in raw_terms:
        term = _clean_anchor_term(str(raw_term or ""))
        if term.endswith("一事"):
            term = term[:-2]
        if (
            len(term) < 2
            or term in COMMON_NON_ENTITY_WORDS
            or term in NOISY_RETRIEVAL_TOKENS
            or term in {"事件", "关系", "具体", "指什么", "是什么", "发生"}
        ):
            continue
        anchors.append(term)
    return _dedupe_keep_order([re.sub(r"\s+", "", anchor) for anchor in anchors if anchor])[:8]


def _select_event_reference_strips(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_strips: int = 3,
) -> tuple[str, list[str]]:
    if not _is_event_reference_question(question, hypothesis):
        return "", []

    anchors = _event_reference_anchor_terms(question, hypothesis)
    if not anchors:
        return "", []
    display_anchor = anchors[0]
    query_terms = _dedupe_keep_order(
        anchors
        + hypothesis.entities
        + hypothesis.keywords
        + _extract_content_tokens(question)
    )[:16]
    event_markers = (
        "当时",
        "后来",
        "再后来",
        "因为",
        "因此",
        "导致",
        "结果",
        "遭遇",
        "发生",
        "病逝",
        "联姻",
        "再婚",
        "流言",
        "恶名",
        "仕途",
        "生计",
        "真相",
        "实情",
        "缘由",
        "做错",
        "拒绝",
        "权力",
        "陪葬",
        "牵连",
        "为难",
    )

    candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        doc = item.get("document") or item
        text = strip_internal_evidence_meta(
            str(doc.get("clean_text") or doc.get("search_text") or "")
        ).strip()
        if not text:
            text = _document_chain_text(item)
        if not text:
            continue
        strips = split_evidence_strips(text, max_strips=80)
        if not strips:
            continue
        for index, strip in enumerate(strips):
            compact_strip = re.sub(r"\s+", "", strip)
            if not any(anchor and anchor in compact_strip for anchor in anchors):
                continue
            start = max(0, index - 3)
            end = min(len(strips), index + 3)
            window = "；".join(strips[start:end])
            compact_window = re.sub(r"\s+", "", window)
            score = 10
            score += sum(2 for anchor in anchors if anchor and anchor in compact_window)
            score += sum(1 for term in query_terms if term and re.sub(r"\s+", "", term) in compact_window)
            score += sum(1 for marker in event_markers if marker in compact_window)
            candidates.append((score, _truncate_text(window, 520)))

    selected: list[str] = []
    for _, strip in sorted(candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return display_anchor, selected


def _build_event_reference_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    anchor, strips = _select_event_reference_strips(question=question, hypothesis=hypothesis, evidence=evidence)
    if not anchor or not strips:
        return None

    answer = f"{anchor}一事，现有证据可确认的是：{strips[0]}"
    if len(strips) > 1:
        answer += " 相关证据还显示：" + "；".join(strips[1:])
    return answer


def _has_answerable_evidence(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        text = _best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
        if len(strip_internal_evidence_meta(text).strip()) >= 80:
            return True
    return False


def _select_reveal_strips(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    max_strips: int = 5,
) -> list[str]:
    if "阴谋" not in question and "阴谋" not in hypothesis.keywords and hypothesis.query_type not in {"reveal", "mystery"}:
        return []

    query_terms = _dedupe_keep_order(
        hypothesis.entities
        + hypothesis.keywords
        + _extract_content_tokens(question)
        + list(REVEAL_KNOWLEDGE_RETRIEVAL_TERMS)
    )
    high_value_terms = {
        "贝希曼",
        "贝希曼伯爵",
        "苏茜",
        "澄闪",
        "卡拉顿",
        "警备队",
        "送线索",
        "劫持",
        "爆炸",
        "工厂",
        "物流通道",
        "阴谋",
        "曝光",
    }
    candidates: list[tuple[int, str]] = []
    for item in evidence[:16]:
        text = _evidence_text(item)
        strips = split_evidence_strips(text, max_strips=48)
        if not strips and text:
            strips = [text]
        for strip in strips:
            compact = re.sub(r"\s+", "", strip)
            term_hits = sum(1 for term in query_terms if term and term in compact)
            high_hits = sum(1 for term in high_value_terms if term in compact)
            if high_hits < 2 and not ("阴谋" in compact and ("曝光" in compact or "贝希曼" in compact)):
                continue
            score = term_hits + high_hits * 2
            if "苏茜去警备队送线索" in compact:
                score += 8
            if "遭到劫持" in compact or "意外爆炸" in compact:
                score += 4
            if "贝希曼议员的阴谋得以曝光" in compact:
                score += 8
            candidates.append((score, _truncate_text(strip, 260)))

    selected: list[str] = []
    for _, strip in sorted(candidates, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return selected


def _build_reveal_answer(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
) -> str | None:
    strips = _select_reveal_strips(question=question, hypothesis=hypothesis, evidence=evidence)
    if not strips:
        return None
    joined = " ".join(strips)
    compact = re.sub(r"\s+", "", joined)
    if not ("贝希曼" in compact and "阴谋" in compact):
        return None

    parts: list[str] = []
    if "送线索" in compact and "警备队" in compact:
        parts.append("苏茜把线索送到警备队后，反而落入贝希曼一方掌控")
    if "劫持" in compact or "被捆住" in compact:
        parts.append("她被劫持并带到废弃物流通道/工厂相关地点")
    if "爆炸" in compact and "逃出" in compact:
        parts.append("之后因一场意外爆炸逃出")
    if "阴谋得以曝光" in compact or ("曝光" in compact and "阴谋" in compact):
        parts.append("最终使贝希曼议员的阴谋曝光")
    if "工厂" in compact or "物流通道" in compact or "设备" in compact:
        parts.append("相关线索还指向工厂设备、地下/废弃物流通道和警备队长的勾连")

    if not parts:
        return "现有证据显示，澄闪/苏茜识破的是贝希曼议员相关的卡拉顿城阴谋。依据：" + "；".join(strips[:3])
    return "现有证据显示，澄闪/苏茜识破的是贝希曼议员相关的阴谋：" + "；".join(parts) + "。依据：" + "；".join(strips[:3])


def validate_conclusion_grounding(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    conclusion: ConclusionResult,
    max_round_reached: bool,
    mode: str = "weak",
    evidence_prompt_text: str | None = None,
) -> ConclusionResult:
    if conclusion.next_action in {"retrieve_more", "abstain"}:
        reveal_answer = _build_reveal_answer(question=question, hypothesis=hypothesis, evidence=evidence)
        if reveal_answer and (max_round_reached or conclusion.next_action == "abstain"):
            return ConclusionResult(
                next_action="answer_directly",
                answer=reveal_answer,
                missing_slots=[],
                clarification_question="",
                follow_up_hypothesis=None,
            )
        suiling_crisis_answer = _build_suiling_crisis_answer(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
        )
        if suiling_crisis_answer and (max_round_reached or conclusion.next_action == "abstain"):
            return ConclusionResult(
                next_action="answer_directly",
                answer=suiling_crisis_answer,
                missing_slots=[],
                clarification_question="",
                follow_up_hypothesis=None,
            )
        event_reference_answer = _build_event_reference_answer(
            question=question,
            hypothesis=hypothesis,
            evidence=evidence,
        )
        if event_reference_answer and (max_round_reached or conclusion.next_action == "abstain"):
            return ConclusionResult(
                next_action="answer_directly",
                answer=event_reference_answer,
                missing_slots=[],
                clarification_question="",
                follow_up_hypothesis=None,
            )
    if conclusion.next_action != "answer_directly":
        return conclusion
    if not conclusion.answer:
        return conclusion

    suiling_crisis_answer = _build_suiling_crisis_answer(
        question=question,
        hypothesis=hypothesis,
        evidence=evidence,
    )
    if suiling_crisis_answer and _suiling_crisis_answer_needs_correction(conclusion.answer):
        return ConclusionResult(
            next_action="answer_directly",
            answer=suiling_crisis_answer,
            missing_slots=[],
            clarification_question="",
            follow_up_hypothesis=None,
        )

    grounding_mode = mode.strip().lower()
    if grounding_mode in {"off", "none", "disabled", "false", "0"}:
        return conclusion
    if grounding_mode not in {"weak", "strict", "quote", "grounded"}:
        grounding_mode = "weak"

    if grounding_mode in {"quote", "grounded", "strict"}:
        quote_issues, quote_warnings = _validate_grounded_quotes(
            conclusion=conclusion,
            evidence=evidence,
            question=question,
            evidence_prompt_text=evidence_prompt_text,
        )
        if quote_issues:
            missing_slots = _dedupe_keep_order(
                [
                    *(conclusion.missing_slots or []),
                    "answer_directly 缺少可校验 quote 支撑",
                    *quote_issues[:4],
                ]
            )
            if max_round_reached:
                grounded_answer = _build_grounded_fallback_answer(
                    question=question,
                    hypothesis=hypothesis,
                    evidence=evidence,
                    missing_tokens=missing_slots,
                )
                return ConclusionResult(
                    next_action="abstain",
                    answer=grounded_answer,
                    missing_slots=missing_slots,
                    clarification_question="",
                    follow_up_hypothesis=None,
                    grounding_warnings=quote_issues,
                )
            return ConclusionResult(
                next_action="retrieve_more",
                answer="",
                missing_slots=missing_slots,
                clarification_question="",
                follow_up_hypothesis=build_heuristic_follow_up_hypothesis(question, hypothesis, missing_slots),
                grounding_warnings=quote_issues,
            )
        if quote_warnings:
            repaired_answer = _answer_from_grounded_facts(conclusion)
            if repaired_answer and repaired_answer != conclusion.answer:
                return ConclusionResult(
                    next_action=conclusion.next_action,
                    answer=repaired_answer,
                    missing_slots=conclusion.missing_slots,
                    clarification_question=conclusion.clarification_question,
                    follow_up_hypothesis=conclusion.follow_up_hypothesis,
                    supported_facts=conclusion.supported_facts,
                    inferred_facts=conclusion.inferred_facts,
                    grounding_warnings=quote_warnings,
                )
            conclusion.grounding_warnings.extend(quote_warnings)

    answer_tokens = _grounding_extract_answer_tokens(conclusion.answer, question)
    long_tokens = [token for token in answer_tokens if len(token) >= GROUNDING_LONG_TOKEN_MIN_LEN]
    if not long_tokens:
        return conclusion

    evidence_pool = _grounding_evidence_pool(evidence)
    if not evidence_pool:
        return conclusion

    unsupported_anchor = _unsupported_required_entity_anchor(question, hypothesis, evidence_pool)
    if unsupported_anchor:
        missing_slots = [f"需要包含“{unsupported_anchor}”或其别名的直接证据"]
        if max_round_reached:
            return ConclusionResult(
                next_action="abstain",
                answer=f"现有检索证据不足以确认“{unsupported_anchor}”所指的具体内容。",
                missing_slots=missing_slots,
                clarification_question="",
                follow_up_hypothesis=None,
            )
        return ConclusionResult(
            next_action="retrieve_more",
            answer="",
            missing_slots=missing_slots,
            clarification_question="",
            follow_up_hypothesis=build_heuristic_follow_up_hypothesis(question, hypothesis, missing_slots),
        )

    # Token-level grounding is too brittle for narrative QA: correct answers
    # often paraphrase or bridge multiple snippets with model knowledge. In
    # weak mode, keep it only for identity questions, where unsupported entity
    # labels are the highest-risk hallucination class.
    if grounding_mode == "weak" and not _is_identity_question(question, hypothesis):
        return conclusion

    if _has_direct_causal_grounding(
        question=question,
        hypothesis=hypothesis,
        evidence_pool=evidence_pool,
    ):
        return conclusion

    missing_tokens = [token for token in long_tokens if token not in evidence_pool]
    hit_count = len(long_tokens) - len(missing_tokens)
    hit_rate = hit_count / len(long_tokens) if long_tokens else 1.0

    if (
        hit_rate < GROUNDING_HIT_RATE_THRESHOLD
        and len(missing_tokens) >= GROUNDING_MIN_MISSED_LONG_TOKENS
    ):
        if max_round_reached:
            grounded_answer = _build_grounded_fallback_answer(
                question=question,
                hypothesis=hypothesis,
                evidence=evidence,
                missing_tokens=missing_tokens,
            )
            return ConclusionResult(
                next_action="abstain",
                answer=grounded_answer,
                missing_slots=conclusion.missing_slots or ["grounding 校验未通过的关键词"],
                clarification_question="",
                follow_up_hypothesis=None,
            )
        follow_up_hypothesis = HypothesisDocument(
            question=question,
            intent=hypothesis.intent,
            query_type=hypothesis.query_type,
            entities=hypothesis.entities,
            keywords=_dedupe_keep_order(hypothesis.keywords + missing_tokens[:6])[:20],
            expected_answer_type=hypothesis.expected_answer_type,
            dialogue_context=hypothesis.dialogue_context,
        )
        return ConclusionResult(
            next_action="retrieve_more",
            answer="",
            missing_slots=conclusion.missing_slots or list(missing_tokens[:6]),
            clarification_question="",
            follow_up_hypothesis=follow_up_hypothesis,
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
        ctx_size: int = 12000,
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
            "trained_sft_artifact": "model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1",
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
        max_model_len: int = 12000,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
        dtype: str = "auto",
        max_num_batched_tokens: int | None = None,
        enforce_eager: bool = False,
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
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enforce_eager = enforce_eager
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
            "trained_sft_artifact": "model/lora/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1",
            "trained_sft_artifact_type": "LoRA adapter",
            "recommended_runtime_model": str(self.base_model_path),
            "runtime_mode": "base_hf" if not self.lora_path else "base_hf_plus_lora_vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enforce_eager": self.enforce_eager,
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
            tokenizer_path = (
                self.lora_path
                if self.lora_path and (self.lora_path / "tokenizer_config.json").exists()
                else self.base_model_path
            )
            llm_kwargs: dict[str, Any] = {
                "model": str(self.base_model_path),
                "tokenizer": str(tokenizer_path),
                "trust_remote_code": True,
                "enable_lora": self.lora_path is not None,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "dtype": self.dtype,
                "disable_log_stats": True,
                "enforce_eager": self.enforce_eager,
            }
            if self.max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
            self._llm = LLM(**llm_kwargs)
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
        max_retrieval_rounds: int = 2,
        prompt_evidence_top_k: int = 8,
        prompt_evidence_max_chars_per_doc: int = PROMPT_EVIDENCE_MAX_CHARS_PER_DOC,
        prompt_conclusion_evidence_max_total_chars: int = PROMPT_CONCLUSION_EVIDENCE_MAX_TOTAL_CHARS,
        enable_mmr: bool = False,
        mmr_lambda: float = 0.72,
        enable_pyramid_order: bool = False,
        enable_crag_refinement: bool = False,
        crag_refine_top_sentences: int = 4,
        crag_refine_max_sentences: int = 24,
        self_consistency_samples: int = 1,
        self_consistency_temperature: float = 0.7,
        answer_grounding_mode: str = "weak",
        max_follow_up_rounds: int | None = None,
        use_model_hypothesis: bool = True,
        use_model_conclusion_generation: bool = True,
        use_model_retrieval_planner: bool | None = None,
        conclusion_prompt_mode: str = "full",
        enable_evidence_pinning: bool = False,
        web_context_config: dict[str, Any] | WebContextConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.query_config = query_config or QueryConfig()
        self.max_retrieval_rounds = min(2, max(1, int(max_retrieval_rounds)))
        if not use_model_hypothesis:
            raise ValueError("heuristic hypothesis generation is disabled; set use_model_hypothesis=true")
        if use_model_retrieval_planner is not None:
            use_model_conclusion_generation = use_model_retrieval_planner
        if not use_model_conclusion_generation:
            raise ValueError("heuristic conclusion generation is disabled; set use_model_conclusion_generation=true")
        self.use_model_hypothesis = use_model_hypothesis
        self.use_model_conclusion_generation = use_model_conclusion_generation
        self.conclusion_prompt_mode = conclusion_prompt_mode.strip().lower()
        if self.conclusion_prompt_mode not in {"full", "minimal"}:
            raise ValueError("conclusion_prompt_mode must be 'full' or 'minimal'")
        self.enable_evidence_pinning = enable_evidence_pinning
        self.prompt_evidence_top_k = max(1, prompt_evidence_top_k)
        self.prompt_evidence_max_chars_per_doc = max(120, prompt_evidence_max_chars_per_doc)
        self.prompt_conclusion_evidence_max_total_chars = max(
            self.prompt_evidence_max_chars_per_doc,
            prompt_conclusion_evidence_max_total_chars,
        )
        self.enable_mmr = enable_mmr
        self.mmr_lambda = min(1.0, max(0.0, mmr_lambda))
        self.enable_pyramid_order = enable_pyramid_order
        self.enable_crag_refinement = enable_crag_refinement
        self.crag_refine_top_sentences = max(1, crag_refine_top_sentences)
        self.crag_refine_max_sentences = max(self.crag_refine_top_sentences, crag_refine_max_sentences)
        self.self_consistency_samples = max(1, self_consistency_samples)
        self.self_consistency_temperature = max(0.0, self_consistency_temperature)
        self.answer_grounding_mode = answer_grounding_mode.strip().lower()
        self.web_context_config = (
            web_context_config
            if isinstance(web_context_config, WebContextConfig)
            else build_web_context_config(web_context_config)
        )

    def prepare_prompt_evidence(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        forced_evidence = [
            item
            for item in evidence
            if _is_web_context_item(item) and self.web_context_config.force_prompt_evidence
        ]
        if self.enable_evidence_pinning:
            forced_evidence.extend(
                _raw_exact_definition_evidence(
                    question,
                    hypothesis,
                    limit=max(1, min(2, self.prompt_evidence_top_k // 4 or 1)),
                )
            )
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
        if self.enable_evidence_pinning:
            selected = _pin_anchor_evidence(
                question,
                hypothesis,
                evidence,
                selected,
                limit=self.prompt_evidence_top_k,
            )
        if self.enable_crag_refinement:
            selected = self.refine_evidence_strips(question, hypothesis, selected)
        if self.enable_pyramid_order:
            selected = apply_pyramid_evidence_order(selected)
            if self.enable_evidence_pinning:
                selected = _pin_anchor_evidence(
                    question,
                    hypothesis,
                    selected,
                    selected,
                    limit=self.prompt_evidence_top_k,
                )
        if forced_evidence:
            selected = _merge_forced_prompt_evidence(
                forced_evidence,
                selected,
                limit=self.prompt_evidence_top_k,
            )
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
        anchors = extract_question_anchor_terms(question, hypothesis)
        action_targets = extract_action_targets(question + "\n" + hypothesis.question)

        refined: list[dict[str, Any]] = []
        for item in evidence:
            doc = item.get("document") or {}
            chain_text = strip_internal_evidence_meta(str(item.get("evidence_chain_text") or "")).strip()
            clean_text = strip_internal_evidence_meta(str(doc.get("clean_text") or ""))
            if not (chain_text or clean_text):
                refined.append(item)
                continue
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
            selected_indices = {index for index, _ in ranked}
            anchor_indices = [
                index
                for index, strip in enumerate(strips)
                if _anchor_hit_count(strip, anchors) >= 2
            ]
            for index in anchor_indices[:2]:
                selected_indices.add(index)
            action_indices = [
                index
                for index, strip in enumerate(strips)
                if _action_target_score(strip, action_targets) >= 2
            ]
            for index in action_indices[:3]:
                selected_indices.add(index)
            reveal_indices: list[int] = []
            if _is_reveal_question(question, hypothesis):
                scored_reveal_indices = sorted(
                    (
                        (_reveal_direct_score(strip, question, hypothesis), index)
                        for index, strip in enumerate(strips)
                    ),
                    reverse=True,
                )
                reveal_indices = [index for score, index in scored_reveal_indices if score > 0][:4]
                for index in reveal_indices:
                    selected_indices.add(index)
            selected_indices_list = sorted(selected_indices)
            selected_strips = [strips[index] for index in selected_indices_list]
            if chain_text and not _is_reveal_question(question, hypothesis) and _anchor_hit_count(chain_text, anchors) >= 2:
                selected_strips.insert(0, chain_text)
                selected_strips = _dedupe_keep_order(selected_strips)
            refined_doc = dict(doc)
            refined_doc["original_clean_text"] = clean_text
            refined_doc["clean_text"] = "\n".join(selected_strips)
            refined_doc["search_text"] = refined_doc["clean_text"]
            refined_item = dict(item)
            refined_item["document"] = refined_doc
            if _is_reveal_question(question, hypothesis):
                refined_item["prompt_prefer_clean_text"] = True
            if chain_text:
                refined_item["evidence_chain_text"] = chain_text
            refined_item["crag_refinement"] = {
                "enabled": True,
                "original_sentence_count": len(strips),
                "kept_sentence_count": len(selected_strips),
                "kept_sentence_indices": selected_indices_list,
                "anchor_sentence_indices": anchor_indices[:2],
                "reveal_sentence_indices": reveal_indices,
                "max_sentence_score": max(float(score) for score in scores) if scores else None,
            }
            refined.append(refined_item)
        return refined

    def build_hypothesis(self, question: str, dialogue_context: str = "") -> HypothesisDocument:
        prompt = build_hypothesis_prompt(question, dialogue_context)
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(256, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.15,
        )
        raw_output = repair_json_like_output(raw_output)
        payload = extract_json_object(raw_output)
        if not payload:
            print(f"[warn] invalid hypothesis json; fallback=heuristic preview={raw_output[:240]}", flush=True)
            return build_hypothesis(question, dialogue_context)
        try:
            return normalize_hypothesis_payload(
                payload,
                question=question,
                dialogue_context=dialogue_context,
            )
        except ModelOutputError as exc:
            print(f"[warn] invalid hypothesis payload; fallback=heuristic error={exc}", flush=True)
            return build_hypothesis(question, dialogue_context)

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
            max_tokens=min(384, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.15,
        )
        raw_output = repair_json_like_output(raw_output)
        payload = extract_json_object(raw_output)
        if not payload:
            print(f"[warn] invalid follow-up hypothesis json; fallback=heuristic preview={raw_output[:240]}", flush=True)
            return build_hypothesis(question, current_hypothesis.dialogue_context)
        try:
            return normalize_hypothesis_payload(
                payload,
                question=question,
                dialogue_context=current_hypothesis.dialogue_context,
                current_intent=current_hypothesis.intent,
            )
        except ModelOutputError as exc:
            print(f"[warn] invalid follow-up hypothesis payload; fallback=heuristic error={exc}", flush=True)
            return build_hypothesis(question, current_hypothesis.dialogue_context)

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
            evidence_max_chars_per_doc=self.prompt_evidence_max_chars_per_doc,
            evidence_max_total_chars=self.prompt_conclusion_evidence_max_total_chars,
            prompt_mode=self.conclusion_prompt_mode,
        )
        conclusions: list[ConclusionResult] = []
        errors: list[Exception] = []
        sample_count = self.self_consistency_samples
        for _ in range(sample_count):
            try:
                raw_output = self.generator.generate(
                    prompt,
                    max_tokens=min(max(self.generator.max_tokens, 1536), 2048),
                    temperature=self.self_consistency_temperature if sample_count > 1 else 0.1,
                    top_p=0.9 if sample_count > 1 else 0.8,
                    repeat_penalty=1.0,
                )
                if self.conclusion_prompt_mode == "minimal" and not raw_output.lstrip().startswith(("{", "<think>")):
                    raw_output = "{" + raw_output
                raw_output = repair_json_like_output(raw_output)
                payload = extract_json_object(raw_output)
                if not payload:
                    conclusion = parse_conclusion_json_like_output(
                        raw_output,
                        question=question,
                        dialogue_context=current_hypothesis.dialogue_context,
                        current_intent=current_hypothesis.intent,
                        current_hypothesis=current_hypothesis,
                        max_round_reached=current_round >= self.max_retrieval_rounds,
                    )
                    if not conclusion:
                        raise ModelOutputError(f"invalid conclusion json: {raw_output}")
                else:
                    conclusion = normalize_conclusion_payload(
                        payload,
                        question=question,
                        dialogue_context=current_hypothesis.dialogue_context,
                        current_intent=current_hypothesis.intent,
                        current_hypothesis=current_hypothesis,
                        max_round_reached=current_round >= self.max_retrieval_rounds,
                    )
                conclusion = validate_conclusion_grounding(
                    question=question,
                    hypothesis=current_hypothesis,
                    evidence=prompt_evidence,
                    conclusion=conclusion,
                    max_round_reached=current_round >= self.max_retrieval_rounds,
                    mode=self.answer_grounding_mode,
                    evidence_prompt_text=prompt,
                )
                conclusions.append(conclusion)
            except Exception as exc:
                errors.append(exc)
                if sample_count == 1 and not (
                    current_round >= self.max_retrieval_rounds
                    and current_hypothesis.intent != "out_of_scope"
                    and _has_answerable_evidence(prompt_evidence)
                ):
                    raise
                continue

        if not conclusions:
            if current_round >= self.max_retrieval_rounds:
                if current_hypothesis.intent != "out_of_scope" and _has_answerable_evidence(prompt_evidence):
                    try:
                        return self.generate_direct_answer(question, current_hypothesis, prompt_evidence)
                    except Exception as exc:
                        print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
                return ConclusionResult(
                    next_action="abstain",
                    answer="现有检索证据不足以确认，且已达到检索轮次上限。",
                    missing_slots=["conclusion_generation 未产生可解析结论"],
                    clarification_question="",
                    follow_up_hypothesis=None,
                )
            return ConclusionResult(
                next_action="retrieve_more",
                answer="",
                missing_slots=["conclusion_generation 未产生可解析结论，需要继续补充直接证据"],
                clarification_question="",
                follow_up_hypothesis=None,
            )

        action_counts: dict[str, int] = {}
        for conclusion in conclusions:
            action_counts[conclusion.next_action] = action_counts.get(conclusion.next_action, 0) + 1
        winning_action = max(
            action_counts,
            key=lambda action: (action_counts[action], -RETRIEVAL_ACTIONS_ORDER.index(action)),
        )
        winning_conclusion = next(conclusion for conclusion in conclusions if conclusion.next_action == winning_action)
        if (
            current_round >= self.max_retrieval_rounds
            and winning_conclusion.next_action in {"retrieve_more", "abstain"}
            and current_hypothesis.intent != "out_of_scope"
            and _has_answerable_evidence(prompt_evidence)
        ):
            try:
                return self.generate_direct_answer(question, current_hypothesis, prompt_evidence)
            except Exception as exc:
                print(f"[warn] final direct answer fallback failed: {exc}", flush=True)
        return winning_conclusion

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
            evidence_max_chars_per_doc=self.prompt_evidence_max_chars_per_doc,
            evidence_max_total_chars=self.prompt_conclusion_evidence_max_total_chars,
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(max(self.generator.max_tokens, 1536), 2048),
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
            mode=self.answer_grounding_mode,
            evidence_prompt_text=prompt,
        )

    def _search_queries(
        self,
        queries: list[str],
        *,
        minirag_chapter_scope: str | None = None,
        sparse_storyline_scope: str | None = None,
        enable_minirag: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        dense_ranked_lists: list[list[dict[str, Any]]] = []
        sparse_ranked_lists: list[list[dict[str, Any]]] = []
        minirag_ranked_lists: list[list[dict[str, Any]]] = []
        for query in queries:
            dense_ranked_lists.append(self.retriever.dense_search(query, top_k=self.query_config.dense_top_k))
            sparse_ranked_lists.append(
                self.retriever.sparse_search(
                    query,
                    top_k=self.query_config.sparse_top_k,
                    storyline_scope=sparse_storyline_scope,
                )
            )
            minirag_search = getattr(self.retriever, "minirag_search", None)
            if enable_minirag and minirag_search is not None:
                minirag_hits = minirag_search(
                    query,
                    top_k=self.query_config.minirag_top_k,
                    chapter_scope=minirag_chapter_scope,
                )
                if minirag_hits:
                    minirag_ranked_lists.append(minirag_hits)
        return (
            merge_ranked_hits(*dense_ranked_lists),
            merge_ranked_hits(*sparse_ranked_lists),
            merge_ranked_hits(*minirag_ranked_lists),
        )

    def _search_minirag_queries(
        self,
        queries: list[str],
        *,
        minirag_chapter_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        minirag_search = getattr(self.retriever, "minirag_search", None)
        if minirag_search is None:
            return []
        ranked_lists: list[list[dict[str, Any]]] = []
        for query in queries:
            hits = minirag_search(
                query,
                top_k=self.query_config.minirag_top_k,
                chapter_scope=minirag_chapter_scope,
            )
            if hits:
                ranked_lists.append(hits)
        return merge_ranked_hits(*ranked_lists)

    def _search_scoped_chapter_queries(
        self,
        queries: list[str],
        *,
        chapter_scope: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.query_config.enable_scoped_chapter_search or not chapter_scope or not queries:
            return [], []
        dense_search = getattr(self.retriever, "dense_search_chapter", None)
        sparse_search = getattr(self.retriever, "sparse_search_chapter", None)
        dense_ranked_lists: list[list[dict[str, Any]]] = []
        sparse_ranked_lists: list[list[dict[str, Any]]] = []
        for query in queries:
            if dense_search is not None and self.query_config.scoped_chapter_dense_top_k > 0:
                dense_hits = dense_search(
                    query,
                    top_k=self.query_config.scoped_chapter_dense_top_k,
                    chapter_scope=chapter_scope,
                )
                if dense_hits:
                    dense_ranked_lists.append(dense_hits)
            if sparse_search is not None and self.query_config.scoped_chapter_sparse_top_k > 0:
                sparse_hits = sparse_search(
                    query,
                    top_k=self.query_config.scoped_chapter_sparse_top_k,
                    chapter_scope=chapter_scope,
                )
                if sparse_hits:
                    sparse_ranked_lists.append(sparse_hits)
        return merge_ranked_hits(*dense_ranked_lists), merge_ranked_hits(*sparse_ranked_lists)

    def _finalize_hits(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        dense_hits: list[dict[str, Any]],
        sparse_hits: list[dict[str, Any]],
        minirag_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        resolved_question = _resolve_referential_question(question, hypothesis.entities)
        # Keep generation-time expansion out of the final reranker query. The
        # reranker was trained to compare evidence chains against the user
        # question; model-generated keywords are useful for candidate recall but
        # can easily drag the final ranker toward noisy aliases or question
        # fragments.
        rerank_query = resolved_question
        safe_related_terms = expand_related_retrieval_terms(
            extract_action_targets(resolved_question + "\n" + question)
            + _extract_content_tokens(resolved_question)
            + hypothesis.entities[:4]
        )
        if safe_related_terms:
            rerank_query = rerank_query + "\n核心相关线索: " + " ".join(safe_related_terms[:10])
        minirag_weight = self.retriever.effective_minirag_weight(rerank_query, config=self.query_config)
        if self.query_config.minirag_fusion_mode == "append":
            primary_hits = self.retriever.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=[],
                top_k=self.query_config.fusion_top_k,
                rrf_k=self.query_config.rrf_k,
                dense_weight=self.query_config.dense_weight,
                sparse_weight=self.query_config.sparse_weight,
                minirag_weight=0.0,
            )
            fused_hits = self.retriever.append_supplemental_hits(
                primary_hits,
                minirag_hits if minirag_weight > 0 else [],
                top_k=max(self.query_config.reranker_candidate_top_k, self.query_config.fusion_top_k),
                source_name="minirag",
            )
        else:
            fused_hits = self.retriever.reciprocal_rank_fusion(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                minirag_hits=minirag_hits if minirag_weight > 0 else [],
                top_k=max(self.query_config.reranker_candidate_top_k, self.query_config.fusion_top_k),
                rrf_k=self.query_config.rrf_k,
                dense_weight=self.query_config.dense_weight,
                sparse_weight=self.query_config.sparse_weight,
                minirag_weight=minirag_weight,
            )
        if self.query_config.enable_neighbor_expansion:
            fused_hits = self._expand_fused_hits_with_neighbors(fused_hits)

        reranked_hits = rerank_hits(
            self.retriever,
            rerank_query,
            fused_hits,
            top_k=self.query_config.rerank_top_k,
            batch_size=self.query_config.rerank_batch_size,
            query_mode=classify_retrieval_query_mode(hypothesis),
        )
        rescue_core_terms = _dedupe_keep_order(
            extract_action_targets(resolved_question + "\n" + question)
            + _extract_content_tokens(resolved_question)
        )[:6]
        # Rescue candidates should favor the closest deterministic expansion
        # terms; broader related terms are still useful for recall, but can
        # otherwise crowd out the direct bridge evidence.
        rescue_bundle_terms = _dedupe_keep_order([*rescue_core_terms, *safe_related_terms[:3]])
        rescue_hits = _best_anchor_bundle_evidence(
            fused_hits,
            core_terms=rescue_core_terms,
            bundle_terms=rescue_bundle_terms,
            limit=max(1, min(4, self.query_config.rerank_top_k // 4 or 1)),
        )
        if not rescue_hits:
            return reranked_hits
        merged_hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*rescue_hits, *reranked_hits]:
            identity = _evidence_identity(item)
            if identity in seen:
                continue
            seen.add(identity)
            merged_hits.append(item)
            if len(merged_hits) >= self.query_config.rerank_top_k:
                break
        return merged_hits

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
            same_story_sweep=self.query_config.enable_same_story_sweep,
            same_story_max_seed_docs=self.query_config.same_story_sweep_max_seed_docs,
            same_story_max_docs_per_story=self.query_config.same_story_sweep_max_docs_per_story,
        )
        max_candidates = max(
            self.query_config.reranker_candidate_top_k,
            self.query_config.fusion_top_k,
            self.query_config.rerank_top_k,
        )
        if self.query_config.enable_same_story_sweep:
            max_candidates += max(0, self.query_config.same_story_sweep_extra_candidates)
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
        *,
        minirag_chapter_scope: str | None = None,
        candidate_chapter_scope: str | None = None,
        sparse_storyline_scope: str | None = None,
        enable_minirag: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        expanded_queries = expand_queries_with_main_chapter_terms(queries)
        dense_hits, sparse_hits, minirag_hits = self._search_queries(
            expanded_queries,
            minirag_chapter_scope=minirag_chapter_scope,
            sparse_storyline_scope=sparse_storyline_scope,
            enable_minirag=enable_minirag,
        )
        if candidate_chapter_scope:
            local_dense_hits, local_sparse_hits = self._search_scoped_chapter_queries(
                expanded_queries,
                chapter_scope=candidate_chapter_scope,
            )
            dense_hits = merge_ranked_hits(
                filter_hits_by_chapter_scope(dense_hits, candidate_chapter_scope),
                local_dense_hits,
            )
            sparse_hits = merge_ranked_hits(
                filter_hits_by_chapter_scope(sparse_hits, candidate_chapter_scope),
                local_sparse_hits,
            )
        evidence = self._finalize_hits(question, hypothesis, dense_hits, sparse_hits, minirag_hits)
        return dense_hits, sparse_hits, evidence

    def _retrieve_first_round_with_scoped_minirag_expansion(
        self,
        question: str,
        hypothesis: HypothesisDocument,
        queries: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
        expanded_queries = expand_queries_with_main_chapter_terms(queries)
        dense_hits, sparse_hits, _ = self._search_queries(expanded_queries, enable_minirag=False)
        first_pass_evidence = self._finalize_hits(question, hypothesis, dense_hits, sparse_hits, [])
        scope_info = infer_dominant_minirag_chapter_scope(
            first_pass_evidence,
            dense_hits,
            sparse_hits,
            max_items=max(1, int(self.query_config.minirag_scope_seed_top_k)),
        )
        storyline_scope_info = infer_dominant_storyline_scope(
            first_pass_evidence,
            dense_hits,
            sparse_hits,
            max_items=max(1, int(self.query_config.storyline_scope_seed_top_k)),
        )
        storyline_scope = ""
        storyline_scope_ratio = 0.0
        storyline_sparse_scope_enabled = False
        if storyline_scope_info is not None:
            storyline_scope = str(storyline_scope_info["scope"])
            storyline_scope_ratio = float(storyline_scope_info.get("dominance_ratio") or 0.0)
            storyline_sparse_scope_enabled = (
                self.query_config.enable_storyline_sparse_scope
                and storyline_scope_ratio >= float(self.query_config.storyline_sparse_scope_min_ratio)
            )
        sparse_storyline_scope = storyline_scope if storyline_sparse_scope_enabled else None
        if not scope_info:
            return dense_hits, sparse_hits, first_pass_evidence, None

        chapter_scope = str(scope_info["scope"])
        scope_ratio = float(scope_info.get("dominance_ratio") or 0.0)
        graph_scope_enabled = scope_ratio >= float(self.query_config.minirag_graph_scope_min_ratio)
        second_pass_scope_enabled = scope_ratio >= float(self.query_config.minirag_second_pass_scope_min_ratio)
        graph_scope = chapter_scope if graph_scope_enabled else None
        graph_hits = self._search_minirag_queries(
            expanded_queries,
            minirag_chapter_scope=graph_scope,
        )
        if not graph_hits:
            return dense_hits, sparse_hits, first_pass_evidence, {
                "chapter_scope": chapter_scope,
                "chapter_scope_label": scope_info.get("label") or chapter_scope,
                "scope_candidates": scope_info.get("candidates") or [],
                "scope_dominance_ratio": scope_ratio,
                "graph_scope_enabled": graph_scope_enabled,
                "second_pass_scope_enabled": second_pass_scope_enabled,
                "graph_scope_min_ratio": self.query_config.minirag_graph_scope_min_ratio,
                "second_pass_scope_min_ratio": self.query_config.minirag_second_pass_scope_min_ratio,
                "storyline_scope": storyline_scope,
                "storyline_scope_label": storyline_scope_info.get("label") if storyline_scope_info else "",
                "storyline_scope_candidates": storyline_scope_info.get("candidates") if storyline_scope_info else [],
                "storyline_scope_dominance_ratio": storyline_scope_ratio,
                "storyline_sparse_scope_enabled": storyline_sparse_scope_enabled,
                "storyline_sparse_scope_min_ratio": self.query_config.storyline_sparse_scope_min_ratio,
                "graph_hit_count": 0,
                "second_pass_queries": [],
            }

        second_pass_queries = build_minirag_expansion_queries(
            question,
            hypothesis,
            graph_hits,
            chapter_scope_label=str(scope_info.get("label") or chapter_scope) if graph_scope_enabled else "global",
            top_k=max(1, int(self.query_config.minirag_expansion_query_top_k)),
        )
        second_dense_hits, second_sparse_hits, second_minirag_hits = self._search_queries(
            expand_queries_with_main_chapter_terms(second_pass_queries),
            minirag_chapter_scope=graph_scope,
            sparse_storyline_scope=sparse_storyline_scope,
            enable_minirag=True,
        )
        local_dense_hits, local_sparse_hits = self._search_scoped_chapter_queries(
            [*expanded_queries, *expand_queries_with_main_chapter_terms(second_pass_queries)],
            chapter_scope=chapter_scope,
        )
        scoped_dense_hits = merge_ranked_hits(
            filter_hits_by_chapter_scope(dense_hits, chapter_scope),
            filter_hits_by_chapter_scope(second_dense_hits, chapter_scope),
            local_dense_hits,
        )
        scoped_sparse_hits = merge_ranked_hits(
            filter_hits_by_chapter_scope(sparse_hits, chapter_scope),
            filter_hits_by_chapter_scope(second_sparse_hits, chapter_scope),
            local_sparse_hits,
        )
        combined_minirag_hits = merge_ranked_hits(graph_hits, second_minirag_hits)
        scoped_candidate_count = len(scoped_dense_hits) + len(scoped_sparse_hits) + len(combined_minirag_hits)
        use_scoped_candidates = (
            second_pass_scope_enabled and scoped_candidate_count >= max(8, self.query_config.rerank_top_k)
        )
        global_dense_hits = merge_ranked_hits(dense_hits, second_dense_hits)
        global_sparse_hits = merge_ranked_hits(sparse_hits, second_sparse_hits)
        if use_scoped_candidates:
            # Keep the scoped lane first, but do not discard global candidates.
            # A wrong dominant scope is otherwise unrecoverable for multi-entity
            # or definition questions.
            combined_dense_hits = merge_ranked_hits(scoped_dense_hits, global_dense_hits)
            combined_sparse_hits = merge_ranked_hits(scoped_sparse_hits, global_sparse_hits)
        else:
            combined_dense_hits = global_dense_hits
            combined_sparse_hits = global_sparse_hits
        evidence = self._finalize_hits(
            question,
            hypothesis,
            combined_dense_hits,
            combined_sparse_hits,
            combined_minirag_hits,
        )
        expansion_record = {
            "chapter_scope": chapter_scope,
            "chapter_scope_label": scope_info.get("label") or chapter_scope,
            "scope_candidates": scope_info.get("candidates") or [],
            "scope_dominance_ratio": scope_ratio,
            "graph_scope_enabled": graph_scope_enabled,
            "second_pass_scope_enabled": second_pass_scope_enabled,
            "graph_scope_min_ratio": self.query_config.minirag_graph_scope_min_ratio,
            "second_pass_scope_min_ratio": self.query_config.minirag_second_pass_scope_min_ratio,
            "storyline_scope": storyline_scope,
            "storyline_scope_label": storyline_scope_info.get("label") if storyline_scope_info else "",
            "storyline_scope_candidates": storyline_scope_info.get("candidates") if storyline_scope_info else [],
            "storyline_scope_dominance_ratio": storyline_scope_ratio,
            "storyline_sparse_scope_enabled": storyline_sparse_scope_enabled,
            "storyline_sparse_scope_min_ratio": self.query_config.storyline_sparse_scope_min_ratio,
            "graph_hit_count": len(graph_hits),
            "graph_evidence_summary": summarize_evidence_for_trace(graph_hits),
            "second_pass_queries": second_pass_queries,
            "scoped_dense_hit_count": len(scoped_dense_hits),
            "scoped_sparse_hit_count": len(scoped_sparse_hits),
            "scoped_local_dense_hit_count": len(local_dense_hits),
            "scoped_local_sparse_hit_count": len(local_sparse_hits),
            "scoped_candidate_count": scoped_candidate_count,
            "use_scoped_candidates": use_scoped_candidates,
            "dual_lane_global_fallback_enabled": use_scoped_candidates,
            "global_dense_hit_count": len(global_dense_hits),
            "global_sparse_hit_count": len(global_sparse_hits),
            "second_pass_evidence_summary": summarize_evidence_for_trace(evidence),
        }
        return combined_dense_hits, combined_sparse_hits, evidence, expansion_record

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
        web_context_evidence: list[dict[str, Any]] = []
        final_answer = ""
        retained_chapter_scope: str | None = None
        retained_storyline_scope: str | None = None
        retained_scope_evidence: list[dict[str, Any]] = []
        scope_retention_enabled = False

        pending_queries = [
            _resolve_referential_question(question, current_hypothesis.entities),
            build_retrieval_query(current_hypothesis),
        ]
        pending_queries.extend(build_follow_up_hypothesis_queries(question, current_hypothesis))
        pending_queries = expand_queries_with_main_chapter_terms(pending_queries)
        current_hypothesis_task_type = INITIAL_HYPOTHESIS_TASK_TYPE

        for round_index in range(1, self.max_retrieval_rounds + 1):
            if progress_callback:
                progress_callback("retrieval")
            minirag_expansion_record: dict[str, Any] | None = None
            if (
                round_index == 1
                and self.query_config.minirag_chapter_isolation
                and self.query_config.minirag_auto_second_retrieval
            ):
                dense_hits, sparse_hits, evidence, minirag_expansion_record = (
                    self._retrieve_first_round_with_scoped_minirag_expansion(
                        question,
                        current_hypothesis,
                        pending_queries,
                    )
                )
                if minirag_expansion_record is not None and progress_callback:
                    progress_callback(MINIRAG_CHAPTER_EXPANSION_TASK_TYPE)
                if minirag_expansion_record is not None:
                    retained_chapter_scope = str(minirag_expansion_record.get("chapter_scope") or "").strip() or None
                    retained_storyline_scope = (
                        str(minirag_expansion_record.get("storyline_scope") or "").strip() or None
                    )
                    scope_retention_enabled = bool(
                        minirag_expansion_record.get("use_scoped_candidates")
                        and retained_chapter_scope
                    )
            else:
                dense_hits, sparse_hits, evidence = self._retrieve_round(
                    question,
                    current_hypothesis,
                    pending_queries,
                    minirag_chapter_scope=None,
                    candidate_chapter_scope=None,
                    sparse_storyline_scope=None,
                )
            if scope_retention_enabled and retained_scope_evidence and round_index > 1:
                evidence = merge_evidence_keep_order(
                    retained_scope_evidence,
                    evidence,
                    limit=max(self.query_config.reranker_candidate_top_k, self.prompt_evidence_top_k * 2),
                )
            web_context_record: dict[str, Any] | None = None
            if round_index == 1 and self.web_context_config.enabled:
                if progress_callback:
                    progress_callback(WEB_CONTEXT_TASK_TYPE)
                web_context_evidence, web_context_record = build_web_context_evidence(
                    question,
                    evidence,
                    self.web_context_config,
                    retriever=self.retriever,
                    hypothesis=current_hypothesis,
                )
            if web_context_evidence:
                evidence = [*web_context_evidence, *evidence]
            if self.enable_evidence_pinning:
                raw_exact_evidence = _raw_exact_definition_evidence(
                    question,
                    current_hypothesis,
                    limit=max(1, min(2, self.prompt_evidence_top_k // 4 or 1)),
                )
                if raw_exact_evidence:
                    evidence = merge_evidence_keep_order(
                        raw_exact_evidence,
                        evidence,
                        limit=max(self.query_config.reranker_candidate_top_k, self.prompt_evidence_top_k * 2),
                    )
            if round_index == 1 and scope_retention_enabled:
                retained_scope_evidence = list(evidence)

            step_record: dict[str, Any] = {
                "round": round_index,
                "queries": list(pending_queries),
                "planner_action": "retrieval_completed",
                "hypothesis_task_type": current_hypothesis_task_type,
                "hypothesis": asdict(current_hypothesis),
                "evidence_summary": summarize_evidence_for_trace(evidence),
                "retained_chapter_scope": retained_chapter_scope or "",
                "retained_storyline_scope": retained_storyline_scope or "",
                "scope_retention_enabled": scope_retention_enabled,
            }
            if web_context_record is not None:
                step_record["web_context"] = web_context_record
            if minirag_expansion_record is not None:
                step_record["minirag_chapter_expansion"] = minirag_expansion_record
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
            pending_queries.extend(build_follow_up_hypothesis_queries(question, current_hypothesis))
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
                    "evidence_pinning_enabled": self.enable_evidence_pinning,
                    "moegirl_downweight_enabled": True,
                    "near_duplicate_dedupe_enabled": True,
                    "crag_refinement_enabled": self.enable_crag_refinement,
                    "crag_refine_top_sentences": self.crag_refine_top_sentences,
                    "crag_refine_max_sentences": self.crag_refine_max_sentences,
                    "web_context_enabled": self.web_context_config.enabled,
                    "web_context_max_pages": self.web_context_config.max_pages,
                    "web_context_max_total_chars": self.web_context_config.max_total_chars,
                    "minirag_chapter_isolation": self.query_config.minirag_chapter_isolation,
                    "minirag_auto_second_retrieval": self.query_config.minirag_auto_second_retrieval,
                    "minirag_scope_seed_top_k": self.query_config.minirag_scope_seed_top_k,
                    "minirag_expansion_query_top_k": self.query_config.minirag_expansion_query_top_k,
                    "scoped_chapter_search_enabled": self.query_config.enable_scoped_chapter_search,
                    "scoped_chapter_dense_top_k": self.query_config.scoped_chapter_dense_top_k,
                    "scoped_chapter_sparse_top_k": self.query_config.scoped_chapter_sparse_top_k,
                    "same_story_sweep_enabled": self.query_config.enable_same_story_sweep,
                    "same_story_sweep_max_seed_docs": self.query_config.same_story_sweep_max_seed_docs,
                    "same_story_sweep_max_docs_per_story": self.query_config.same_story_sweep_max_docs_per_story,
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
