#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_fix1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/llama_factory/teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3"
)
DEFAULT_DOCS_PATH = PROJECT_ROOT / "indexes/arknights_story/documents.jsonl"

INITIAL_TASK = "user_question_hypothesis_generation"
FOLLOW_UP_TASK = "follow_up_hypothesis_generation"
CONCLUSION_TASK = "conclusion_generation"

INTENTS = {
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
RETRIEVAL_ACTIONS = {"answer_directly", "retrieve_more", "clarify_user", "abstain"}

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}

INTERNAL_EVIDENCE_META_RE = re.compile(
    r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]+\]\s*|\[E\d+\]\s*"
)
DOC_ID_RE = re.compile(r"([A-Za-z0-9_\[\]./\-]+#chunk-\d+)")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
ROUND_RE = re.compile(r"round:\s*(\d+)\s*/\s*(\d+)")
QUOTED_RE = re.compile(r"[“\"'‘]([^”\"'’]{1,24})[”\"'’]")
CJK_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_.-]+")
CHINESE_SPLIT_RE = re.compile(r"[的是和与及或为在把被让给从向对将了后前中上下因而认并但又却才再也都只]")
QUESTION_SPLIT_RE = re.compile(
    r"(为什么|因为什么|为何|怎么|如何|是什么|是谁|指谁|有何|哪些|哪个|多少|是否|"
    r"吗|呢|认为|觉得|说明|表达|体现|反映|揭示|发生|可以|能够|能否|能不能|"
    r"帮忙|帮助|用来|可能|真正|具体|直接|相关|没有|区别|不同|原因|目的|动机|"
    r"态度|关系|建议|要求|选择|拒绝|决定|最终|之前|之后|时候|什么样|什么|怎样|"
    r"解读|急着|并不|等人|口头声称|提出|需要|为何|命令|仿造|追查|提到|实际上|"
    r"根据|关于|之间|导致|造成|避免|获得|来到|成为|使用|创造|成功|继续|必须|"
    r"声称|针对|担心|回应|描述|经历|关键|主要|实际)"
)
PHRASE_SPLIT_RE = re.compile(
    r"[，,。！？?；;：:、\s]+|"
    r"[的了与和及或为在把被让给从向对将中上下前后而并但又却才再也只其她他出]|"
    r"(?:为什么|因为什么|为何|怎么|如何|是什么|是谁|指谁|有何|哪些|哪个|多少|是否|"
    r"认为|觉得|说明|表达|体现|反映|揭示|可以|能够|能否|能不能|可能|真正|具体|直接|"
    r"相关|没有|区别|不同|原因|目的|动机|态度|关系|建议|要求|选择|拒绝|决定|最终|"
    r"之前|之后|时候|什么样|什么|怎样|解读|提出|需要|命令|仿造|追查|提到|实际上|"
    r"根据|关于|之间|导致|造成|避免|获得|来到|成为|使用|创造|成功|继续|必须|声称|"
    r"针对|担心|回应|描述|经历|关键|主要|实际|通过|推断|透露|计划|扮演|承认|策划|"
    r"教唆|发现|记录|究竟|指控|设计|沾染|邀请|进入|通知|靠近|指着|出气|深层|共同点|同意|说服|"
    r"最初|后来|除了|还有|各自|那些|这个|那个|这背后|并非|不是|而是|并|且|或)"
)
FUNCTION_CHAR_RE = re.compile(r"[的是和与及或为在把被让给从向对将了后前中上下因而认并但又却才再也都只]")
ENTITY_BAD_SUBSTRINGS = (
    "知道",
    "认为",
    "觉得",
    "担心",
    "需要",
    "决定",
    "选择",
    "建议",
    "要求",
    "请求",
    "提醒",
    "解释",
    "判断",
    "推断",
    "体现",
    "反映",
    "揭示",
    "发现",
    "确认",
    "质疑",
    "拒绝",
    "帮助",
    "看到",
    "出席",
    "辞去",
    "受到",
    "打伤",
    "患有",
    "追着",
    "前往",
    "回来",
    "离开",
    "加入",
    "使用",
    "牺牲",
    "射杀",
    "寻找",
    "改变",
    "发生",
    "出现",
    "处理",
    "应对",
    "评价",
    "称呼",
    "计划",
    "扮演",
    "邀请",
    "询问",
    "下手",
)

KALTSIT_INTERNAL_ALIAS = "凯尔希·思衡托"
KALTSIT_NATURAL_NAME = "凯尔希"

GENERIC_TERMS = {
    "为什么",
    "为何",
    "怎么",
    "如何",
    "什么",
    "哪里",
    "哪儿",
    "是否",
    "有没有",
    "用户",
    "问题",
    "剧情",
    "证据",
    "当前",
    "回答",
    "关系",
    "身份",
    "原因",
    "具体",
    "这个",
    "那个",
    "这种",
    "这件事",
    "是因",
    "这是",
    "用来",
    "可能",
    "真的",
    "哪些",
    "多少",
    "以及",
    "时候",
    "故事",
    "任务",
    "信息",
    "情况",
    "内容",
    "人物",
    "角色",
    "事件",
    "影响",
    "目的",
    "动机",
    "表现",
    "体现",
    "相关",
    "线索",
    "直接",
    "原文",
    "片段",
    "上述",
    "检索",
    "chunk",
    "明日方舟",
    "发生",
    "表达",
    "说明",
    "自己",
    "还有",
    "后来",
    "过程",
    "遇到",
    "出气",
    "共同点",
    "真相",
    "事实",
    "隐藏",
    "现时",
    "城时",
    "方式有",
    "布兰",
    "事情",
    "受到",
    "知道",
    "看到",
    "促使",
    "主意",
    "一个",
    "自己以",
    "哥哥很",
    "患有严重",
    "乔万娜手",
    "塞先生",
    "松果看",
    "得奖",
    "照片",
    "说不会做这种事",
    "现会影响杜林人",
    "往地面",
}
BAD_TERM_SUBSTRINGS = (
    "为什么",
    "为何",
    "怎么",
    "如何",
    "什么",
    "是否",
    "有没有",
    "用户问题",
    "当前证据",
    "检索",
    "证据",
    "chunk",
    "MiniRAG",
    "minirag",
    "多种她",
    "多种他",
    "解释这",
    "解释该",
    "如何解释这",
    "认为森西等人",
    "可以帮忙解",
    "不急着赶路",
    "提到纠",
    "头子实际",
    "命令海",
    "茨仿造",
    "般素",
    "之间关",
    "于煌",
    "息共享情况",
    "样时",
    "动要塞",
    "内出现",
    "话揭示",
    "市建设",
    "之间存",
    "那个瘦弱",
    "孩子之间",
    "告别反映",
    "帝国理",
    "工大学",
    "凶手说",
    "你当初真",
    "应该跟",
    "着你",
    "例子反驳",
    "于家族",
    "乌萨斯身",
    "维特却",
    "战斗最",
    "刻会犹豫",
    "研究所来到罗德",
    "策略来创造攻击机",
    "场景能获得稀有素",
    "商业联合会选",
    "绵生物",
    "主要针",
    "切斯柏声称要",
    "莉娜复仇",
    "根据托兰",
    "以至于她必须继续",
    "哪一刻起",
    "剧组另有",
    "要走一条",
    "传闻背",
    "深池收买时说",
    "来表明",
    "说阿斯卡纶",
    "老威尼斯明明",
    "卢奇诺面",
    "天师学徒要那种漂亮水稻",
    "现会影响杜林人",
    "往地面",
    "话还",
    "请您少说",
    "卡恩提",
    "来未",
    "席葬礼",
    "诺伯特区正",
    "博士所说",
    "指特蕾西娅",
    "渡桥用",
    "称呼妮芙",
    "深池收买",
    "特蕾西娅希望博士",
    "泰拉解决",
    "替博士来索取塔拉利益",
    "失手打伤猎蜂",
    "会用滋呜呜追着哥哥姐姐跑",
    "请求号角射杀",
    "请去检查死者遗体",
    "天师学徒要",
    "瑟奇亚克判断通讯",
    "麟青砚能接触到煌",
    "禾生锄地",
    "人物有关",
    "卡兹戴尔感染者时",
    "说乌提卡伯爵塔",
    "萨卢佐家族除名",
    "你到底还要卷入",
    "改名背",
    "英格丽原本考虑留安东尼奥一命",
    "灰喉呼叫煌支援时",
    "天灾信使伊恩",
    "商人躺",
    "放缓血液流速",
    "相信队友",
    "质疑推进之王",
    "罗宾不适合养云兽",
    "席瑙曼夫妇",
    "赦罪师各有",
)
BAD_EXACT_TERMS = {
    "察队",
    "般素",
    "样时",
    "方式",
    "看法",
    "有关",
    "关键",
    "主要",
    "两人",
    "描述",
    "透露",
    "使用",
    "策略",
    "获得",
    "能力",
    "继续",
    "寻找",
    "关联",
    "理由",
    "目标",
    "行动",
    "会来",
}
FALLBACK_ANSWER_MARKERS = (
    "已检索到的证据能确认",
    "但原答案中的",
    "grounding 校验",
    "原答案中的",
    "没有在当前证据中得到直接支撑",
    "没有在当前证据中得到直接支持",
)
NO_EVIDENCE_MARKERS = (
    "没有检索到",
    "未检索到",
    "当前证据未",
    "当前证据中未",
    "现有证据未",
    "无法回答",
    "不能回答",
    "不足以确认",
)
WEAK_ANSWER_MARKERS = (
    "无法确定",
    "不能确定",
    "缺少关键",
    "缺失关键",
    "无法直接回答",
)
CAUSAL_WORDS = ("为什么", "为何", "原因", "动机", "目的", "怎么会", "为何会")
RELATION_WORDS = ("关系", "联系", "关联")
COMPARE_WORDS = ("区别", "不同", "相比", "比较", "分歧")
TIMELINE_WORDS = ("时间线", "先后", "何时", "什么时候", "顺序")
REVEAL_WORDS = ("真相", "秘密", "来历", "身世", "是谁", "指谁", "身份")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def task_type(record: dict[str, Any]) -> str:
    value = str(record.get("task_type") or "").strip()
    if value:
        return value
    prompt = user_text(record)
    match = re.search(r"^task:\s*([^\n]+)", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def user_text(record: dict[str, Any]) -> str:
    return "\n".join(
        str(message.get("value") or message.get("content") or "")
        for message in record.get("conversations") or []
        if message.get("from") in {"human", "user"}
    )


def assistant_message(record: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(record.get("conversations") or []):
        if message.get("from") in {"gpt", "assistant"}:
            return message
    return None


def assistant_text(record: dict[str, Any]) -> str:
    message = assistant_message(record)
    return str(message.get("value") or message.get("content") or "") if message else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def dump_compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_internal_meta(text: str) -> str:
    cleaned = INTERNAL_EVIDENCE_META_RE.sub("", str(text or ""))
    cleaned = cleaned.replace(KALTSIT_INTERNAL_ALIAS, KALTSIT_NATURAL_NAME)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_prompt_text(text: str) -> str:
    lines: list[str] = []
    for line in clean_internal_meta(text).splitlines():
        if line.startswith("minirag_hints:"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def extract_question_from_prompt(prompt: str) -> str:
    match = re.search(r"^question:\s*([^\n]+)", prompt, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    match = re.search(r"用户原问题[:：]\s*([^\n]+)", prompt)
    return match.group(1).strip() if match else ""


def extract_round(prompt: str) -> tuple[int | None, int | None]:
    match = ROUND_RE.search(prompt)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def extract_evidence(prompt: str) -> str:
    for label in ("当前证据:", "evidence_brief:"):
        if label not in prompt:
            continue
        tail = prompt.split(label, 1)[1]
        stops = [
            index
            for marker in (
                "\n输出要求:",
                "\noutput_schema:",
                "\nprevious_action:",
                "\nmissing_slots:",
                "\nfields:",
            )
            if (index := tail.find(marker)) >= 0
        ]
        if stops:
            tail = tail[: min(stops)]
        return clean_internal_meta(tail)
    return ""


def extract_hypothesis_from_prompt(prompt: str) -> dict[str, Any] | None:
    if "hypothesis:" not in prompt:
        return None
    tail = prompt.split("hypothesis:", 1)[1]
    for marker in ("\nround:", "\nevidence_brief:", "\nmissing_slots:", "\nprevious_action:"):
        index = tail.find(marker)
        if index >= 0:
            tail = tail[:index]
    match = JSON_OBJECT_RE.search(tail)
    return parse_json_object(match.group(0)) if match else None


def load_documents(path: Path) -> dict[str, dict[str, str]]:
    docs: dict[str, dict[str, str]] = {}
    if not path.exists():
        return docs
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            docs[str(payload["id"])] = {
                "text": clean_internal_meta(payload.get("clean_text") or payload.get("search_text") or ""),
                "activity_name": str(payload.get("activity_name") or ""),
                "story_name": str(payload.get("story_name") or ""),
                "stage_code": str(payload.get("stage_code") or ""),
            }
    return docs


def evidence_context(prompt: str, docs: dict[str, dict[str, str]]) -> tuple[str, str, list[str]]:
    evidence_prompt = extract_evidence(prompt)
    doc_ids = list(dict.fromkeys(DOC_ID_RE.findall(evidence_prompt)))
    full_parts = [docs[doc_id]["text"] for doc_id in doc_ids if doc_id in docs and docs[doc_id]["text"]]
    if not full_parts:
        full_parts = [
            re.sub(r"^\s*\d+\.\s*[^:：]{1,160}[:：]\s*", "", line).strip()
            for line in evidence_prompt.splitlines()
            if line.strip()
        ]
    return evidence_prompt, "\n".join(part for part in full_parts if part).strip(), doc_ids


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", clean_internal_meta(text))
    chunks = re.split(r"(?<=[。！？；.!?])\s*|(?<=\s)\d+\.\s*", compact)
    output: list[str] = []
    for chunk in chunks:
        item = chunk.strip(" \n\t。；")
        if not item:
            continue
        if len(item) > 260:
            subparts = re.split(r"(?<=[，,：:])\s*", item)
            current = ""
            for part in subparts:
                if len(current) + len(part) > 220 and current:
                    output.append(current.strip("，,：: "))
                    current = part
                else:
                    current += part
            if current:
                output.append(current.strip("，,：: "))
        else:
            output.append(item)
    return [item for item in output if 8 <= len(item) <= 260]


def normalize_term(term: str) -> str:
    value = str(term or "").strip()
    value = value.replace(KALTSIT_INTERNAL_ALIAS, KALTSIT_NATURAL_NAME)
    value = re.sub(r"^[\"'“”‘’\s]+|[\"'“”‘’\s]+$", "", value)
    return value


def is_bad_term(term: str, *, field: str = "keywords") -> bool:
    value = normalize_term(term)
    compact = re.sub(r"\s+", "", value)
    if not compact or len(compact) < 2:
        return True
    if compact in BAD_EXACT_TERMS:
        return True
    if compact in GENERIC_TERMS:
        return True
    if compact[0] in "的了与和及或为在把被让给从向对将中上下前后而何并但又却才再也只其她他出":
        return True
    if compact[-1] in "的了与和及或为在把被让给从向对将中上下前后而何并但又却才再也只其她他出":
        return True
    if len(compact) == 2 and compact.endswith(("之", "的", "了", "为", "在", "被", "把", "与", "和", "中")):
        return True
    if len(compact) <= 3 and compact[0] in "的了与和在被把为是":
        return True
    if compact.endswith(("因", "认", "并", "何", "而")) and len(compact) <= 8:
        return True
    if compact.endswith(("这", "该")) and len(compact) <= 8:
        return True
    if compact.startswith(("而", "何", "因", "为")) and len(compact) <= 6:
        return True
    if re.search(r"(感到|认为|说明|表达|体现|发生|为何|为什么|怎么|如何)$", compact) and len(compact) <= 6:
        return True
    if any(marker in compact for marker in BAD_TERM_SUBSTRINGS):
        return True
    if field == "entities" and any(marker in compact for marker in ENTITY_BAD_SUBSTRINGS):
        return True
    if len(compact) >= 7 and FUNCTION_CHAR_RE.search(compact):
        return True
    if field == "entities" and (len(compact) > 12 or any(x in compact for x in ("原因", "目的", "动机", "关系"))):
        return True
    if field == "keywords" and len(compact) > 16:
        return True
    return False


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = normalize_term(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def split_question_terms(question: str) -> list[str]:
    text = normalize_term(question)
    quoted = [normalize_term(item) for item in QUOTED_RE.findall(text)]
    normalized_text = PHRASE_SPLIT_RE.sub(" ", QUESTION_SPLIT_RE.sub(" ", text))
    parts: list[str] = []
    for raw in CJK_RE.findall(normalized_text):
        if raw.isascii():
            parts.append(normalize_term(raw))
            continue
        raw = normalize_term(raw)
        if not is_bad_term(raw, field="keywords"):
            parts.append(raw)
        parts.extend(normalize_term(item) for item in CHINESE_SPLIT_RE.split(raw) if item)
    return [
        item
        for item in dedupe_keep_order(quoted + parts)
        if not is_bad_term(item, field="keywords")
    ][:16]


def normalize_list(value: Any, *, field: str, extra_terms: list[str] | None = None, limit: int = 16) -> list[str]:
    raw: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (str, int, float)):
                raw.extend(re.split(r"[\s,，、/|]+", str(item)))
    elif isinstance(value, (str, int, float)):
        raw.extend(re.split(r"[\s,，、/|]+", str(value)))
    if extra_terms:
        raw.extend(extra_terms)
    cleaned = [normalize_term(item) for item in raw]
    return [item for item in dedupe_keep_order(cleaned) if not is_bad_term(item, field=field)][:limit]


def infer_intent_query_answer(question: str) -> tuple[str, str, str]:
    q = question or ""
    if any(word in q for word in COMPARE_WORDS):
        return "compare", "reasoning", "对比分析"
    if any(word in q for word in RELATION_WORDS):
        return "character_relation", "relation", "关系说明"
    if any(word in q for word in TIMELINE_WORDS):
        return "timeline", "fact", "时间线说明"
    if any(word in q for word in CAUSAL_WORDS):
        return "plot_reasoning", "causality", "原因解释"
    if any(word in q for word in REVEAL_WORDS):
        return "plot_fact", "reveal", "身份/真相说明"
    return "plot_fact", "fact", "剧情事实"


def repair_hypothesis_payload(
    payload: dict[str, Any],
    *,
    prompt: str,
    follow_up: bool,
) -> tuple[dict[str, Any], list[str]]:
    repaired: dict[str, Any] = {}
    reasons: list[str] = []
    question = str(payload.get("question") or extract_question_from_prompt(prompt)).strip()
    inferred_intent, inferred_query_type, inferred_answer_type = infer_intent_query_answer(question)
    q_terms = split_question_terms(question)

    if follow_up:
        allowed = ("question", "query_type", "entities", "keywords", "expected_answer_type", "dialogue_context")
    else:
        allowed = ("question", "intent", "query_type", "entities", "keywords", "expected_answer_type", "dialogue_context")

    repaired["question"] = question
    if not follow_up:
        intent = str(payload.get("intent") or "").strip()
        if intent not in INTENTS:
            intent = inferred_intent
            reasons.append("fix_intent")
        repaired["intent"] = intent

    query_type = str(payload.get("query_type") or "").strip()
    if query_type not in QUERY_TYPES:
        query_type = inferred_query_type
        reasons.append("fix_query_type")
    if any(word in question for word in CAUSAL_WORDS) and query_type in {"fact", "answerability"}:
        query_type = "causality"
        reasons.append("align_query_type_causality")
    if any(word in question for word in COMPARE_WORDS) and query_type == "fact":
        query_type = "reasoning"
        reasons.append("align_query_type_compare")
    if any(word in question for word in RELATION_WORDS) and query_type == "fact":
        query_type = "relation"
        reasons.append("align_query_type_relation")
    repaired["query_type"] = query_type

    entities = normalize_list(payload.get("entities"), field="entities", extra_terms=q_terms[:6], limit=10)
    keywords = normalize_list(payload.get("keywords"), field="keywords", extra_terms=q_terms, limit=18)
    if not entities:
        entities = q_terms[:4] or ["剧情"]
        reasons.append("fill_entities")
    if not keywords:
        keywords = q_terms[:8] or entities
        reasons.append("fill_keywords")
    if entities != payload.get("entities"):
        reasons.append("clean_entities")
    if keywords != payload.get("keywords"):
        reasons.append("clean_keywords")
    repaired["entities"] = entities
    repaired["keywords"] = keywords

    answer_type = str(payload.get("expected_answer_type") or "").strip()
    if not answer_type or answer_type in {"答案", "回答", "信息"}:
        answer_type = inferred_answer_type
        reasons.append("fix_expected_answer_type")
    repaired["expected_answer_type"] = answer_type
    repaired["dialogue_context"] = str(payload.get("dialogue_context") or "").strip()

    return {key: repaired[key] for key in allowed}, sorted(set(reasons))


def hypothesis_terms(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    terms: list[str] = []
    for field in ("entities", "keywords"):
        value = payload.get(field)
        if isinstance(value, list):
            terms.extend(str(item) for item in value)
    return normalize_list(terms, field="keywords", limit=24)


def compact_contains(text: str, term: str) -> bool:
    return re.sub(r"\s+", "", normalize_term(term)) in re.sub(r"\s+", "", text or "")


def support_profile(
    *,
    question: str,
    answer: str,
    evidence_text: str,
    hypothesis_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    q_terms = dedupe_keep_order(split_question_terms(question) + hypothesis_terms(hypothesis_payload))
    q_hits = [term for term in q_terms[:18] if compact_contains(evidence_text, term)]
    answer_terms = normalize_list(split_question_terms(answer), field="keywords", limit=40)
    answer_hits = [term for term in answer_terms if compact_contains(evidence_text, term) or compact_contains(question, term)]
    quoted_terms = [normalize_term(item) for item in QUOTED_RE.findall(question)]
    quoted_hits = [term for term in quoted_terms if term and compact_contains(evidence_text, term)]
    answer_support = len(answer_hits) / max(1, len(answer_terms))
    return {
        "q_terms": q_terms,
        "q_hits": q_hits,
        "quoted_hits": quoted_hits,
        "answer_terms": answer_terms,
        "answer_hits": answer_hits,
        "answer_support": answer_support,
    }


def sentence_score(sentence: str, terms: list[str], question: str) -> int:
    score = 0
    for term in terms[:24]:
        if compact_contains(sentence, term):
            score += 3 if term in QUOTED_RE.findall(question) else 1
    if any(word in sentence for word in ("因为", "所以", "为了", "因此", "但", "不过", "而是", "不是", "指", "就是", "是")):
        score += 1
    if any(noise in sentence for noise in ("animStyle", "focusStyle", "dialogue", "Highlight", "black=")):
        score -= 3
    return score


def select_relevant_sentences(question: str, evidence_text: str, terms: list[str], *, limit: int = 3) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(split_sentences(evidence_text)):
        score = sentence_score(sentence, terms, question)
        if score >= 2:
            scored.append((score, -index, sentence))
    scored.sort(reverse=True)
    selected: list[str] = []
    for _, _, sentence in scored:
        if any(sentence in old or old in sentence for old in selected):
            continue
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def answer_signal_ok(question: str, selected: list[str], profile: dict[str, Any]) -> bool:
    if not selected:
        return False
    text = " ".join(selected)
    if len(text) < 24:
        return False
    if all(("？" in sentence or "?" in sentence) and len(sentence) < 80 for sentence in selected):
        return False
    q_hit_count = len(profile.get("q_hits") or [])
    quoted_hit_count = len(profile.get("quoted_hits") or [])
    if any(word in question for word in CAUSAL_WORDS):
        cause_markers = (
            "因为",
            "由于",
            "所以",
            "因此",
            "为了",
            "以便",
            "以免",
            "否则",
            "不然",
            "担心",
            "害怕",
            "希望",
            "想要",
            "需要",
            "必须",
            "选择",
            "决定",
            "打算",
            "目的",
            "原因",
            "动机",
        )
        return q_hit_count >= 3 and any(marker in text for marker in cause_markers)
    if any(word in question for word in RELATION_WORDS):
        relation_markers = ("朋友", "家人", "父亲", "母亲", "老师", "学生", "同伴", "关系", "同事", "敌人")
        return q_hit_count >= 3 and any(marker in text for marker in relation_markers)
    if any(word in question for word in COMPARE_WORDS):
        compare_markers = ("不同", "区别", "相比", "而", "但", "不是", "分歧")
        return q_hit_count >= 3 and any(marker in text for marker in compare_markers)
    if any(word in question for word in REVEAL_WORDS):
        reveal_markers = ("是", "就是", "名为", "叫", "身份", "原名", "本名", "指", "来自", "作为")
        return (q_hit_count >= 2 or quoted_hit_count > 0) and any(marker in text for marker in reveal_markers)
    return q_hit_count >= 3 or (quoted_hit_count > 0 and q_hit_count >= 1)


def build_grounded_answer(question: str, evidence_text: str, terms: list[str], profile: dict[str, Any]) -> str:
    selected = select_relevant_sentences(question, evidence_text, terms, limit=2)
    if not answer_signal_ok(question, selected, profile):
        return ""
    if not selected:
        return ""
    body = "；".join(sentence.rstrip("。；") for sentence in selected)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 520:
        body = body[:520].rstrip("，,；; ") + "。"
    elif body and body[-1] not in "。！？":
        body += "。"
    if any(word in question for word in CAUSAL_WORDS):
        return "根据当前证据，可以确定的原因是：" + body
    if any(word in question for word in COMPARE_WORDS):
        return "根据当前证据，可以确定的分歧或区别是：" + body
    if any(word in question for word in REVEAL_WORDS):
        return "根据当前证据，可以确定：" + body
    return "根据当前证据，可以确定：" + body


def concise_abstain_answer(question: str, evidence_text: str, terms: list[str]) -> str:
    selected = select_relevant_sentences(question, evidence_text, terms, limit=2)
    if selected:
        body = "；".join(sentence.rstrip("。；") for sentence in selected)
        if len(body) > 360:
            body = body[:360].rstrip("，,；; ") + "。"
        return "现有证据只能确认部分片段：" + body + "；但还不足以完整回答用户问题。"
    return "现有检索证据不足以确认该问题的关键事实。"


def is_bad_missing_slot(slot: str) -> bool:
    value = normalize_term(slot)
    compact = re.sub(r"\s+", "", value)
    if not compact or len(compact) < 6 or len(compact) > 42:
        return True
    if "原答案" in compact:
        return True
    if "补充与" in compact and "直接相关" in compact:
        return True
    return any(marker in compact for marker in BAD_TERM_SUBSTRINGS)


def generic_missing_slots(question: str, query_type: str) -> list[str]:
    q = question or ""
    if query_type == "causality" or any(word in q for word in CAUSAL_WORDS):
        return [
            "补充直接说明原因或动机的原文证据",
            "补充关键人物行动前后的对话证据",
            "补充事件前因后果的剧情证据",
        ]
    if query_type == "relation" or any(word in q for word in RELATION_WORDS):
        return [
            "补充人物或组织关系的直接表述",
            "补充双方互动场景的原文证据",
            "补充关系变化前后的剧情证据",
        ]
    if any(word in q for word in COMPARE_WORDS):
        return [
            "补充两个对象各自表现的原文证据",
            "补充直接体现差异的对话或旁白",
            "补充对比结论所需的共同场景证据",
        ]
    if query_type == "reveal" or any(word in q for word in REVEAL_WORDS):
        return [
            "补充身份或真相揭示的原文证据",
            "补充人物称呼与身份绑定证据",
            "补充揭示前后的关键对话",
        ]
    return [
        "补充直接回答问题的原文证据",
        "补充关键实体所在场景的剧情证据",
        "补充答案所需的上下文片段",
    ]


def make_missing_slots(question: str, query_type: str, existing_slots: Any = None) -> list[str]:
    slots: list[str] = []
    if isinstance(existing_slots, list):
        for slot in existing_slots:
            text = normalize_term(str(slot))
            if not is_bad_missing_slot(text):
                slots.append(text)
    slots.extend(generic_missing_slots(question, query_type))
    return dedupe_keep_order(slots)[:4]


def make_follow_up_payload(
    *,
    question: str,
    old_follow_up: Any,
    hypothesis_payload: dict[str, Any] | None,
    missing_terms: list[str],
) -> dict[str, Any]:
    base = old_follow_up if isinstance(old_follow_up, dict) else {}
    inferred_intent, inferred_query_type, inferred_answer_type = infer_intent_query_answer(question)
    base_query_type = str(base.get("query_type") or inferred_query_type)
    query_type = base_query_type if base_query_type in QUERY_TYPES else inferred_query_type
    q_terms = split_question_terms(question)
    entities = normalize_list(
        base.get("entities"),
        field="entities",
        extra_terms=hypothesis_terms(hypothesis_payload)[:8] + q_terms[:6],
        limit=10,
    )
    keywords = normalize_list(
        base.get("keywords"),
        field="keywords",
        extra_terms=missing_terms[:8] + q_terms + hypothesis_terms(hypothesis_payload),
        limit=18,
    )
    return {
        "question": question,
        "query_type": query_type,
        "entities": entities or q_terms[:4] or ["剧情"],
        "keywords": keywords or q_terms[:8] or entities,
        "expected_answer_type": str(base.get("expected_answer_type") or inferred_answer_type),
        "dialogue_context": str(base.get("dialogue_context") or ""),
    }


def has_fallback_answer(answer: str) -> bool:
    return any(marker in answer for marker in FALLBACK_ANSWER_MARKERS)


def is_no_evidence_answer(answer: str) -> bool:
    return any(marker in answer for marker in NO_EVIDENCE_MARKERS)


def is_weak_answer(answer: str) -> bool:
    return any(marker in answer for marker in WEAK_ANSWER_MARKERS)


def repair_conclusion_payload(
    payload: dict[str, Any],
    *,
    prompt: str,
    docs: dict[str, dict[str, str]],
    min_anchor_hits_for_answer: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    repaired = copy.deepcopy(payload)
    reasons: list[str] = []
    question = str(repaired.get("question") or extract_question_from_prompt(prompt)).strip()
    if not question:
        question = extract_question_from_prompt(prompt)
    repaired["question"] = question
    action = str(repaired.get("next_action") or "").strip()
    if action not in RETRIEVAL_ACTIONS:
        action = "retrieve_more"
        repaired["next_action"] = action
        reasons.append("fix_bad_action")
    inferred_intent, inferred_query_type, _ = infer_intent_query_answer(question)

    answer = clean_internal_meta(str(repaired.get("answer") or ""))
    if answer != str(repaired.get("answer") or ""):
        reasons.append("clean_answer_internal_meta")
    repaired["answer"] = answer
    repaired["clarification_question"] = str(repaired.get("clarification_question") or "")

    evidence_prompt, evidence_full, doc_ids = evidence_context(prompt, docs)
    hypothesis_payload = extract_hypothesis_from_prompt(prompt)
    profile = support_profile(
        question=question,
        answer=answer,
        evidence_text=evidence_full or evidence_prompt,
        hypothesis_payload=hypothesis_payload,
    )
    q_hits = profile["q_hits"]
    strong_evidence = bool(profile["quoted_hits"]) or len(q_hits) >= min_anchor_hits_for_answer
    terms = dedupe_keep_order(profile["q_terms"] + profile["quoted_hits"])
    round_index, max_rounds = extract_round(prompt)
    max_round_reached = round_index is not None and max_rounds is not None and round_index >= max_rounds
    generated_answer = build_grounded_answer(question, evidence_full or evidence_prompt, terms, profile)
    high_confidence_answer = bool(generated_answer) and answer_signal_ok(
        question,
        select_relevant_sentences(question, evidence_full or evidence_prompt, terms, limit=2),
        profile,
    )

    fallback_answer = has_fallback_answer(answer)
    weak_answer = is_weak_answer(answer)
    no_evidence_answer = is_no_evidence_answer(answer)
    low_support_answer = (
        action == "answer_directly"
        and bool(answer)
        and not fallback_answer
        and len(profile["answer_terms"]) >= 5
        and profile["answer_support"] < 0.35
    )

    if fallback_answer:
        reasons.append("remove_grounding_fallback_answer")
        if max_round_reached:
            repaired["next_action"] = "abstain"
            repaired["answer"] = concise_abstain_answer(question, evidence_full or evidence_prompt, terms)
            repaired["missing_slots"] = []
            repaired["follow_up_hypothesis"] = None
            reasons.append("fallback_to_abstain")
        else:
            missing_terms = [term for term in terms if term not in q_hits][:6] or split_question_terms(question)[:6]
            repaired["next_action"] = "retrieve_more"
            repaired["answer"] = ""
            repaired["missing_slots"] = make_missing_slots(question, inferred_query_type)
            repaired["follow_up_hypothesis"] = make_follow_up_payload(
                question=question,
                old_follow_up=repaired.get("follow_up_hypothesis"),
                hypothesis_payload=hypothesis_payload,
                missing_terms=missing_terms,
            )
            reasons.append("fallback_to_retrieve_more")
    elif action == "answer_directly" and (low_support_answer or no_evidence_answer):
        if max_round_reached:
            repaired["next_action"] = "abstain"
            repaired["answer"] = concise_abstain_answer(question, evidence_full or evidence_prompt, terms)
            repaired["missing_slots"] = []
            repaired["follow_up_hypothesis"] = None
            reasons.append("downgrade_low_support_answer_to_abstain")
        else:
            missing_terms = [term for term in terms if term not in q_hits][:6] or split_question_terms(question)[:6]
            repaired["next_action"] = "retrieve_more"
            repaired["answer"] = ""
            repaired["missing_slots"] = make_missing_slots(question, inferred_query_type)
            repaired["follow_up_hypothesis"] = make_follow_up_payload(
                question=question,
                old_follow_up=repaired.get("follow_up_hypothesis"),
                hypothesis_payload=hypothesis_payload,
                missing_terms=missing_terms,
            )
            reasons.append("downgrade_low_support_answer_to_retrieve_more")
    elif action == "abstain":
        repaired["answer"] = concise_abstain_answer(question, evidence_full or evidence_prompt, terms)
        repaired["missing_slots"] = []
        repaired["follow_up_hypothesis"] = None
        if weak_answer or no_evidence_answer:
            reasons.append("normalize_abstain_answer")
    elif action == "retrieve_more":
        missing_terms = [term for term in terms if term not in q_hits][:6] or split_question_terms(question)[:6]
        repaired["answer"] = ""
        repaired["missing_slots"] = make_missing_slots(
            question,
            str((repaired.get("follow_up_hypothesis") or {}).get("query_type") if isinstance(repaired.get("follow_up_hypothesis"), dict) else inferred_query_type),
            repaired.get("missing_slots"),
        )
        repaired["follow_up_hypothesis"] = make_follow_up_payload(
            question=question,
            old_follow_up=repaired.get("follow_up_hypothesis"),
            hypothesis_payload=hypothesis_payload,
            missing_terms=missing_terms,
        )
        reasons.append("normalize_retrieve_more")
    elif action == "answer_directly":
        repaired["missing_slots"] = []
        repaired["follow_up_hypothesis"] = None
    elif action == "clarify_user":
        repaired["follow_up_hypothesis"] = None

    if repaired.get("next_action") in {"answer_directly", "abstain"}:
        repaired["follow_up_hypothesis"] = None
        repaired["clarification_question"] = "" if repaired.get("next_action") != "clarify_user" else repaired["clarification_question"]
    if repaired.get("next_action") == "retrieve_more":
        repaired["clarification_question"] = ""

    profile_out = {
        "q_hits": q_hits,
        "quoted_hits": profile["quoted_hits"],
        "answer_support": round(float(profile["answer_support"]), 4),
        "doc_ids": doc_ids[:8],
        "evidence_preview": (evidence_full or evidence_prompt)[:900],
        "generated_answer": generated_answer,
    }
    return repaired, sorted(set(reasons)), profile_out


def action_of(record: dict[str, Any]) -> str:
    if task_type(record) != CONCLUSION_TASK:
        return ""
    payload = parse_json_object(assistant_text(record))
    return str(payload.get("next_action") or "") if payload else "invalid_json"


def question_of(record: dict[str, Any]) -> str:
    payload = parse_json_object(assistant_text(record))
    if payload and payload.get("question"):
        return str(payload["question"])
    meta_question = str((record.get("meta") or {}).get("source_question") or "").strip()
    if meta_question:
        return meta_question
    return extract_question_from_prompt(user_text(record))


def downsample_conclusions(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_retrieve_per_question: int,
    retrieve_ratio: float,
    abstain_ratio: float,
    min_retrieve: int,
    min_abstain: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    non_conclusions = [record for record in records if task_type(record) != CONCLUSION_TASK]
    conclusions = [record for record in records if task_type(record) == CONCLUSION_TASK]
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in conclusions:
        by_action[action_of(record)].append(record)

    answer_records = by_action.get("answer_directly", [])
    clarify_records = by_action.get("clarify_user", [])
    dropped_samples: list[dict[str, Any]] = []

    retrieve_records = by_action.get("retrieve_more", [])
    retrieve_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in retrieve_records:
        retrieve_by_question[question_of(record)].append(record)
    capped_retrieve: list[dict[str, Any]] = []
    for question, grouped in retrieve_by_question.items():
        grouped = sorted(grouped, key=lambda item: str(item.get("id") or ""))
        capped_retrieve.extend(grouped[:max_retrieve_per_question])
        for record in grouped[max_retrieve_per_question:]:
            if len(dropped_samples) < 500:
                dropped_samples.append({"id": record.get("id"), "reason": "drop_extra_retrieve_more", "question": question})

    if answer_records and len(capped_retrieve) > max(min_retrieve, int(round(len(answer_records) * retrieve_ratio))):
        target = max(1, min_retrieve, int(round(len(answer_records) * retrieve_ratio)))
        keep = {id(record) for record in rng.sample(capped_retrieve, target)}
        for record in capped_retrieve:
            if id(record) not in keep and len(dropped_samples) < 500:
                dropped_samples.append({"id": record.get("id"), "reason": "downsample_retrieve_more", "question": question_of(record)})
        capped_retrieve = [record for record in capped_retrieve if id(record) in keep]

    abstain_records = by_action.get("abstain", [])
    if answer_records and len(abstain_records) > max(min_abstain, int(round(len(answer_records) * abstain_ratio))):
        target = max(1, min_abstain, int(round(len(answer_records) * abstain_ratio)))
        keep = {id(record) for record in rng.sample(abstain_records, target)}
        for record in abstain_records:
            if id(record) not in keep and len(dropped_samples) < 500:
                dropped_samples.append({"id": record.get("id"), "reason": "downsample_abstain", "question": question_of(record)})
        abstain_records = [record for record in abstain_records if id(record) in keep]

    keep_ids = {id(record) for record in non_conclusions + answer_records + capped_retrieve + abstain_records + clarify_records}
    return [record for record in records if id(record) in keep_ids], dropped_samples


def repair_split(
    records: list[dict[str, Any]],
    *,
    docs: dict[str, dict[str, str]],
    rng: random.Random,
    min_anchor_hits_for_answer: int,
    disable_downsample: bool,
    max_retrieve_per_question: int,
    retrieve_ratio: float,
    abstain_ratio: float,
    min_retrieve: int,
    min_abstain: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    before_actions = Counter(action_of(record) for record in records if task_type(record) == CONCLUSION_TASK)
    repair_reasons: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    repaired_records: list[dict[str, Any]] = []

    for record in records:
        new_record = copy.deepcopy(record)
        for message in new_record.get("conversations") or []:
            if message.get("from") in {"human", "user"}:
                message["value"] = sanitize_prompt_text(str(message.get("value") or ""))

        message = assistant_message(new_record)
        payload = parse_json_object(str(message.get("value") or "")) if message else None
        reasons: list[str] = []
        profile: dict[str, Any] = {}
        before_payload = copy.deepcopy(payload)
        if payload is not None and message is not None:
            prompt = user_text(new_record)
            current_task = task_type(new_record)
            if current_task == INITIAL_TASK:
                repaired_payload, reasons = repair_hypothesis_payload(payload, prompt=prompt, follow_up=False)
            elif current_task == FOLLOW_UP_TASK:
                repaired_payload, reasons = repair_hypothesis_payload(payload, prompt=prompt, follow_up=True)
            elif current_task == CONCLUSION_TASK:
                repaired_payload, reasons, profile = repair_conclusion_payload(
                    payload,
                    prompt=prompt,
                    docs=docs,
                    min_anchor_hits_for_answer=min_anchor_hits_for_answer,
                )
            else:
                repaired_payload = payload
            message["value"] = dump_compact_json(repaired_payload)
            for reason in reasons:
                repair_reasons[reason] += 1
            if reasons and len(samples) < 1000:
                samples.append(
                    {
                        "id": new_record.get("id"),
                        "task_type": current_task,
                        "question": question_of(new_record),
                        "before": before_payload,
                        "after": repaired_payload,
                        "reasons": reasons,
                        "profile": profile,
                    }
                )
        repaired_records.append(new_record)

    if disable_downsample:
        after_records = repaired_records
        dropped_samples: list[dict[str, Any]] = []
    else:
        after_records, dropped_samples = downsample_conclusions(
            repaired_records,
            rng=rng,
            max_retrieve_per_question=max_retrieve_per_question,
            retrieve_ratio=retrieve_ratio,
            abstain_ratio=abstain_ratio,
            min_retrieve=min_retrieve,
            min_abstain=min_abstain,
        )
    for sample in dropped_samples:
        if len(samples) < 1000:
            samples.append(sample)

    report = {
        "records_before": len(records),
        "records_after": len(after_records),
        "task_counts_before": dict(Counter(task_type(record) for record in records)),
        "task_counts_after": dict(Counter(task_type(record) for record in after_records)),
        "conclusion_actions_before": dict(before_actions),
        "conclusion_actions_after": dict(Counter(action_of(record) for record in after_records if task_type(record) == CONCLUSION_TASK)),
        "repair_reasons": dict(repair_reasons),
    }
    return after_records, report, samples


def marker_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        text = assistant_text(record)
        for marker in FALLBACK_ANSWER_MARKERS + NO_EVIDENCE_MARKERS + WEAK_ANSWER_MARKERS:
            if marker in text:
                counts[marker] += 1
    return counts


def make_dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
        f"{dataset_name}_test": entry("test.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair short-prompt SFT data quality and factual grounding.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--min-anchor-hits-for-answer", type=int, default=2)
    parser.add_argument("--disable-downsample", action="store_true")
    parser.add_argument("--max-retrieve-per-question", type=int, default=2)
    parser.add_argument("--retrieve-ratio", type=float, default=0.80)
    parser.add_argument("--abstain-ratio", type=float, default=0.35)
    parser.add_argument("--min-retrieve", type=int, default=220)
    parser.add_argument("--min-abstain", type=int, default=120)
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    docs_path = args.documents if args.documents.is_absolute() else PROJECT_ROOT / args.documents
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")
    docs = load_documents(docs_path)
    rng = random.Random(args.seed)

    report: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "documents": str(docs_path),
        "document_count": len(docs),
        "dataset_name": output_dir.name,
        "seed": args.seed,
        "settings": {
            "min_anchor_hits_for_answer": args.min_anchor_hits_for_answer,
            "disable_downsample": args.disable_downsample,
            "max_retrieve_per_question": args.max_retrieve_per_question,
            "retrieve_ratio": args.retrieve_ratio,
            "abstain_ratio": args.abstain_ratio,
            "min_retrieve": args.min_retrieve,
            "min_abstain": args.min_abstain,
        },
        "splits": {},
    }
    all_records: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.json"
        if not input_path.exists():
            continue
        records = load_json(input_path)
        repaired, split_report, samples = repair_split(
            records,
            docs=docs,
            rng=rng,
            min_anchor_hits_for_answer=args.min_anchor_hits_for_answer,
            disable_downsample=args.disable_downsample,
            max_retrieve_per_question=args.max_retrieve_per_question,
            retrieve_ratio=args.retrieve_ratio,
            abstain_ratio=args.abstain_ratio,
            min_retrieve=args.min_retrieve,
            min_abstain=args.min_abstain,
        )
        write_json(output_dir / f"{split}.json", repaired)
        report["splits"][split] = split_report
        all_records.extend(repaired)
        for sample in samples:
            sample["split"] = split
        all_samples.extend(samples)

    build_summary = {
        "records": len(all_records),
        "task_counts": dict(Counter(task_type(record) for record in all_records)),
        "conclusion_actions": dict(Counter(action_of(record) for record in all_records if task_type(record) == CONCLUSION_TASK)),
        "assistant_marker_counts": dict(marker_counts(all_records)),
    }
    write_json(output_dir / "dataset_info.json", make_dataset_info(output_dir.name))
    write_jsonl(output_dir / "records.jsonl", all_records)
    write_jsonl(output_dir / "repair_samples.jsonl", all_samples)
    write_json(output_dir / "build_summary.json", build_summary)
    write_json(output_dir / "repair_report.json", report)
    print(json.dumps({"build_summary": build_summary, "report": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
