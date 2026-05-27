#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data/processed/opd_candidates/qwen35_4b_full_chain_sample500/candidates.jsonl"
DEFAULT_SCORES = PROJECT_ROOT / "data/processed/opd_teacher_scores/qwen35_4b_full_chain_sample500_deepseek/scores.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/opd_kto_full_chain_sample500_deepseek_v2"

SYSTEM_PROMPT = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
INITIAL_TASK = "user_question_hypothesis_generation"
CONCLUSION_TASK = "conclusion_generation"
FOLLOW_UP_TASK = "follow_up_hypothesis_generation"

HYPOTHESIS_FIELDS = (
    "question",
    "intent",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
FOLLOW_UP_FIELDS = (
    "question",
    "query_type",
    "entities",
    "keywords",
    "expected_answer_type",
    "dialogue_context",
)
CONCLUSION_FIELDS = (
    "question",
    "next_action",
    "answer",
    "missing_slots",
    "clarification_question",
    "follow_up_hypothesis",
)

FINAL_NEGATIVE_FAILURE_PREFIXES = (
    "wrong_action",
    "hallucinated_answer",
    "bad_full_chain",
    "over_abstain",
    "unnecessary_followup",
    "low_rank_hit",
    "grounding",
    "grounding_error",
    "grounding_low",
    "decision",
    "decision_error",
    "decision_quality",
    "schema_error",
)
RETRIEVAL_NEGATIVE_FAILURE_PREFIXES = (
    "no_retrieval_gain",
    "insufficient_retrieval_gain",
    "keyword_repetition",
    "alias_pollution",
    "anti_repetition",
)
GENERATION_NEGATIVE_FAILURE_PREFIXES = (
    "keyword_repetition",
    "alias_pollution",
    "anti_repetition",
)
BAD_POSITIVE_ANSWER_MARKERS = (
    "已检索到的证据能确认",
    "原答案中",
    "但原答案",
    "这部分不能确定",
    "没有在当前证据中得到直接支撑",
    "上海鹰角网络科技有限公司",
    "制作的游戏《明日方舟》",
    "及其衍生作品",
    "萌娘百科",
    "萌百",
    "现有证据未",
    "当前证据未",
    "未明确说明",
    "没有明确说明",
    "无法确认",
    "不能确认",
    "不足以确认",
    "仅显示",
)
BAD_POSITIVE_DIRECT_ANSWER_PAIRS = (
    ("月见夜为什么要每天在训练场锻炼？", "他想梓兰小姐"),
    ("德克萨斯发现空的密码是什么？空去追踪了什么？", "德克萨斯可能会成为可疑分子"),
    ("迈克尔为什么想逃离移动地块？", "卢西恩越来越确信剧团长"),
)
FRAGMENT_SUFFIXES = (
    "目",
    "真",
    "博",
    "感",
    "染者",
)
GENERIC_BAD_ENTITY_TERMS = {
    "谁",
    "什么",
    "哪里",
    "何时",
    "为何",
    "为什么",
    "原因",
    "关系",
    "身份",
    "目的",
    "学校",
    "计划",
    "事件",
    "背景",
    "真相",
    "回收者",
    "年轻人",
    "女孩",
    "学生",
    "学校关系",
    "对话内容",
    "救下者",
    "人",
    "名字",
}
GENERIC_BAD_ANSWER_TERMS = GENERIC_BAD_ENTITY_TERMS | {"原因", "动机", "过程", "细节"}
BAD_QUESTION_FRAGMENT_RE = re.compile(r"(谁策划了|为什么|是什么|有什么|怎么|如何|是否|真相|目的).{0,8}$")
BAD_ENTITY_PHRASE_RE = re.compile(r"(为什么|是什么|有什么|怎么|如何|是否|谁策划|谁杀|谁指使|真正|真实|真相|目的|关系|原因|来到|发生|决定|选择|毒害|绑架|回收|出手|遇到|看到|负责)")
BAD_ENTITY_ACTION_RE = re.compile(
    r"(造成|推动|伤害|接受|承担|引导|联系|解决|得知|放弃|公开|承认|申请|描述|夸大|说服|救下|带走|出现|组织|参与|发现|透露|警告|询问|策划|制造|毁灭|杀害|偷取|出手|回归|加入|来到|选择)"
)
BAD_ENTITY_ENDINGS = ("上", "中", "时", "后", "前", "里", "内", "外", "者")
BAD_KEYWORD_FRAGMENTS = {
    "真实目",
    "真正目",
    "否真",
    "案真相",
    "正幕后主使",
    "谁策划了绑架感",
    "染者",
    "袭击事",
    "件可能",
    "谁有关",
    "事背后隐藏着怎样",
    "阿斯帕齐娅来到博",
    "物馆",
    "轨道上",
    "敌人造成伤害并",
    "推动他们",
}
GENERIC_BAD_KEYWORD_TERMS = {
    "谁",
    "什么",
    "哪里",
    "何时",
    "为何",
    "为什么",
    "是什么",
    "有什么",
    "怎么",
    "如何",
    "是否",
}
MIN_POSITIVE_TOTAL_SCORE = 85
MIN_POSITIVE_ANSWER_SCORE = 90
MIN_POSITIVE_FOLLOWUP_SCORE = 90
MIN_POSITIVE_FINAL_RANK = 20
MIN_POSITIVE_FOLLOWUP_FINAL_RANK = 10
NEGATIVE_RATIO_CAP = 1.8
NEGATIVE_REASON_CAPS = {
    "bad_initial_hypothesis": 120,
    "weak_or_polluted_follow_up": 180,
    "bad_hypothesis_terms_in_conclusion_context": 120,
    "unnecessary_followup": 180,
    "over_abstain": 180,
}

META_TAG_RE = re.compile(r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]*\]\s*", re.IGNORECASE)
EVIDENCE_TAG_RE = re.compile(r"\[E\d+\]\s*")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_path_list(values: list[Path] | None, fallback: Path) -> list[Path]:
    if not values:
        return [resolve_path(fallback)]
    output: list[Path] = []
    for value in values:
        for part in str(value).split(","):
            if part.strip():
                output.append(resolve_path(Path(part.strip())))
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_text(text: Any, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"


def strip_internal_evidence_meta(text: str) -> str:
    text = META_TAG_RE.sub("", text)
    text = EVIDENCE_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_string_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
        if len(output) >= limit:
            break
    return output


def clean_optional_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "null", "n/a", "na"} or text in {"无", "无。"} else text


def normalize_hypothesis(payload: Any, *, follow_up: bool) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    fields = FOLLOW_UP_FIELDS if follow_up else HYPOTHESIS_FIELDS
    output: dict[str, Any] = {}
    for field in fields:
        if field in {"entities", "keywords"}:
            output[field] = clean_string_list(payload.get(field), limit=12 if field == "entities" else 20)
        else:
            output[field] = clean_optional_text(payload.get(field))
    if not output.get("question"):
        return None
    return output


def normalize_conclusion(payload: Any, *, question: str = "") -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    output = {field: payload.get(field) for field in CONCLUSION_FIELDS}
    output["question"] = str(output.get("question") or question or "").strip()
    output["next_action"] = str(output.get("next_action") or "").strip()
    output["answer"] = str(output.get("answer") or "").strip()
    output["missing_slots"] = clean_string_list(output.get("missing_slots"), limit=6)
    output["clarification_question"] = clean_optional_text(output.get("clarification_question"))
    follow_up = normalize_hypothesis(output.get("follow_up_hypothesis"), follow_up=True)
    output["follow_up_hypothesis"] = follow_up if output["next_action"] == "retrieve_more" else None
    if not output["question"] or output["next_action"] not in {
        "answer_directly",
        "retrieve_more",
        "clarify_user",
        "abstain",
    }:
        return None
    return output


def render_evidence_brief(items: list[dict[str, Any]], *, max_items: int, max_chars: int) -> list[str]:
    brief: list[str] = []
    for item in items[:max_items]:
        doc_id = item.get("id") or item.get("doc_id") or ""
        stage = item.get("stage_code") or ""
        story = item.get("story_name") or ""
        activity = item.get("activity_name") or ""
        label = " / ".join(str(part) for part in (activity, story, stage) if part) or str(doc_id)
        text = (
            item.get("snippet")
            or item.get("evidence_chain_text")
            or item.get("clean_text")
            or ""
        )
        text = compact_text(strip_internal_evidence_meta(str(text)), max_chars)
        if text:
            brief.append(f"{doc_id}: {label}: {text}" if doc_id else f"{label}: {text}")
    return brief


def build_initial_prompt(question: str, dialogue_context: str) -> str:
    lines = [f"task: {INITIAL_TASK}", f"question: {question}"]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append("output_schema: hypothesis_v2")
    return "\n".join(lines)


def build_conclusion_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: dict[str, Any],
    round_index: int,
    max_rounds: int,
    evidence_summary: list[dict[str, Any]],
    max_evidence_items: int,
    max_evidence_chars: int,
) -> str:
    lines = [f"task: {CONCLUSION_TASK}", f"question: {question}"]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append(f"hypothesis: {compact_json(hypothesis)}")
    lines.append(f"round: {round_index}/{max_rounds}")
    brief = render_evidence_brief(evidence_summary, max_items=max_evidence_items, max_chars=max_evidence_chars)
    if brief:
        lines.append("evidence_brief:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(brief, start=1))
    lines.append("output_schema: conclusion_v2")
    return "\n".join(lines)


def build_follow_up_prompt(
    *,
    question: str,
    dialogue_context: str,
    hypothesis: dict[str, Any],
    conclusion: dict[str, Any],
    round_index: int,
    max_rounds: int,
    evidence_summary: list[dict[str, Any]],
    max_evidence_items: int,
    max_evidence_chars: int,
) -> str:
    lines = [f"task: {FOLLOW_UP_TASK}", f"question: {question}"]
    if dialogue_context:
        lines.append(f"dialogue_context: {compact_text(dialogue_context, 260)}")
    lines.append(f"hypothesis: {compact_json(hypothesis)}")
    lines.append(f"round: {round_index}/{max_rounds}")
    brief = render_evidence_brief(evidence_summary, max_items=max_evidence_items, max_chars=max_evidence_chars)
    if brief:
        lines.append("evidence_brief:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(brief, start=1))
    missing_slots = clean_string_list(conclusion.get("missing_slots"), limit=5)
    if missing_slots:
        lines.append("missing_slots: " + "; ".join(missing_slots))
    lines.append("output_schema: follow_up_hypothesis_v2")
    return "\n".join(lines)


def bucket(metrics: dict[str, Any]) -> str:
    hit = bool(metrics.get("final_hit"))
    action = str(metrics.get("final_action") or "")
    if hit and action == "answer_directly":
        return "hit_answer"
    if hit and action == "abstain":
        return "hit_abstain"
    if (not hit) and action == "answer_directly":
        return "miss_answer"
    if (not hit) and action == "abstain":
        return "miss_abstain"
    return f"other:{action or 'unknown'}"


def metric_int(metrics: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(metrics.get(key))
    except (TypeError, ValueError):
        return default


def is_answered_high_rank_hit(metrics: dict[str, Any], *, max_rank: int = MIN_POSITIVE_FINAL_RANK) -> bool:
    return (
        bool(metrics.get("final_hit"))
        and str(metrics.get("final_action") or "") == "answer_directly"
        and metric_int(metrics, "final_rank_value", 10**9) <= max_rank
    )


def is_over_abstain(metrics: dict[str, Any]) -> bool:
    return bool(metrics.get("final_hit")) and str(metrics.get("final_action") or "") == "abstain"


def failure_prefixes(score: dict[str, Any]) -> set[str]:
    prefixes: set[str] = set()
    for item in score.get("hard_failures") or []:
        text = str(item or "").strip()
        if not text:
            continue
        prefixes.add(text.split(":", 1)[0].strip())
    return prefixes


def has_any(prefixes: set[str], candidates: tuple[str, ...]) -> bool:
    return any(prefix.startswith(candidate) for prefix in prefixes for candidate in candidates)


def has_list_repetition(payload: dict[str, Any]) -> bool:
    for field in ("entities", "keywords"):
        values = [str(item).strip() for item in payload.get(field) or [] if str(item).strip()]
        if len(values) != len(set(values)):
            return True
        for value in values:
            parts = [part for part in re.split(r"[、,，/\\s]+", value) if part]
            if len(parts) >= 4 and len(set(parts)) <= 2:
                return True
    return False


def payload_terms(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    terms: list[str] = []
    for field in ("entities", "keywords"):
        values = payload.get(field)
        if isinstance(values, list):
            terms.extend(str(item or "").strip() for item in values if str(item or "").strip())
    return terms


def looks_like_bad_entity(term: str, question: str = "") -> bool:
    text = str(term or "").strip()
    if not text:
        return True
    in_question = bool(question and text in question)
    if looks_like_question_span_fragment(text, question):
        return True
    if in_question and text not in BAD_KEYWORD_FRAGMENTS and text not in {"谁", "什么", "哪里", "何时", "为何", "为什么"}:
        return False
    if text in GENERIC_BAD_ENTITY_TERMS:
        return True
    if text in BAD_KEYWORD_FRAGMENTS:
        return True
    if len(text) == 2 and text in {"否真", "真正", "什么", "为何"}:
        return True
    if BAD_QUESTION_FRAGMENT_RE.search(text):
        return True
    if len(text) >= 3 and any(text.endswith(suffix) for suffix in FRAGMENT_SUFFIXES):
        return True
    if len(text) >= 4 and BAD_ENTITY_PHRASE_RE.search(text):
        return True
    if len(text) >= 4 and BAD_ENTITY_ACTION_RE.search(text):
        return True
    if len(text) >= 3 and text.endswith(BAD_ENTITY_ENDINGS) and text not in question:
        return True
    if len(text) >= 3 and text.endswith(BAD_ENTITY_ENDINGS) and not in_question and any(char in text for char in ("上", "中", "时", "后", "前")):
        return True
    if len(text) >= 8 and question and text not in question and not any(char in text for char in "·-/"):
        # Long Chinese chunks that are neither aliases nor literal question terms.
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        if chinese_chars >= 8:
            return True
    return False


def looks_like_bad_keyword(term: str, question: str = "") -> bool:
    text = str(term or "").strip()
    if not text:
        return True
    if looks_like_question_span_fragment(text, question):
        return True
    if text in BAD_KEYWORD_FRAGMENTS:
        return True
    if text in GENERIC_BAD_KEYWORD_TERMS:
        return True
    if question and text in question:
        return False
    if len(text) >= 3 and any(text.endswith(suffix) for suffix in FRAGMENT_SUFFIXES):
        return True
    if len(text) >= 8 and any(marker in text for marker in ("谁策划", "是否真", "为什么", "是什么")):
        return True
    return False


def looks_like_question_span_fragment(text: str, question: str = "") -> bool:
    if text in BAD_KEYWORD_FRAGMENTS:
        return True
    if not question or text not in question:
        return False
    start = 0
    while True:
        index = question.find(text, start)
        if index < 0:
            return False
        prev_char = question[index - 1] if index > 0 else ""
        next_index = index + len(text)
        next_char = question[next_index] if next_index < len(question) else ""
        if text.endswith("事") and next_char == "件":
            return True
        if text.startswith("件") and prev_char == "事":
            return True
        if text.endswith("博") and next_char == "物":
            return True
        if text.startswith("物") and prev_char == "博":
            return True
        if len(text) >= 3 and text.endswith("目") and next_char == "的" and text[-2:] in {"实目", "正目"}:
            return True
        if text.startswith(("谁", "为何", "为什么", "是否")) and len(text) >= 3:
            return True
        if "造成伤害并" in text or text.endswith(("造成伤害并", "隐藏着怎样")):
            return True
        if text in {"轨道上", "推动他们", "推动它们"}:
            return True
        start = index + 1


def has_bad_terms(payload: dict[str, Any] | None, *, question: str = "") -> bool:
    if not isinstance(payload, dict):
        return True
    entities = [str(item or "").strip() for item in payload.get("entities") or [] if str(item or "").strip()]
    keywords = [str(item or "").strip() for item in payload.get("keywords") or [] if str(item or "").strip()]
    return any(looks_like_bad_entity(term, question) for term in entities) or any(
        looks_like_bad_keyword(term, question) for term in keywords
    )


def normalized_term_set(payload: dict[str, Any] | None) -> set[str]:
    return {term for term in payload_terms(payload) if term}


def follow_up_delta(
    *,
    previous_hypothesis: dict[str, Any] | None,
    follow_payload: dict[str, Any] | None,
) -> set[str]:
    return normalized_term_set(follow_payload) - normalized_term_set(previous_hypothesis)


def has_strong_follow_up_delta(
    *,
    previous_hypothesis: dict[str, Any] | None,
    follow_payload: dict[str, Any] | None,
    question: str,
) -> bool:
    delta = {
        term
        for term in follow_up_delta(previous_hypothesis=previous_hypothesis, follow_payload=follow_payload)
        if not looks_like_bad_keyword(term, question) and not looks_like_bad_entity(term, question)
    }
    return len(delta) >= 2 or any(len(term) >= 4 for term in delta)


def is_bad_follow_up_payload(
    *,
    previous_hypothesis: dict[str, Any] | None,
    follow_payload: dict[str, Any] | None,
    question: str,
) -> bool:
    return (
        follow_payload is None
        or has_bad_terms(follow_payload, question=question)
        or has_list_repetition(follow_payload)
        or not has_strong_follow_up_delta(
            previous_hypothesis=previous_hypothesis,
            follow_payload=follow_payload,
            question=question,
        )
    )


def evidence_mentions_terms(evidence_summary: list[dict[str, Any]], terms: list[str], *, min_hits: int = 1) -> bool:
    if not evidence_summary or not terms:
        return False
    evidence_text = "\n".join(
        str(item.get("snippet") or item.get("clean_text") or item.get("evidence_chain_text") or "")
        for item in evidence_summary
        if isinstance(item, dict)
    )
    if not evidence_text:
        return False
    hits = 0
    for term in terms:
        text = str(term or "").strip()
        if len(text) >= 2 and text in evidence_text:
            hits += 1
        if hits >= min_hits:
            return True
    return False


def has_bad_positive_answer_style(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    answer = str(payload.get("answer") or "")
    if not answer:
        return False
    question = str(payload.get("question") or "")
    if any(question == bad_question and bad_answer in answer for bad_question, bad_answer in BAD_POSITIVE_DIRECT_ANSWER_PAIRS):
        return True
    return any(marker in answer for marker in BAD_POSITIVE_ANSWER_MARKERS)


def should_skip_positive_conclusion(
    *,
    conclusion: dict[str, Any] | None,
    previous_hypothesis: dict[str, Any] | None,
    is_final_step: bool,
    sample_bucket: str,
    total_score: Any,
    evidence_summary: list[dict[str, Any]],
    question: str,
    metrics: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(conclusion, dict):
        return True
    if has_bad_positive_answer_style(conclusion):
        return True
    if isinstance(total_score, (int, float)) and total_score < MIN_POSITIVE_TOTAL_SCORE:
        return True
    if metrics is not None and not is_answered_high_rank_hit(metrics):
        return True
    action = str(conclusion.get("next_action") or "")
    if action == "retrieve_more":
        follow_payload = normalize_hypothesis(conclusion.get("follow_up_hypothesis"), follow_up=True)
        if is_bad_follow_up_payload(
            previous_hypothesis=previous_hypothesis,
            follow_payload=follow_payload,
            question=question,
        ):
            return True
    if action == "answer_directly":
        if conclusion.get("missing_slots"):
            return True
        if isinstance(total_score, (int, float)) and total_score < MIN_POSITIVE_ANSWER_SCORE:
            return True
    return bool(is_final_step and sample_bucket == "hit_abstain")


def make_record(
    *,
    record_id: str,
    task_type: str,
    prompt: str,
    payload: dict[str, Any],
    kto_tag: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "opd_kto",
        "system": SYSTEM_PROMPT,
        "tools": "[]",
        "kto_tag": kto_tag,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": compact_json(payload)},
        ],
        "meta": meta,
    }


def dedupe_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    dropped = 0
    for record in records:
        conversations = record.get("conversations") or []
        prompt = conversations[0].get("value") if conversations else ""
        response = conversations[1].get("value") if len(conversations) > 1 else ""
        key = (
            record.get("task_type"),
            bool(record.get("kto_tag")),
            str(prompt),
            str(response),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(record)
    return deduped, dropped


def cap_negative_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    negative_ratio_cap: float,
    reason_caps: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    positives = [record for record in records if record.get("kto_tag")]
    negatives = [record for record in records if not record.get("kto_tag")]
    if not positives or not negatives:
        return records, 0

    rng = random.Random(seed)
    by_reason: dict[str, list[dict[str, Any]]] = {}
    for record in negatives:
        reason = str(record.get("meta", {}).get("preference_reason") or "unknown")
        by_reason.setdefault(reason, []).append(record)

    selected_negatives: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    for reason, items in sorted(by_reason.items()):
        rng.shuffle(items)
        cap = reason_caps.get(reason)
        if cap is None:
            selected_negatives.extend(items)
            continue
        selected_negatives.extend(items[:cap])
        overflow.extend(items[cap:])

    max_negatives = max(1, int(round(len(positives) * negative_ratio_cap)))
    if len(selected_negatives) > max_negatives:
        rng.shuffle(selected_negatives)
        selected_negatives = selected_negatives[:max_negatives]
    elif len(selected_negatives) < max_negatives and overflow:
        rng.shuffle(overflow)
        selected_negatives.extend(overflow[: max_negatives - len(selected_negatives)])

    capped = positives + selected_negatives
    rng.shuffle(capped)
    return capped, len(records) - len(capped)


def split_records_by_source(
    records: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(records) <= 10:
        return records, []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        source_id = str(record.get("meta", {}).get("source_candidate_id") or record.get("id") or index)
        by_source.setdefault(source_id, []).append(record)

    rng = random.Random(seed)
    source_ids = list(by_source)
    rng.shuffle(source_ids)
    target_val_size = max(1, int(round(len(records) * val_ratio)))
    val_source_ids: set[str] = set()
    val_count = 0
    for source_id in source_ids:
        if val_count >= target_val_size and val_source_ids:
            break
        val_source_ids.add(source_id)
        val_count += len(by_source[source_id])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for source_id in source_ids:
        target = val_records if source_id in val_source_ids else train_records
        target.extend(by_source[source_id])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def add_record(
    records: list[dict[str, Any]],
    seen: set[str],
    *,
    record_id: str,
    task_type: str,
    prompt: str,
    payload: dict[str, Any] | None,
    kto_tag: bool,
    meta: dict[str, Any],
) -> bool:
    if payload is None or record_id in seen:
        return False
    seen.add(record_id)
    records.append(
        make_record(
            record_id=record_id,
            task_type=task_type,
            prompt=prompt,
            payload=payload,
            kto_tag=kto_tag,
            meta=meta,
        )
    )
    return True


def add_negative_if_present(
    records: list[dict[str, Any]],
    seen: set[str],
    *,
    record_id: str,
    task_type: str,
    prompt: str,
    payload: dict[str, Any] | None,
    meta: dict[str, Any],
) -> bool:
    return add_record(
        records,
        seen,
        record_id=record_id,
        task_type=task_type,
        prompt=prompt,
        payload=payload,
        kto_tag=False,
        meta=meta,
    )


def build_records_for_candidate(
    candidate: dict[str, Any],
    score: dict[str, Any],
    *,
    max_evidence_items: int,
    max_evidence_chars: int,
) -> list[dict[str, Any]]:
    cid = str(candidate.get("candidate_id") or "")
    question = str(candidate.get("question") or "").strip()
    dialogue_context = str(candidate.get("dialogue_context") or "").strip()
    payload = candidate.get("candidate") if isinstance(candidate.get("candidate"), dict) else {}
    trace = payload.get("retrieval_trace") if isinstance(payload.get("retrieval_trace"), list) else []
    metrics = candidate.get("retrieval_metrics") if isinstance(candidate.get("retrieval_metrics"), dict) else {}
    score_prefixes = failure_prefixes(score)
    sample_bucket = bucket(metrics)
    accepted = bool(score.get("accept"))
    total_score = score.get("total_score")
    max_rounds = max([int(step.get("round") or 0) for step in trace] + [int(metrics.get("rounds_run") or 0), 1])

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_meta = {
        "source_candidate_id": cid,
        "source_index": score.get("source_index"),
        "sample_bucket": sample_bucket,
        "teacher_accept": accepted,
        "teacher_total_score": total_score,
        "teacher_hard_failures": score.get("hard_failures") or [],
        "teacher_reasons": score.get("reasons") or [],
        "retrieval_metrics": metrics,
    }
    initial_payload = normalize_hypothesis((trace[0].get("hypothesis") if trace else payload.get("final_hypothesis")), follow_up=False)
    initial_prompt = build_initial_prompt(question, dialogue_context)
    initial_is_bad = initial_payload is None or has_bad_terms(initial_payload, question=question) or has_list_repetition(initial_payload or {})
    accepted_initial = (
        accepted
        and not initial_is_bad
        and isinstance(total_score, (int, float))
        and total_score >= MIN_POSITIVE_TOTAL_SCORE
        and is_answered_high_rank_hit(metrics)
    )

    if accepted_initial:
        add_record(
            records,
            seen,
            record_id=f"{cid}-pos-initial",
            task_type=INITIAL_TASK,
            prompt=initial_prompt,
            payload=initial_payload,
            kto_tag=True,
            meta={**base_meta, "preference_reason": "accepted_full_chain_initial"},
        )
    elif initial_is_bad or has_any(score_prefixes, GENERATION_NEGATIVE_FAILURE_PREFIXES) or (
        has_any(score_prefixes, RETRIEVAL_NEGATIVE_FAILURE_PREFIXES)
        and initial_payload is not None
        and has_list_repetition(initial_payload)
    ):
        add_record(
            records,
            seen,
            record_id=f"{cid}-neg-initial",
            task_type=INITIAL_TASK,
            prompt=initial_prompt,
            payload=initial_payload,
            kto_tag=False,
            meta={**base_meta, "preference_reason": "bad_initial_hypothesis"},
        )

    final_step = trace[-1] if trace else {}
    final_conclusion = normalize_conclusion(final_step.get("conclusion"), question=question)
    is_wrong_abstain = is_over_abstain(metrics)
    is_miss_answer = (not metrics.get("final_hit")) and metrics.get("final_action") == "answer_directly" and not accepted
    is_unnecessary_followup = bool(metrics.get("final_hit")) and metrics.get("final_action") == "retrieve_more"
    final_negative = (
        is_wrong_abstain
        or is_unnecessary_followup
        or is_miss_answer
        or has_any(score_prefixes, FINAL_NEGATIVE_FAILURE_PREFIXES)
        or (not accepted and isinstance(total_score, (int, float)) and total_score <= 65)
    )
    retrieval_negative = (not accepted) and has_any(score_prefixes, RETRIEVAL_NEGATIVE_FAILURE_PREFIXES)

    for step_index, step in enumerate(trace):
        round_index = int(step.get("round") or (step_index + 1))
        hypothesis = normalize_hypothesis(step.get("hypothesis"), follow_up=False)
        conclusion = normalize_conclusion(step.get("conclusion"), question=question)
        evidence_summary = step.get("evidence_summary") if isinstance(step.get("evidence_summary"), list) else []
        if hypothesis is None:
            hypothesis = initial_payload or {}
        hypothesis_bad = has_bad_terms(hypothesis, question=question) or has_list_repetition(hypothesis)
        conclusion_prompt = build_conclusion_prompt(
            question=question,
            dialogue_context=dialogue_context,
            hypothesis=hypothesis,
            round_index=round_index,
            max_rounds=max_rounds,
            evidence_summary=evidence_summary,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
        )
        if accepted and not should_skip_positive_conclusion(
            conclusion=conclusion,
            previous_hypothesis=hypothesis,
            is_final_step=step is final_step,
            sample_bucket=sample_bucket,
            total_score=total_score,
            evidence_summary=evidence_summary,
            question=question,
            metrics=metrics,
        ):
            add_record(
                records,
                seen,
                record_id=f"{cid}-pos-conclusion-r{round_index}",
                task_type=CONCLUSION_TASK,
                prompt=conclusion_prompt,
                payload=conclusion,
                kto_tag=True,
                meta={**base_meta, "preference_reason": "accepted_full_chain_conclusion", "round": round_index},
            )
        elif final_negative and step is final_step:
            reason = (
                "over_abstain"
                if is_wrong_abstain
                else "unnecessary_followup"
                if is_unnecessary_followup
                else "miss_answer"
                if is_miss_answer
                else "bad_final_conclusion"
            )
            add_negative_if_present(
                records,
                seen,
                record_id=f"{cid}-neg-conclusion-r{round_index}",
                task_type=CONCLUSION_TASK,
                prompt=conclusion_prompt,
                payload=final_conclusion or conclusion,
                meta={**base_meta, "preference_reason": reason, "round": round_index},
            )
        elif hypothesis_bad and conclusion is not None:
            add_negative_if_present(
                records,
                seen,
                record_id=f"{cid}-neg-conclusion-hypothesis-r{round_index}",
                task_type=CONCLUSION_TASK,
                prompt=conclusion_prompt,
                payload=conclusion,
                meta={**base_meta, "preference_reason": "bad_hypothesis_terms_in_conclusion_context", "round": round_index},
            )

        follow_payload = None
        if conclusion is not None:
            follow_payload = normalize_hypothesis(conclusion.get("follow_up_hypothesis"), follow_up=True)
        if follow_payload is None:
            continue
        follow_bad = is_bad_follow_up_payload(
            previous_hypothesis=hypothesis,
            follow_payload=follow_payload,
            question=question,
        )
        follow_prompt = build_follow_up_prompt(
            question=question,
            dialogue_context=dialogue_context,
            hypothesis=hypothesis,
            conclusion=conclusion or {},
            round_index=round_index,
            max_rounds=max_rounds,
            evidence_summary=evidence_summary,
            max_evidence_items=max_evidence_items,
            max_evidence_chars=max_evidence_chars,
        )
        positive_follow = (
            accepted
            and not follow_bad
            and sample_bucket == "hit_answer"
            and is_answered_high_rank_hit(metrics, max_rank=MIN_POSITIVE_FOLLOWUP_FINAL_RANK)
            and isinstance(total_score, (int, float))
            and total_score >= MIN_POSITIVE_FOLLOWUP_SCORE
        )
        if positive_follow:
            add_record(
                records,
                seen,
                record_id=f"{cid}-pos-followup-r{round_index}",
                task_type=FOLLOW_UP_TASK,
                prompt=follow_prompt,
                payload=follow_payload,
                kto_tag=True,
                meta={**base_meta, "preference_reason": "accepted_full_chain_follow_up", "round": round_index},
            )
        elif retrieval_negative or follow_bad:
            reason = "bad_retrieval_follow_up"
            if follow_bad:
                reason = "weak_or_polluted_follow_up"
            add_negative_if_present(
                records,
                seen,
                record_id=f"{cid}-neg-followup-r{round_index}",
                task_type=FOLLOW_UP_TASK,
                prompt=follow_prompt,
                payload=follow_payload,
                meta={**base_meta, "preference_reason": reason, "round": round_index},
            )
    return records


def dataset_info(dataset_name: str) -> dict[str, Any]:
    role_tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "observation_tag": "observation",
        "function_tag": "function_call",
    }

    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": role_tags,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLaMA-Factory KTO data from OPD full-chain teacher scores.")
    parser.add_argument("--candidates", type=Path, action="append", default=None)
    parser.add_argument("--scores", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--max-evidence-items", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=360)
    parser.add_argument("--negative-ratio-cap", type=float, default=NEGATIVE_RATIO_CAP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates_paths = parse_path_list(args.candidates, DEFAULT_CANDIDATES)
    scores_paths = parse_path_list(args.scores, DEFAULT_SCORES)
    if len(candidates_paths) != len(scores_paths):
        raise SystemExit(f"--candidates and --scores counts differ: {len(candidates_paths)} != {len(scores_paths)}")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    input_stats: Counter[str] = Counter()
    score_stats: Counter[str] = Counter()
    input_candidates_total = 0
    input_scores_total = 0
    for pair_index, (candidates_path, scores_path) in enumerate(zip(candidates_paths, scores_paths)):
        candidates = {f"p{pair_index}:{item['candidate_id']}": item for item in read_jsonl(candidates_path)}
        scores = read_jsonl(scores_path)
        score_by_id = {f"p{pair_index}:{item['candidate_id']}": item for item in scores}
        input_candidates_total += len(candidates)
        input_scores_total += len(scores)
        input_stats[f"source:{pair_index}:candidates"] += len(candidates)
        input_stats[f"source:{pair_index}:scores"] += len(scores)
        for scoped_candidate_id, candidate in candidates.items():
            score = score_by_id.get(scoped_candidate_id)
            if score is None:
                input_stats["missing_score"] += 1
                continue
            candidate = dict(candidate)
            score = dict(score)
            original_candidate_id = str(candidate.get("candidate_id") or "")
            candidate["candidate_id"] = f"s{pair_index}-{original_candidate_id}"
            score["candidate_id"] = candidate["candidate_id"]
            metrics = candidate.get("retrieval_metrics") if isinstance(candidate.get("retrieval_metrics"), dict) else {}
            input_stats[f"bucket:{bucket(metrics)}"] += 1
            input_stats[f"accept:{bool(score.get('accept'))}"] += 1
            for failure in score.get("hard_failures") or []:
                score_stats[f"hard_failure:{str(failure).split(':', 1)[0].strip()}"] += 1
            built = build_records_for_candidate(
                candidate,
                score,
                max_evidence_items=max(1, args.max_evidence_items),
                max_evidence_chars=max(80, args.max_evidence_chars),
            )
            records.extend(built)

    records_before_dedupe = len(records)
    records, duplicate_records_dropped = dedupe_records(records)
    records_after_dedupe = len(records)
    records, negative_records_capped = cap_negative_records(
        records,
        seed=args.seed,
        negative_ratio_cap=max(0.5, args.negative_ratio_cap),
        reason_caps=NEGATIVE_REASON_CAPS,
    )

    train_records, val_records = split_records_by_source(
        records,
        seed=args.seed,
        val_ratio=max(0.0, min(0.5, args.val_ratio)),
    )

    write_json(output_dir / "train.json", train_records)
    write_json(output_dir / "val.json", val_records)
    write_json(output_dir / "dataset_info.json", dataset_info(output_dir.name))
    write_jsonl(output_dir / "audit_records.jsonl", records)

    record_stats = Counter()
    for record in records:
        record_stats[f"kto_tag:{record['kto_tag']}"] += 1
        record_stats[f"task:{record['task_type']}"] += 1
        reason = record.get("meta", {}).get("preference_reason")
        if reason:
            record_stats[f"reason:{reason}"] += 1
        if record["kto_tag"]:
            record_stats[f"positive_task:{record['task_type']}"] += 1
        else:
            record_stats[f"negative_task:{record['task_type']}"] += 1

    summary = {
        "candidates": [str(path) for path in candidates_paths],
        "scores": [str(path) for path in scores_paths],
        "output_dir": str(output_dir),
        "input_candidates": input_candidates_total,
        "input_scores": input_scores_total,
        "records_before_dedupe": records_before_dedupe,
        "duplicate_records_dropped": duplicate_records_dropped,
        "records_after_dedupe": records_after_dedupe,
        "negative_records_capped": negative_records_capped,
        "negative_ratio_cap": args.negative_ratio_cap,
        "negative_reason_caps": NEGATIVE_REASON_CAPS,
        "records_total": len(records),
        "records_train": len(train_records),
        "records_val": len(val_records),
        "split": "grouped_by_source_candidate_id",
        "input_stats": dict(input_stats),
        "score_stats": dict(score_stats),
        "record_stats": dict(record_stats),
        "notes": [
            "KTO positives are teacher accepted runtime-stage outputs.",
            "KTO negatives are explicit hard failures plus soft wrong-abstain/miss-answer final decisions.",
            "Grey cases without accept or actionable failure are not used for training.",
            "Accepted final hit_abstain conclusions are not used as positives to avoid reinforcing over-abstention.",
        ],
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
