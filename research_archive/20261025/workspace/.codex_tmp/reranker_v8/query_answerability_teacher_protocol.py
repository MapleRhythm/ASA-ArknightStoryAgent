#!/usr/bin/env python3
"""Query-only teacher protocol for answerability reranker supervision.

The teacher sees a query and a shuffled fixed candidate pool.  Reference
answers, gold evidence, old labels, retrieval ranks, scores, and source IDs are
never included in the payload.  Every candidate is judged independently so
the result can supervise a pointwise ``(query, candidate)`` cross encoder.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

PROTOCOL_VERSION = "query_answerability_v3"
PROMPT_VERSION = "query_answerability_v3.5_positive_caps"

MAX_LABEL_COUNTS = {
    "direct_answer": 12,
    "partial_answer": 20,
    "supporting_context": 16,
    "hard_context": 16,
}

LABEL_ORDER = {
    "direct_answer": 4,
    "partial_answer": 3,
    "supporting_context": 2,
    "hard_context": 1,
    "irrelevant": 0,
}
LABEL_CODES = {
    "d": "direct_answer",
    "direct": "direct_answer",
    "direct_answer": "direct_answer",
    "p": "partial_answer",
    "partial": "partial_answer",
    "partial_answer": "partial_answer",
    "s": "supporting_context",
    "supporting": "supporting_context",
    "supporting_context": "supporting_context",
    "x": "hard_context",
    "hard": "hard_context",
    "hard_context": "hard_context",
    "i": "irrelevant",
    "irrelevant": "irrelevant",
}
RECORD_FIELDS = {"query", "query_type", "source_name", "story_family", "candidates"}
CANDIDATE_FIELDS = {"candidate_id", "text", "kind"}
FORBIDDEN_RECORD_FIELDS = {
    "answer",
    "reference_answer",
    "claims",
    "gold_evidence",
    "gold_evidence_ids",
    "coverable_evidence_ids",
    "required_evidence",
    "answer_evidence",
    "answer_focus",
}
FORBIDDEN_CANDIDATE_FIELDS = {
    "retrieval_rank",
    "original_rank",
    "doc_id",
    "story_id",
    "stage_code",
    "rerank_score",
    "fusion_score",
    "score",
    "covered_evidence_ids",
    "relevance",
    "label",
}
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    return WHITESPACE_RE.sub("", str(text or ""))


def normalize_quote_text(text: Any) -> str:
    """Normalize layout punctuation commonly altered while copying a span."""

    return normalize_text(text).replace("…", "...").replace("⋯", "...")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def record_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_name', '')}\n{record.get('query', '')}"


def clean_candidate(candidate: dict[str, Any], *, candidate_id: str | None = None) -> dict[str, str]:
    return {
        "candidate_id": str(candidate_id or candidate.get("candidate_id") or ""),
        "text": str(candidate.get("text") or ""),
        "kind": str(candidate.get("kind") or "document"),
    }


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only auditable query-only fields; notably there is no answer."""

    return {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "source_name": str(record.get("source_name") or ""),
        "story_family": str(record.get("story_family") or ""),
        "candidates": [clean_candidate(item) for item in record.get("candidates") or []],
    }


def leaked_supervision_fields(record: dict[str, Any]) -> list[str]:
    leaked = [field for field in FORBIDDEN_RECORD_FIELDS if field in record]
    for index, candidate in enumerate(record.get("candidates") or [], start=1):
        if not isinstance(candidate, dict):
            leaked.append(f"candidates[{index}](not_object)")
            continue
        leaked.extend(
            f"candidates[{index}].{field}"
            for field in FORBIDDEN_CANDIDATE_FIELDS
            if field in candidate
        )
    return sorted(leaked)


def validate_input_record(
    record: dict[str, Any], expected_candidates: int | None, *, exact_fields: bool = True
) -> None:
    if not isinstance(record, dict):
        raise ValueError("input record is not an object")
    leaked = leaked_supervision_fields(record)
    if leaked:
        raise ValueError(f"input leaks answer/gold/legacy supervision: {leaked[:12]}")
    if exact_fields:
        unknown = sorted(set(record) - RECORD_FIELDS)
        missing = sorted(RECORD_FIELDS - set(record))
        if unknown or missing:
            raise ValueError(f"invalid record fields: missing={missing} unknown={unknown}")
    if not str(record.get("query") or "").strip():
        raise ValueError("input has an empty query")
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("input has no candidates list")
    if expected_candidates is not None and len(candidates) != expected_candidates:
        raise ValueError(f"candidate count is {len(candidates)}, expected {expected_candidates}")
    candidate_ids: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} is not an object")
        if exact_fields:
            unknown = sorted(set(candidate) - CANDIDATE_FIELDS)
            missing = sorted(CANDIDATE_FIELDS - set(candidate))
            if unknown or missing:
                raise ValueError(
                    f"candidate {index} fields invalid: missing={missing} unknown={unknown}"
                )
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError(f"candidate {index} has an empty ID")
        if not str(candidate.get("text") or "").strip():
            raise ValueError(f"candidate {candidate_id} has empty text")
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("input contains duplicate candidate IDs")


def _prompt_candidate_aliases(record: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    source_candidates = list(record.get("candidates") or [])
    seed_material = f"{record.get('source_name', '')}\n{record.get('query', '')}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(source_candidates)
    candidates: list[dict[str, str]] = []
    alias_to_stored_id: dict[str, str] = {}
    for index, candidate in enumerate(source_candidates, start=1):
        alias = f"P{index:03d}"
        alias_to_stored_id[alias] = str(candidate.get("candidate_id") or "")
        candidates.append(clean_candidate(candidate, candidate_id=alias))
    return candidates, alias_to_stored_id


def prompt_candidate_aliases(record: dict[str, Any]) -> dict[str, str]:
    return _prompt_candidate_aliases(record)[1]


def teacher_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return the exact teacher-visible data (query, type, candidate text)."""

    candidates, _ = _prompt_candidate_aliases(record)
    return {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "candidates": candidates,
    }


def build_messages(
    record: dict[str, Any], expected_candidates: int | None = None
) -> list[dict[str, str]]:
    validate_input_record(record, expected_candidates)
    data = teacher_payload(record)
    candidate_count = len(data["candidates"])
    expected_count = expected_candidates if expected_candidates is not None else candidate_count
    system = (
        "你是 ASA 检索重排数据的严格标注器。目标是判断每条候选正文自身对回答用户问题的帮助程度。"
        "你没有参考答案，也不得使用外部知识或脑补缺失事实。候选顺序和 P 编号均为稳定随机别名，"
        "不代表检索名次。每条候选必须独立判断，不得把多条候选拼接后再给其中一条加分。"
        "Return exactly one valid json object (JSON) and no prose outside json."
    )
    instruction = {
        "protocol": PROTOCOL_VERSION,
        "task": [
            f"阅读全部 {candidate_count} 条候选；ranking 必须返回 exactly {expected_count} 个 candidate_id，每个输入 ID 恰好一次。",
            "先输出 query_decision：[\"answerable\",\"\"] 或 [\"ambiguous\",\"无法唯一消解的简短原因\"]。ambiguous 时必须遵守后述歧义拒标规则。",
            "先做对象绑定门禁：识别 query 所问的具体人物、群体、物件、事件与关系。候选若明确说的是另一个人物、另一部族、另一法案、另一场梦或另一事件，即使问题类型和情节模板相似，也只能标 X 或 I；不得把相似案例当成本题答案。",
            "如果 query 含‘离开后/随后/当时/那里/这次/该事件’等指代表达，先从候选池识别它指向的唯一局部事件与相邻时序。只有能绑定该局部锚点的候选可标 D/P/S；人物在其他时间离开其他地点后做的事一律标 X/I。",
            "歧义拒标规则：若仅凭 query 与候选池仍存在两个以上合理局部锚点，或无法唯一判断‘离开’指哪一次离开，则该题所有候选都只能标 X/I，D/P/S 必须为零。不要把所有可能答案都列为正例。",
            "D=direct_answer：单条候选同时绑定本题所问对象/事件，并明确给出所问的原因、方式、内容、结果等核心答案事实，或完整回答一个可独立子问；无需另一候选即可提取。",
            "P=partial_answer：单条候选仍须绑定本题同一对象/事件，并含所问关系的实质答案片段，但缺少限定、因果环节、做法或结果，不能独立形成完整答案。只出现同一人物却没有所问关系，不算 P。",
            "S=supporting_context：候选仍须能明确桥接到本题对象/局部事件；它不直接回答，但提供完成回答真正需要或高度有用的身份、时序或前提。泛化世界观、相似他人经历、仅同一场景不能标 S；如果 query 已经给出了这项背景，它通常也不是额外的 S。",
            "X=hard_context：同人物、同事件、同地点、同主题或前后相邻，看似相关但不能帮助回答所问内容；它是高价值难负例。",
            "I=irrelevant：无关。I 不写入 judgements，由校验器自动补全。",
            "针对‘为什么’的问题，D 必须给出适用于本题对象的明确原因/动机/因果桥；另一个对象因相似原因做相似事仍是 X。针对‘如何’的问题，D 必须明确给出本题施事者采取的做法/机制。针对‘哪些/什么’的问题，D 必须给出本题对象的所问细节。",
            "D/P/S 必须带 candidate.text 中连续逐字复制的一小段 4-120 字 quote；优先只复制最能证明标签的一至两句，不要复制整段，不要改写、补字或用省略号拼接。无法给出直接证明该标签的短逐字引文，就降低为 X 或 I。X 的 quote 必须为空。",
            "judgements 写出所有高置信 D/P/S，并最多写 16 个最有迷惑性的 X；不要凑数。一个候选只写一次并取其最高适用等级。",
            "数量是异常检测而非配额：D 最多 12 条、P 最多 20 条、S 最多 16 条、X 最多 16 条。若某档超过上限，说明对象/事件绑定仍不明确；应把不能精确绑定的候选降为 X/I，绝不能用相似经历填满正例。",
            "ranking 必须满足 D > P > S > X > I；同档仅把更明确、更完整、更少依赖省略语境者放前。不要为了覆盖答案的不同方面而改变单条候选的等级，也不要因另一候选已表达同一事实而降低本候选。",
            "不要输出 claims、参考答案、解释、0-100 分数或候选之外的事实。",
        ],
        "output_schema": {
            "protocol": PROTOCOL_VERSION,
            "query_decision": ["answerable", ""],
            "ranking": ["P002", "P001"],
            "judgements": [
                ["P002", "D", "从该候选正文逐字复制的答案引文"],
                ["P005", "P", "从该候选正文逐字复制的部分答案引文"],
                ["P010", "S", "从该候选正文逐字复制的必要背景引文"],
                ["P001", "X", ""],
            ],
        },
        "input": data,
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(instruction, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_json_content(content: str) -> dict[str, Any]:
    stripped = str(content or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped:
        raise ValueError("teacher returned empty content")
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("teacher content is not a JSON object")
    return parsed


def _normalize_label(raw_label: Any, candidate_id: str) -> str:
    label = LABEL_CODES.get(str(raw_label or "").strip().lower())
    if label not in LABEL_ORDER:
        raise ValueError(f"invalid label for {candidate_id}: {raw_label}")
    return label


def _explicit_labels(raw_judgements: Any) -> dict[str, str]:
    """Extract labels before full validation to check global tier ordering."""

    if not isinstance(raw_judgements, list):
        return {}
    labels: dict[str, str] = {}
    for raw in raw_judgements:
        if isinstance(raw, list) and len(raw) == 3:
            candidate_id = str(raw[0] or "")
            raw_label = raw[1]
        elif isinstance(raw, dict):
            candidate_id = str(raw.get("candidate_id") or "")
            raw_label = raw.get("label", raw.get("label_code"))
        else:
            continue
        label = LABEL_CODES.get(str(raw_label or "").strip().lower())
        if candidate_id and label in LABEL_ORDER:
            labels[candidate_id] = label
    return labels


def _quote_is_verbatim(quote: str, candidate_text: str) -> bool:
    normalized_quote = normalize_quote_text(quote)
    return (
        4 <= len(normalized_quote) <= 120
        and normalized_quote in normalize_quote_text(candidate_text)
    )


def _bounded_verbatim_quote(quote: str, candidate_text: str) -> str | None:
    """Return a bounded quote when a teacher copied a verbatim but long span.

    Long exact spans are mechanically shortened instead of spending multiple
    API retries on a formatting-only issue.  Non-verbatim text is never fixed.
    """

    normalized_quote = normalize_quote_text(quote)
    if len(normalized_quote) < 4 or normalized_quote not in normalize_quote_text(candidate_text):
        return None
    if len(normalized_quote) <= 120:
        return quote
    bounded: list[str] = []
    normalized_length = 0
    for character in quote:
        if not character.isspace():
            normalized_length += 1
        if normalized_length > 120:
            break
        bounded.append(character)
    shortened = "".join(bounded).strip()
    return shortened if _quote_is_verbatim(shortened, candidate_text) else None


def validate_judgement(record: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    validate_input_record(record, expected_candidates=None)
    if set(judgement) != {"protocol", "query_decision", "ranking", "judgements"}:
        raise ValueError(
            "output fields must be exactly protocol, query_decision, ranking, judgements; "
            f"got={sorted(judgement)}"
        )
    if str(judgement.get("protocol") or "") != PROTOCOL_VERSION:
        raise ValueError(f"protocol must be {PROTOCOL_VERSION}")

    stored_candidates = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in record.get("candidates") or []
    }
    alias_to_stored_id = prompt_candidate_aliases(record)
    candidates = {
        alias: stored_candidates[stored_id]
        for alias, stored_id in alias_to_stored_id.items()
    }
    expected_ids = list(candidates)
    expected_set = set(expected_ids)

    raw_ranking = judgement.get("ranking")
    if not isinstance(raw_ranking, list):
        raise ValueError("ranking is not a list")
    ranking_ids = [str(item or "") for item in raw_ranking]
    if len(ranking_ids) != len(expected_ids):
        raise ValueError(f"ranking length is {len(ranking_ids)}, expected {len(expected_ids)}")
    if len(ranking_ids) != len(set(ranking_ids)):
        raise ValueError("ranking contains duplicate candidate IDs")
    if set(ranking_ids) != expected_set:
        raise ValueError(
            f"ranking mismatch: missing={sorted(expected_set-set(ranking_ids))[:8]} "
            f"unknown={sorted(set(ranking_ids)-expected_set)[:8]}"
        )

    raw_query_decision = judgement.get("query_decision")
    if not isinstance(raw_query_decision, list) or len(raw_query_decision) != 2:
        raise ValueError("query_decision must be [answerable|ambiguous, short_reason]")
    query_status = str(raw_query_decision[0] or "")
    query_reason = str(raw_query_decision[1] or "").strip()
    if query_status not in {"answerable", "ambiguous"}:
        raise ValueError(f"invalid query_decision status: {query_status}")
    if query_status == "ambiguous" and not query_reason:
        raise ValueError("ambiguous query_decision requires a short reason")

    raw_judgements = judgement.get("judgements")
    if not isinstance(raw_judgements, list):
        raise ValueError("judgements is not a list")
    details: dict[str, dict[str, str]] = {}
    hard_context_count = 0
    label_counts = {label: 0 for label in MAX_LABEL_COUNTS}
    for raw in raw_judgements:
        if isinstance(raw, list) and len(raw) == 3:
            candidate_id = str(raw[0] or "")
            label = _normalize_label(raw[1], candidate_id)
            quote = str(raw[2] or "").strip()
        elif isinstance(raw, dict):
            if not set(raw) <= {"candidate_id", "label", "label_code", "quote", "support_quote"}:
                raise ValueError(f"unexpected judgement fields: {sorted(raw)}")
            candidate_id = str(raw.get("candidate_id") or "")
            label = _normalize_label(raw.get("label", raw.get("label_code")), candidate_id)
            quote = str(raw.get("quote", raw.get("support_quote")) or "").strip()
        else:
            raise ValueError("judgement must be [candidate_id,label,quote] or an object")
        if candidate_id not in expected_set:
            raise ValueError(f"unknown judged candidate ID: {candidate_id}")
        if candidate_id in details:
            raise ValueError(f"duplicate judged candidate ID: {candidate_id}")
        if label == "irrelevant":
            raise ValueError(f"irrelevant candidate must be omitted: {candidate_id}")
        if label in {"direct_answer", "partial_answer", "supporting_context"}:
            quote = _bounded_verbatim_quote(
                quote, str(candidates[candidate_id].get("text") or "")
            ) or ""
            if not quote:
                raise ValueError(f"quote is not a 4-120 char verbatim span: {candidate_id}")
        elif label == "hard_context":
            hard_context_count += 1
            if quote:
                raise ValueError(f"hard_context quote must be empty: {candidate_id}")
        label_counts[label] += 1
        details[candidate_id] = {"label": label, "support_quote": quote}
    over_limit = {
        label: (count, MAX_LABEL_COUNTS[label])
        for label, count in label_counts.items()
        if count > MAX_LABEL_COUNTS[label]
    }
    if over_limit:
        raise ValueError(f"too many positive/hard judgements: {over_limit}")
    if query_status == "ambiguous" and any(
        item["label"] in {"direct_answer", "partial_answer", "supporting_context"}
        for item in details.values()
    ):
        raise ValueError("ambiguous query must not have D/P/S judgements")

    # Omitted candidates are I, but teachers often list them interspersed in
    # the full ranking.  Validate tiers using explicit judgements first, then
    # normalize all omitted I candidates to the ranking tail deterministically.
    explicit_labels = _explicit_labels(raw_judgements)
    prior_order = len(LABEL_ORDER)
    for rank, alias in enumerate(ranking_ids, start=1):
        if alias not in explicit_labels:
            continue
        order = LABEL_ORDER[explicit_labels[alias]]
        if order > prior_order:
            raise ValueError(f"ranking violates D>P>S>X among judged items at rank {rank}: {alias}")
        prior_order = order
    normalized_aliases = [alias for alias in ranking_ids if alias in details]
    normalized_aliases.extend(alias for alias in ranking_ids if alias not in details)
    normalized_ranking: list[dict[str, Any]] = []
    for rank, alias in enumerate(normalized_aliases, start=1):
        detail = details.get(alias, {"label": "irrelevant", "support_quote": ""})
        normalized_ranking.append(
            {
                "candidate_id": alias_to_stored_id[alias],
                "rank": rank,
                **detail,
            }
        )
    return {
        "query_decision": {"status": query_status, "reason": query_reason},
        "ranking": normalized_ranking,
    }


def build_result_record(
    record: dict[str, Any],
    *,
    judgement: dict[str, Any],
    provider: str,
    model: str,
    usage: dict[str, Any],
    finish_reason: str,
) -> dict[str, Any]:
    cleaned = clean_record(record)
    cleaned.update(
        {
            "teacher_protocol": PROTOCOL_VERSION,
            "teacher_prompt_version": PROMPT_VERSION,
            "teacher": judgement,
            "teacher_provider": provider,
            "teacher_model": model,
            "finish_reason": finish_reason,
            "usage": usage,
        }
    )
    return cleaned
