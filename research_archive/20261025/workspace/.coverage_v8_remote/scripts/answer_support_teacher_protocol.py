#!/usr/bin/env python3
"""Shared, gold-free protocol for ASA full-candidate reranker teachers.

The reference answer defines the retrieval target, but it is not evidence.  A
teacher must decide whether each candidate passage independently supports an
atomic claim in that answer.  Gold evidence IDs, deterministic relevance
labels, and retrieval scores are deliberately excluded from the prompt.

The model sees every candidate, returns a complete candidate-ID ranking, and
emits detailed judgements only for positives and useful hard negatives.  The
validator expands omitted candidates to ``irrelevant`` and requires verbatim
quotes for every direct/partial/contradiction judgement.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

PROTOCOL_VERSION = "answer_support_v2"
PROMPT_VERSION = "answer_support_v2.2_pointwise"

ALLOWED_LABELS = {
    "direct",
    "partial",
    "hard_context",
    "contradiction",
    "irrelevant",
}
LABEL_CODES = {
    "d": "direct",
    "direct": "direct",
    "s": "direct",
    "support": "direct",
    "p": "partial",
    "partial": "partial",
    "x": "hard_context",
    "context": "hard_context",
    "hard_context": "hard_context",
    "c": "contradiction",
    "contradiction": "contradiction",
    "i": "irrelevant",
    "irrelevant": "irrelevant",
}

WHITESPACE_RE = re.compile(r"\s+")
CLAIM_ID_RE = re.compile(r"Q[1-9][0-9]*\Z")
FORBIDDEN_RECORD_FIELDS = {
    "gold_evidence",
    "gold_evidence_ids",
    "coverable_evidence_ids",
    "answer_evidence",
}
FORBIDDEN_CANDIDATE_FIELDS = {"covered_evidence_ids", "relevance", "label"}


def normalize_text(text: Any) -> str:
    return WHITESPACE_RE.sub("", str(text or ""))


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def record_key(record: dict[str, Any]) -> str:
    return f"{record.get('source_name', '')}\n{record.get('query', '')}"


def clean_candidate(candidate: dict[str, Any], *, candidate_id: str | None = None) -> dict[str, Any]:
    """Keep only fields needed for a teacher decision.

    Retrieval ranks, scores, document IDs, and legacy labels stay in the
    original/evaluation files, not in the teacher-safe pipeline.  This makes
    accidental prompt leakage much harder and removes position anchoring.
    """

    return {
        "candidate_id": str(candidate_id or candidate.get("candidate_id") or ""),
        "text": str(candidate.get("text") or ""),
        "kind": str(candidate.get("kind") or "document"),
    }


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a teacher-safe record with no gold or deterministic labels."""

    cleaned = {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "source_name": str(record.get("source_name") or ""),
        "story_family": str(record.get("story_family") or ""),
        "answer": str(record.get("answer") or ""),
        "candidates": [clean_candidate(candidate) for candidate in record.get("candidates") or []],
    }
    return cleaned


def leaked_supervision_fields(record: dict[str, Any]) -> list[str]:
    leaked = [field for field in FORBIDDEN_RECORD_FIELDS if field in record]
    for index, candidate in enumerate(record.get("candidates") or [], start=1):
        leaked.extend(
            f"candidates[{index}].{field}"
            for field in FORBIDDEN_CANDIDATE_FIELDS
            if field in candidate
        )
    return sorted(leaked)


def validate_input_record(record: dict[str, Any], expected_candidates: int | None) -> None:
    query = str(record.get("query") or "").strip()
    answer = str(record.get("answer") or "").strip()
    candidates = record.get("candidates")
    leaked = leaked_supervision_fields(record)
    if leaked:
        raise ValueError(
            "teacher input contains gold or legacy supervision fields: "
            f"{leaked[:12]}"
        )
    if not query:
        raise ValueError("input record has an empty query")
    if not answer:
        raise ValueError("input record has an empty reference answer")
    if not isinstance(candidates, list):
        raise ValueError("input record has no candidates list")
    if expected_candidates is not None and len(candidates) != expected_candidates:
        raise ValueError(
            f"input candidate count is {len(candidates)}, expected {expected_candidates}"
        )
    ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if any(not candidate_id for candidate_id in ids):
        raise ValueError("input contains an empty candidate_id")
    if len(set(ids)) != len(ids):
        raise ValueError("input contains duplicate candidate_id")
    if any(not str(candidate.get("text") or "").strip() for candidate in candidates):
        raise ValueError("input contains empty candidate text")


def _prompt_candidate_aliases(record: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    source_candidates = list(record.get("candidates") or [])
    seed_material = f"{record.get('source_name', '')}\n{record.get('query', '')}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    random.Random(seed).shuffle(source_candidates)
    candidates: list[dict[str, str]] = []
    alias_to_candidate_id: dict[str, str] = {}
    for index, candidate in enumerate(source_candidates, start=1):
        alias = f"P{index:03d}"
        alias_to_candidate_id[alias] = str(candidate.get("candidate_id") or "")
        candidates.append(
            {
                "candidate_id": alias,
                "kind": str(candidate.get("kind") or "document"),
                "text": str(candidate.get("text") or ""),
            }
        )
    return candidates, alias_to_candidate_id


def prompt_candidate_aliases(record: dict[str, Any]) -> dict[str, str]:
    """Return prompt alias -> stored candidate ID for output validation."""

    _, aliases = _prompt_candidate_aliases(record)
    return aliases


def teacher_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Build the exact payload shown to the teacher.

    Deliberately omitted: gold evidence, old labels/relevance, retrieval rank,
    fusion/reranker scores, and document IDs that encode source ordering.
    """

    candidates, _ = _prompt_candidate_aliases(record)
    return {
        "query": str(record.get("query") or ""),
        "query_type": str(record.get("query_type") or "unknown"),
        "reference_answer": str(record.get("answer") or ""),
        "candidates": candidates,
    }


def build_messages(
    record: dict[str, Any], expected_candidates: int | None = None
) -> list[dict[str, str]]:
    data = teacher_payload(record)
    candidate_count = len(data["candidates"])
    expected_count = expected_candidates if expected_candidates is not None else candidate_count
    system = (
        "你是 ASA 固定候选池的检索重排标注器。参考答案只定义要检索的目标，不是证据。"
        "只能依据单条候选正文判断它能否支持参考答案中的原子主张；不得使用外部知识，"
        "不得把同人物、同事件或同主题误当成事实支持。你不生成、不扩展候选，也不推断"
        "召回来源；候选输入顺序已随机化且没有意义。"
        "Return exactly one valid JSON object and no prose outside JSON."
    )
    instruction = {
        "protocol": PROTOCOL_VERSION,
        "task": [
            "先把 reference_answer 拆成最少数量、互相独立且不引入新事实的原子主张，依次命名 Q1、Q2……。",
            f"必须阅读全部 {candidate_count} 条候选，并在 ranking 中返回 exactly {expected_count} 个 candidate_id；每个 ID 必须且只能出现一次。",
            "逐条独立判断候选，不得把多条候选拼接后再反向给其中一条加标签。",
            "D=direct：候选正文单独足以直接确立至少一个完整主张。",
            "P=partial：候选正文只确立某主张的实质性一部分；仅同主题、同人物、同事件、前后相邻或可由外部知识联想到，必须标 X 而不是 P。",
            "X=hard_context：与问题高度相关、容易迷惑检索器，但没有确立任何答案主张。",
            "C=contradiction：候选正文明确否定或冲突于某个答案主张。明显无关候选不要写入 judgements。",
            "每个 D/P/C 必须提供 candidate.text 中连续复制的 4-80 字 support_quote；不能复制出直接支撑判断的原文，就不能标 D/P/C。",
            "D/P 必须列出其真正支持的 claim_id；X 的 claim_ids 必须为空；C 列出被它冲突的 claim_id。",
            "judgements 应包含所有高置信 D/P/C，并最多保留 12 个最有迷惑性的 X；不要为了凑数降低标准。",
            "ranking 只能按每条候选自身对 query/答案目标的独立支持强度排序：完整且明确的直接证据优先，其次是实质性部分证据，再后是 contradiction/hard context 和无关项。",
            "不要为了让不同主张轮流出现而改变排序；不要因为另一条候选已覆盖同一主张，就降低当前候选的名次。下游 BGE 会把每个 (query,candidate) 独立打分。",
            "同级候选优先正文更明确、更完整、更少依赖省略语境者；不要输出 0-100 主观连续分数。",
            "uncovered_claim_ids 必须精确列出没有任何 D 或 P 候选支持的主张。",
        ],
        "output_schema": {
            "protocol": PROTOCOL_VERSION,
            "claims": [["Q1", "只来自参考答案的原子主张"]],
            "ranking": ["P002", "P001"],
            "judgements": [
                ["P002", "D", ["Q1"], "从候选正文逐字复制的短引文"],
                ["P001", "X", [], ""],
            ],
            "uncovered_claim_ids": [],
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


def build_claim_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build the small, candidate-independent claim decomposition request."""

    system = (
        "你是参考答案结构化工具。只重述参考答案中已经明确包含的事实，"
        "不得添加外部知识或候选信息。Return exactly one valid JSON object."
    )
    instruction = {
        "protocol": PROTOCOL_VERSION,
        "task": [
            "把 reference_answer 拆成最少数量、互相独立、适合逐条检索验证的原子主张。",
            "不得省略限定条件、因果关系、否定、不确定性或答案中的拒答边界。",
            "主张依次命名 Q1、Q2……；不要输出解释。",
        ],
        "output_schema": {"protocol": PROTOCOL_VERSION, "claims": [["Q1", "原子主张"]]},
        "input": {
            "query": str(record.get("query") or ""),
            "query_type": str(record.get("query_type") or "unknown"),
            "reference_answer": str(record.get("answer") or ""),
        },
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(instruction, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def validate_claim_judgement(judgement: dict[str, Any]) -> list[dict[str, str]]:
    if str(judgement.get("protocol") or "") != PROTOCOL_VERSION:
        raise ValueError(f"claim protocol must be {PROTOCOL_VERSION}")
    return _normalize_claims(judgement.get("claims"))


def build_ranking_messages(
    record: dict[str, Any],
    claims: list[dict[str, str]],
    expected_candidates: int | None = None,
) -> list[dict[str, str]]:
    """Build candidate ranking request using a frozen claim decomposition."""

    data = teacher_payload(record)
    # Phase B uses the frozen claims as the sole answer-target representation.
    # Omitting the prose answer avoids asking the ranking model to reconcile two
    # slightly different target formulations and saves prompt tokens.
    data.pop("reference_answer", None)
    candidate_count = len(data["candidates"])
    expected_count = expected_candidates if expected_candidates is not None else candidate_count
    system = (
        "你是 ASA 固定候选池的检索重排标注器。给定原子主张只定义检索目标，不是证据。"
        "只能依据单条候选正文判断支持关系；不得使用外部知识，不得把同人物、同事件或"
        "同主题误当成事实支持。你不生成、不扩展候选，也不推断召回来源；候选输入顺序"
        "及 P 编号均为随机别名，没有排名意义。"
        "Return exactly one valid JSON object and no prose outside JSON."
    )
    instruction = {
        "protocol": PROTOCOL_VERSION,
        "claims": [[claim["claim_id"], claim["text"]] for claim in claims],
        "task": [
            f"必须阅读全部 {candidate_count} 条候选，并在 ranking 中返回 exactly {expected_count} 个 candidate_id；每个 ID 必须且只能出现一次。",
            "逐条独立判断候选，不得把多条候选拼接后再反向给其中一条加标签。",
            "D=direct：候选正文单独足以直接确立至少一个完整主张。",
            "P=partial：候选正文只确立某主张的实质性一部分；仅同主题、同人物、同事件、前后相邻或可由外部知识联想到，必须标 X。",
            "X=hard_context：与问题高度相关、容易迷惑检索器，但没有确立答案主张。",
            "C=contradiction：候选正文明确否定或冲突于某个答案主张。明显无关项不要写入 judgements。",
            "每个 D/P/C 必须提供 candidate.text 中连续复制的 4-80 字 support_quote；不能提供逐字引文，就不能标 D/P/C。",
            "D/P 必须列出支持的 claim_id；X 的 claim_ids 和 quote 必须为空；C 列出冲突的 claim_id。",
            "judgements 包含所有高置信 D/P/C，并最多保留 12 个最有迷惑性的 X。",
            "ranking 只能按每条候选自身对 query/答案目标的独立支持强度排序：完整且明确的直接证据优先，其次是实质性部分证据，再后是 contradiction/hard context 和无关项。",
            "不要为了让不同主张轮流出现而改变排序；不要因为另一条候选已覆盖同一主张，就降低当前候选的名次。下游 BGE 会把每个 (query,candidate) 独立打分。",
            "同级候选优先正文更明确、更完整、更少依赖省略语境者；不要输出主观连续分数。",
            "uncovered_claim_ids 必须精确列出没有任何 D/P 候选支持的主张。",
        ],
        "output_schema": {
            "protocol": PROTOCOL_VERSION,
            "ranking": ["P002", "P001"],
            "judgements": [
                ["P002", "D", ["Q1"], "从候选正文逐字复制的短引文"],
                ["P001", "X", [], ""],
            ],
            "uncovered_claim_ids": [],
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


def validate_ranking_judgement(
    record: dict[str, Any],
    claims: list[dict[str, str]],
    judgement: dict[str, Any],
) -> dict[str, Any]:
    combined = dict(judgement)
    combined["protocol"] = PROTOCOL_VERSION
    combined["claims"] = [[claim["claim_id"], claim["text"]] for claim in claims]
    return validate_judgement(record, combined)


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


def _normalize_claims(raw_claims: Any) -> list[dict[str, str]]:
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("claims must be a non-empty list")
    claims: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_claims:
        if isinstance(raw, list) and len(raw) >= 2:
            claim_id, text = str(raw[0] or ""), str(raw[1] or "").strip()
        elif isinstance(raw, dict):
            claim_id = str(raw.get("claim_id") or "")
            text = str(raw.get("text") or "").strip()
        else:
            raise ValueError("claim is neither [claim_id,text] nor an object")
        if not CLAIM_ID_RE.fullmatch(claim_id):
            raise ValueError(f"invalid claim_id: {claim_id}")
        if claim_id in seen:
            raise ValueError(f"duplicate claim_id: {claim_id}")
        if not text:
            raise ValueError(f"empty claim text: {claim_id}")
        seen.add(claim_id)
        claims.append({"claim_id": claim_id, "text": text})
    expected = [f"Q{index}" for index in range(1, len(claims) + 1)]
    actual = [claim["claim_id"] for claim in claims]
    if actual != expected:
        raise ValueError(f"claim IDs must be sequential: expected={expected} actual={actual}")
    return claims


def _normalize_label(raw_label: Any, candidate_id: str) -> str:
    label = LABEL_CODES.get(str(raw_label or "").strip().lower())
    if label not in ALLOWED_LABELS:
        raise ValueError(f"invalid label for {candidate_id}: {raw_label}")
    return label


def _quote_is_verbatim(quote: str, candidate_text: str) -> bool:
    normalized_quote = normalize_text(quote)
    return len(normalized_quote) >= 4 and normalized_quote in normalize_text(candidate_text)


def validate_judgement(record: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    validate_input_record(record, expected_candidates=None)
    if str(judgement.get("protocol") or "") != PROTOCOL_VERSION:
        raise ValueError(f"protocol must be {PROTOCOL_VERSION}")

    claims = _normalize_claims(judgement.get("claims"))
    claim_ids = {claim["claim_id"] for claim in claims}
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
        raise ValueError(
            f"ranking length is {len(ranking_ids)}, expected {len(expected_ids)}"
        )
    if len(set(ranking_ids)) != len(ranking_ids):
        raise ValueError("ranking contains duplicate candidate IDs")
    if set(ranking_ids) != expected_set:
        raise ValueError(
            f"ranking candidate mismatch: missing={sorted(expected_set-set(ranking_ids))[:8]} "
            f"unknown={sorted(set(ranking_ids)-expected_set)[:8]}"
        )

    raw_judgements = judgement.get("judgements")
    if not isinstance(raw_judgements, list):
        raise ValueError("judgements is not a list")
    details: dict[str, dict[str, Any]] = {}
    covered_claim_ids: set[str] = set()
    hard_context_count = 0
    for raw in raw_judgements:
        if isinstance(raw, list) and len(raw) >= 4:
            candidate_id = str(raw[0] or "")
            label = _normalize_label(raw[1], candidate_id)
            raw_claim_ids = raw[2]
            quote = str(raw[3] or "").strip()
        elif isinstance(raw, dict):
            candidate_id = str(raw.get("candidate_id") or "")
            label = _normalize_label(raw.get("label", raw.get("label_code")), candidate_id)
            raw_claim_ids = raw.get("claim_ids", raw.get("supported_claim_ids"))
            quote = str(raw.get("support_quote") or "").strip()
        else:
            raise ValueError("judgement is neither a compact array nor an object")
        if candidate_id not in expected_set:
            raise ValueError(f"unknown judged candidate_id: {candidate_id}")
        if candidate_id in details:
            raise ValueError(f"duplicate judged candidate_id: {candidate_id}")
        if label == "irrelevant":
            raise ValueError(f"irrelevant candidate must be omitted from judgements: {candidate_id}")
        if not isinstance(raw_claim_ids, list):
            raise ValueError(f"claim_ids for {candidate_id} is not a list")
        normalized_claim_ids = list(dict.fromkeys(str(item) for item in raw_claim_ids))
        unknown_claims = sorted(set(normalized_claim_ids) - claim_ids)
        if unknown_claims:
            raise ValueError(f"unknown claim_ids for {candidate_id}: {unknown_claims}")

        if label in {"direct", "partial", "contradiction"}:
            if not normalized_claim_ids:
                raise ValueError(f"{label} candidate has no claim_ids: {candidate_id}")
            if not _quote_is_verbatim(quote, str(candidates[candidate_id].get("text") or "")):
                raise ValueError(f"support_quote is not verbatim candidate text: {candidate_id}")
        elif label == "hard_context":
            hard_context_count += 1
            if normalized_claim_ids:
                raise ValueError(f"hard_context candidate must have empty claim_ids: {candidate_id}")
            if quote:
                raise ValueError(f"hard_context candidate must have an empty quote: {candidate_id}")

        if label in {"direct", "partial"}:
            covered_claim_ids.update(normalized_claim_ids)
        details[candidate_id] = {
            "label": label,
            "supported_claim_ids": normalized_claim_ids,
            "support_quote": quote,
        }

    if hard_context_count > 12:
        raise ValueError(f"too many hard_context judgements: {hard_context_count} > 12")

    raw_uncovered = judgement.get("uncovered_claim_ids")
    if not isinstance(raw_uncovered, list):
        raise ValueError("uncovered_claim_ids is not a list")
    uncovered = list(dict.fromkeys(str(item) for item in raw_uncovered))
    unknown_uncovered = sorted(set(uncovered) - claim_ids)
    if unknown_uncovered:
        raise ValueError(f"unknown uncovered_claim_ids: {unknown_uncovered}")
    expected_uncovered = claim_ids - covered_claim_ids
    if set(uncovered) != expected_uncovered:
        raise ValueError(
            "uncovered_claim_ids inconsistent with D/P judgements: "
            f"expected={sorted(expected_uncovered)} actual={sorted(uncovered)}"
        )

    normalized_ranking: list[dict[str, Any]] = []
    for rank, candidate_alias in enumerate(ranking_ids, start=1):
        detail = details.get(
            candidate_alias,
            {
                "label": "irrelevant",
                "supported_claim_ids": [],
                "support_quote": "",
            },
        )
        normalized_ranking.append(
            {
                "candidate_id": alias_to_stored_id[candidate_alias],
                "rank": rank,
                **detail,
            }
        )
    return {
        "claims": claims,
        "ranking": normalized_ranking,
        "uncovered_claim_ids": uncovered,
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
