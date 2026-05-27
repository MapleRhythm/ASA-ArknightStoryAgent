#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import random
import re
import sys
from threading import Lock
import time
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"
if TRAIN_PYTHON_OVERLAY_DIR.exists():
    sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
if TRAIN_OVERRIDE_DIR.exists():
    sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import EMBEDDING_MODEL_DIR, INDEX_ROOT, QueryConfig, RERANKER_MODEL_DIR  # noqa: E402
from goldenglow.data.sft_teacher import TeacherApiConfig  # noqa: E402
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    HYPOTHESIS_INTENTS,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    ConclusionResult,
    HypothesisDocument,
    build_retrieval_query,
    classify_retrieval_query_mode,
    detect_intent,
    merge_hypotheses,
    merge_ranked_hits,
    normalize_conclusion_payload,
    normalize_hypothesis_payload,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.build_short_prompt_sft_dataset import compact_text  # noqa: E402
from scripts.evaluate_retrieval_recall import (  # noqa: E402
    candidate_hit_source,
    extract_gold_text,
    load_listwise,
    parse_mode_weights,
)
from scripts.generate_online_teacher_chain_sft import (  # noqa: E402
    append_jsonl,
    as_conclusion_payload,
    as_follow_up_payload,
    as_initial_payload,
    build_student_conclusion_prompt,
    build_student_follow_up_prompt,
    build_student_hypothesis_prompt,
    build_teacher_system,
    call_teacher_json,
    compact_json,
    dedupe_keep_order,
    doc_is_story_seed,
    export_splits,
    is_bad_question,
    is_bad_retrieval_term,
    load_documents,
    load_json,
    load_jsonl,
    log_progress,
    normalize_teacher_entity_or_keyword,
    parse_teacher_payload,
    render_evidence_brief,
    rerank_hits,
    resolve_path,
    sanitize_dialogue_context,
    sanitize_hypothesis_document,
    sanitize_teacher_alias_fields,
    stable_key,
    write_json,
)


DEFAULT_LISTWISE = PROJECT_ROOT / "data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_planned_retrieval_chain_v1"
DEFAULT_MINIRAG_INDEX = PROJECT_ROOT / "indexes/arknights_story_minirag_v3/graph.json"
JSONL_WRITE_LOCK = Lock()
LOG_LOCK = Lock()

QUERY_TYPES = {"fact", "relation", "causality", "reasoning", "reveal", "mystery", "answerability"}
QUERY_TYPE_ALIASES = {
    "plot_fact": "fact",
    "plot_reasoning": "reasoning",
    "timeline": "fact",
    "compare": "reasoning",
    "comparison": "reasoning",
}
WEAK_TERMS = {
    "原因",
    "目的",
    "动机",
    "关系",
    "信息",
    "影响",
    "情况",
    "剧情",
    "分析",
    "背景",
    "线索",
    "真相",
    "过程",
    "细节",
    "解释",
    "暗示",
    "问题",
}
GENERIC_MISSING_SLOTS = {
    "具体背景",
    "具体细节",
    "相关信息",
    "关键证据",
    "核心原因",
    "深层含义",
    "事件背景",
    "身份细节",
    "具体人物关系",
    "缺少能直接连接问题与答案的桥接证据",
    "需要补充关键实体或事件上下文",
}

INTENT_BY_QUERY_TYPE = {
    "relation": "character_relation",
    "causality": "plot_reasoning",
    "reasoning": "plot_reasoning",
    "reveal": "plot_reasoning",
    "mystery": "plot_reasoning",
    "answerability": "plot_reasoning",
    "fact": "plot_fact",
}

INTENT_ALIASES = {
    "事实": "plot_fact",
    "事实问答": "plot_fact",
    "剧情事实": "plot_fact",
    "剧情推理": "plot_reasoning",
    "原因": "plot_reasoning",
    "原因动机": "plot_reasoning",
    "过程解释": "plot_reasoning",
    "关系": "character_relation",
    "身份关系": "character_relation",
    "人物关系": "character_relation",
    "事件总结": "event_summary",
    "剧情总结": "event_summary",
    "时间线": "timeline",
    "对比": "compare",
    "比较": "compare",
}


def normalize_query_type(value: Any) -> str:
    query_type = str(value or "").strip()
    query_type = QUERY_TYPE_ALIASES.get(query_type, query_type)
    return query_type if query_type in QUERY_TYPES else "reasoning"


def coerce_intent(
    value: Any,
    *,
    question: str,
    query_type: str,
    expected_answer_type: str = "",
) -> str:
    raw = str(value or "").strip()
    if raw in HYPOTHESIS_INTENTS:
        return raw
    if raw in INTENT_ALIASES:
        return INTENT_ALIASES[raw]
    combined = question + " " + raw + " " + expected_answer_type
    if any(token in combined for token in ("关系", "身份", "父亲", "母亲", "同伴", "阵营")):
        return "character_relation"
    if any(token in combined for token in ("时间线", "先后", "之前", "之后", "何时", "什么时候")):
        return "timeline"
    if any(token in combined for token in ("对比", "比较", "区别", "不同")):
        return "compare"
    if any(token in combined for token in ("总结", "概括", "讲了什么", "发生了什么")):
        return "event_summary"
    if query_type in INTENT_BY_QUERY_TYPE:
        return INTENT_BY_QUERY_TYPE[query_type]
    detected, _ = detect_intent(question)
    return detected if detected in HYPOTHESIS_INTENTS else "plot_reasoning"


def clean_terms(items: Any, *, field: str, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    output: list[str] = []
    for item in items:
        if not isinstance(item, (str, int, float)):
            continue
        raw = normalize_teacher_entity_or_keyword(str(item), field=field)
        parts = re.split(r"\s+", raw) if field == "keywords" and re.search(r"\s+", raw) else [raw]
        for part in parts:
            term = normalize_teacher_entity_or_keyword(str(part), field=field).strip()
            if not term or is_bad_retrieval_term(term, field=field):
                continue
            output.append(term)
    return dedupe_keep_order(output)[:limit]


def normalize_missing_slots(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[;；、,\n]+", value)
    elif isinstance(value, list):
        raw_items = [str(item) for item in value if isinstance(item, (str, int, float))]
    else:
        raw_items = []
    slots: list[str] = []
    for raw in raw_items:
        slot = re.sub(r"\s+", "", str(raw or "")).strip("。；;，,、：:")
        if len(slot) < 3 or slot in GENERIC_MISSING_SLOTS:
            continue
        if slot in WEAK_TERMS:
            continue
        slots.append(slot)
    return dedupe_keep_order(slots)[:5]


def build_seed_gold_text(seed: dict[str, Any]) -> str:
    gold_text = str(seed.get("gold_text") or "").strip()
    if gold_text:
        return gold_text
    seed_docs = seed.get("seed_docs") if isinstance(seed.get("seed_docs"), list) else []
    parts: list[str] = []
    for index, doc in enumerate(seed_docs[:8], start=1):
        text = compact_text(str(doc.get("clean_text") or ""), 1200)
        if text:
            parts.append(f"[D{index}] {text}")
    return "\n".join(parts).strip()


def validate_hypothesis_payload(
    payload: dict[str, Any],
    *,
    question: str,
    dialogue_context: str,
    current_intent: str | None = None,
) -> HypothesisDocument:
    payload = sanitize_teacher_alias_fields(payload)
    payload = dict(payload)
    if current_intent is not None:
        payload.pop("intent", None)
    else:
        query_type = normalize_query_type(payload.get("query_type"))
        payload["intent"] = coerce_intent(
            payload.get("intent"),
            question=question,
            query_type=query_type,
            expected_answer_type=str(payload.get("expected_answer_type") or ""),
        )
    return sanitize_hypothesis_document(
        normalize_hypothesis_payload(
            payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
        )
    )


def make_question_item(
    *,
    question: str,
    query_type: str,
    source: str,
    seed_key: str,
    dialogue_context: str = "",
    difficulty: str = "hard",
) -> dict[str, str]:
    key = stable_key("teacher_planned_chain", question, dialogue_context, seed_key)
    return {
        "question_key": key,
        "question": question,
        "dialogue_context": dialogue_context,
        "source_split": source,
        "source_task_type": "teacher_planned_retrieval_chain",
        "source_seed_key": seed_key,
        "source_seed_index": "0",
        "query_type": query_type,
        "entities": "",
        "difficulty": difficulty,
    }


def build_listwise_seed(record: dict[str, Any], index: int, *, source: str) -> dict[str, Any] | None:
    question = str(record.get("query") or "").strip()
    gold_text = extract_gold_text(record)
    if not question or not gold_text or is_bad_question(question):
        return None
    return {
        "seed_key": stable_key(source, str(index), question),
        "source": source,
        "question": question,
        "query_type": normalize_query_type(record.get("query_type")),
        "answer": str(record.get("answer") or ""),
        "answer_focus": str(record.get("answer_focus") or ""),
        "gold_text": gold_text,
        "seed_docs": [],
        "raw": record,
    }


def build_eval_seeds(path: Path, *, max_items: int, include_hits: bool) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return []
    seeds: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        cumulative = record.get("cumulative_hit") or {}
        hit20 = bool(cumulative.get("20") or cumulative.get(20))
        hit50 = bool(cumulative.get("50") or cumulative.get(50))
        first_rank = record.get("first_hit_rank")
        hard = (not hit50) or (hit50 and not hit20) or (isinstance(first_rank, int) and first_rank > 20)
        if not hard and not include_hits:
            continue
        question = str(record.get("query") or "").strip()
        gold_text = str(record.get("gold_excerpt") or "").strip()
        if not question or not gold_text or is_bad_question(question):
            continue
        seeds.append(
            {
                "seed_key": stable_key("eval", str(index), question),
                "source": "eval_hard",
                "question": question,
                "query_type": normalize_query_type(record.get("query_type")),
                "answer": "",
                "answer_focus": "",
                "gold_text": gold_text,
                "seed_docs": [],
                "raw": {
                    "rounds_run": record.get("rounds_run"),
                    "first_hit_rank": record.get("first_hit_rank"),
                    "cumulative_hit": cumulative,
                },
            }
        )
        if len(seeds) >= max_items:
            break
    return seeds


def render_seed_docs(docs: list[dict[str, Any]], *, max_docs: int = 5, max_chars: int = 700) -> str:
    lines: list[str] = []
    for index, doc in enumerate(docs[:max_docs], start=1):
        lines.append(
            "\n".join(
                [
                    f"[D{index}] doc_id: {doc.get('id')}",
                    f"activity: {doc.get('activity_name') or ''}",
                    f"story: {doc.get('story_name') or ''}",
                    f"stage: {doc.get('stage_code') or ''} {doc.get('avg_tag') or ''}",
                    "text: " + compact_text(str(doc.get("clean_text") or ""), max_chars),
                ]
            )
        )
    return "\n\n".join(lines)


def build_document_seeds(documents_path: Path, *, max_items: int, docs_per_seed: int, seed: int) -> list[dict[str, Any]]:
    docs = [doc for doc in load_documents(documents_path) if doc_is_story_seed(doc)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        key = str(doc.get("story_id") or doc.get("source_path") or "")
        if key:
            grouped[key].append(doc)
    groups: list[list[dict[str, Any]]] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: str(item.get("id") or ""))
        if len(ordered) < 2:
            continue
        for start in range(0, len(ordered), docs_per_seed):
            chunk = ordered[start : start + docs_per_seed]
            if len(chunk) >= 2:
                groups.append(chunk)
    rng = random.Random(seed)
    rng.shuffle(groups)
    seeds: list[dict[str, Any]] = []
    for index, group in enumerate(groups[:max_items]):
        seed_key = stable_key("docs", *(str(doc.get("id") or "") for doc in group))
        seeds.append(
            {
                "seed_key": seed_key,
                "source": "document_seed",
                "question": "",
                "query_type": "reasoning",
                "answer": "",
                "answer_focus": "",
                "gold_text": "",
                "seed_docs": group,
                "raw": {"doc_ids": [doc.get("id") for doc in group]},
                "index": index,
            }
        )
    return seeds


def load_seed_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    if args.eval_misses:
        seeds.extend(build_eval_seeds(resolve_path(args.eval_misses), max_items=args.max_eval_seeds, include_hits=args.include_eval_hits))
    listwise_path = resolve_path(args.listwise)
    if args.max_listwise_seeds > 0 and listwise_path.exists():
        records = load_listwise(listwise_path)
        rng = random.Random(args.seed)
        indexed = list(enumerate(records))
        rng.shuffle(indexed)
        for index, record in indexed:
            item = build_listwise_seed(record, index, source="listwise")
            if item is None:
                continue
            seeds.append(item)
            if sum(1 for seed in seeds if seed["source"] == "listwise") >= args.max_listwise_seeds:
                break
    if args.max_document_seeds > 0:
        seeds.extend(
            build_document_seeds(
                resolve_path(args.documents),
                max_items=args.max_document_seeds,
                docs_per_seed=args.document_seed_doc_count,
                seed=args.seed,
            )
        )
    deduped: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        key = str(seed.get("question") or seed.get("seed_key"))
        deduped.setdefault(key, seed)
    output = list(deduped.values())
    random.Random(args.seed).shuffle(output)
    return output[: args.max_seeds]


def build_teacher_planned_chain_prompt(seed: dict[str, Any], *, max_rounds: int) -> str:
    source = str(seed.get("source") or "")
    question = str(seed.get("question") or "")
    gold_text = str(seed.get("gold_text") or "")
    answer = str(seed.get("answer") or "")
    answer_focus = str(seed.get("answer_focus") or "")
    seed_docs = seed.get("seed_docs") if isinstance(seed.get("seed_docs"), list) else []
    lines = [
        "任务: 生成 hard retrieval chain 训练候选 JSON。",
        "目标: 让学生模型学会在第一轮证据不足时，生成真正能改变召回方向的 follow_up_hypothesis。",
        "输出必须是单个合法 JSON 对象，不要 markdown。",
        "",
        f"source: {source}",
    ]
    if question:
        lines.append(f"原始问题: {question}")
    if answer:
        lines.append(f"参考答案摘要: {compact_text(answer, 260)}")
    if answer_focus:
        lines.append(f"参考答案焦点: {compact_text(answer_focus, 260)}")
    if gold_text:
        lines.append("gold_evidence_for_teacher_only:")
        lines.append(compact_text(gold_text, 1200))
    if seed_docs:
        lines.append("seed_story_docs:")
        lines.append(render_seed_docs(seed_docs, max_docs=6, max_chars=650))
    lines.extend(
        [
            "",
            "输出字段严格为:",
            "question, dialogue_context, query_type, expected_answer, initial_hypothesis, rounds",
            "",
            "字段说明:",
            "- question: 真实用户会问的中文剧情问题；不要提到片段/证据/检索/chunk/id。",
            "- dialogue_context: 只能填写真实多轮用户上下文；没有就填空字符串，禁止写答案、证据摘要或补充剧情。",
            "- query_type: fact/relation/causality/reasoning/reveal/mystery/answerability。",
            "- expected_answer: 简短参考答案；如果给了 gold_evidence_for_teacher_only 或 seed_story_docs，必须基于证据填写，除非证据确实无法回答。",
            "- initial_hypothesis: 字段为 question,intent,query_type,entities,keywords,expected_answer_type,dialogue_context。",
            "- initial_hypothesis.intent 只能从 plot_fact/plot_reasoning/timeline/character_relation/event_summary/compare/persona_chat/out_of_scope 中选择，禁止写中文短语。",
            f"- rounds: 数组，长度 1-{max_rounds - 1}；每项字段为 missing_slots, follow_up_hypothesis。",
            "- missing_slots: 必须是 2-5 个具体缺失槽位数组，并且要和 follow_up 新增桥接词对应；禁止只写“具体背景/关键证据/相关信息/事件背景”。",
            "- follow_up_hypothesis: 字段为 question,query_type,entities,keywords,expected_answer_type,dialogue_context。",
            "",
            "硬性规则:",
            "1. follow_up_hypothesis.question 必须等于 question，不要改写成追问。",
            "2. 每一轮 follow_up 必须新增 2-5 个具体桥接词：人物别名/本名、组织、地点、章节、事件名、物品、关键台词词组。",
            "3. 禁止只新增 原因/背景/真相/线索/关系/影响/情况/剧情 等泛词。",
            "4. 不要输出内部 alias 串，不要照抄所有别名；凯尔希相关只写自然称谓。",
            "5. 如果给了 gold_evidence_for_teacher_only，可用它判断该补哪些检索词，但不要在 question 里暴露 gold 或 doc_id。",
            "6. 如果 source 是 document_seed，请优先生成跨片段、因果、关系变化、真相揭示类问题。",
        ]
    )
    return "\n".join(lines)


def parse_planned_chain_payload(payload: dict[str, Any], *, seed: dict[str, Any], max_rounds: int) -> dict[str, Any]:
    question = str(payload.get("question") or seed.get("question") or "").strip()
    if len(question) < 6 or is_bad_question(question):
        raise ValueError("bad planned question")
    # Dialogue context is runtime user input. Teacher sees gold evidence and
    # otherwise tends to leak answer-only bridge terms into this field.
    dialogue_context = sanitize_dialogue_context(seed.get("dialogue_context"))
    query_type = normalize_query_type(payload.get("query_type") or seed.get("query_type"))
    expected_answer = str(payload.get("expected_answer") or seed.get("answer") or "").strip()
    if not expected_answer and not str(seed.get("gold_text") or "").strip() and not seed.get("seed_docs"):
        raise ValueError("missing expected_answer")
    initial_raw = payload.get("initial_hypothesis")
    if not isinstance(initial_raw, dict):
        raise ValueError("missing initial_hypothesis")
    initial = validate_hypothesis_payload(initial_raw, question=question, dialogue_context=dialogue_context)
    rounds_raw = payload.get("rounds")
    if not isinstance(rounds_raw, list) or not rounds_raw:
        raise ValueError("missing rounds")
    rounds: list[dict[str, Any]] = []
    current_intent = initial.intent
    for raw_round in rounds_raw[: max(1, max_rounds - 1)]:
        if not isinstance(raw_round, dict):
            continue
        missing_slots = normalize_missing_slots(raw_round.get("missing_slots"))
        if not missing_slots:
            continue
        follow_raw = raw_round.get("follow_up_hypothesis")
        if not isinstance(follow_raw, dict):
            continue
        follow = validate_hypothesis_payload(
            follow_raw,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current_intent,
        )
        rounds.append({"missing_slots": dedupe_keep_order(missing_slots)[:5], "follow_up_hypothesis": follow})
    if not rounds:
        raise ValueError("no valid follow-up rounds")
    return {
        "question": question,
        "dialogue_context": dialogue_context,
        "query_type": query_type,
        "expected_answer": expected_answer,
        "initial_hypothesis": initial,
        "rounds": rounds,
    }


def retrieve_round(
    retriever: ArknightsHybridRetriever,
    *,
    question: str,
    hypothesis: HypothesisDocument,
    queries: list[str],
    query_config: QueryConfig,
) -> list[dict[str, Any]]:
    dense_ranked_lists: list[list[dict[str, Any]]] = []
    sparse_ranked_lists: list[list[dict[str, Any]]] = []
    minirag_ranked_lists: list[list[dict[str, Any]]] = []
    for query in queries:
        dense_ranked_lists.append(retriever.dense_search(query, top_k=query_config.dense_top_k))
        sparse_ranked_lists.append(retriever.sparse_search(query, top_k=query_config.sparse_top_k))
        minirag_hits = retriever.minirag_search(query, top_k=query_config.minirag_top_k)
        if minirag_hits:
            minirag_ranked_lists.append(minirag_hits)
    dense_hits = merge_ranked_hits(*dense_ranked_lists)
    sparse_hits = merge_ranked_hits(*sparse_ranked_lists)
    minirag_hits = merge_ranked_hits(*minirag_ranked_lists)
    minirag_weight = retriever.effective_minirag_weight(question, config=query_config)
    if query_config.minirag_fusion_mode == "append":
        primary_hits = retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=[],
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
            minirag_weight=0.0,
        )
        fused_hits = retriever.append_supplemental_hits(
            primary_hits,
            minirag_hits if minirag_weight > 0 else [],
            top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
            source_name="minirag",
        )
    else:
        fused_hits = retriever.reciprocal_rank_fusion(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            minirag_hits=minirag_hits if minirag_weight > 0 else [],
            top_k=query_config.fusion_top_k,
            rrf_k=query_config.rrf_k,
            dense_weight=query_config.dense_weight,
            sparse_weight=query_config.sparse_weight,
            minirag_weight=minirag_weight,
        )
    if query_config.enable_neighbor_expansion:
        fused_hits = retriever.expand_hits_with_neighbors(
            fused_hits,
            max_seed_docs=query_config.neighbor_max_seed_docs,
            story_window=query_config.neighbor_story_window,
            activity_story_sort_window=query_config.neighbor_activity_story_sort_window,
            top_k=max(query_config.reranker_candidate_top_k, query_config.fusion_top_k),
        )
    return rerank_hits(
        retriever,
        question,
        fused_hits,
        top_k=query_config.rerank_top_k,
        batch_size=query_config.rerank_batch_size,
        query_mode=classify_retrieval_query_mode(hypothesis),
    )


def first_hit_rank(
    hits: list[dict[str, Any]],
    gold_text: str,
    *,
    max_k: int,
    jaccard_threshold: float,
    overlap_threshold: float,
    min_overlap_grams: int,
    min_candidate_grams: int,
) -> tuple[int, str, float] | None:
    if not gold_text:
        return None
    for rank, item in enumerate(hits[:max_k], start=1):
        hit = candidate_hit_source(
            item,
            gold_text,
            jaccard_threshold=jaccard_threshold,
            overlap_threshold=overlap_threshold,
            min_overlap_grams=min_overlap_grams,
            min_candidate_grams=min_candidate_grams,
        )
        if hit is not None:
            source, score = hit
            return rank, source, score
    return None


def hit_rank_value(hit: tuple[int, str, float] | None, *, miss_rank: int) -> int:
    return int(hit[0]) if hit is not None else miss_rank


def is_strong_new_term(term: str, previous_terms: set[str]) -> bool:
    normalized = re.sub(r"\s+", "", str(term or ""))
    if not normalized or normalized in previous_terms or normalized in WEAK_TERMS:
        return False
    if is_bad_retrieval_term(normalized, field="keywords"):
        return False
    if any(marker in normalized for marker in ("原因", "背景", "信息", "线索", "剧情", "情况", "过程")):
        return False
    return len(normalized) >= 2


def follow_up_has_strong_delta(previous: HypothesisDocument, follow_up: HypothesisDocument) -> bool:
    previous_terms = set(previous.entities + previous.keywords)
    current_terms = set(follow_up.entities + follow_up.keywords)
    strong = [term for term in current_terms - previous_terms if is_strong_new_term(term, previous_terms)]
    return len(strong) >= 2


def retrieval_gain_ok(
    previous_hit: tuple[int, str, float] | None,
    next_hit: tuple[int, str, float] | None,
    *,
    max_k: int,
    min_rank_improvement: int,
    require_hit_if_gold: bool,
) -> bool:
    if next_hit is None:
        return not require_hit_if_gold
    if previous_hit is None:
        return True
    previous_rank = hit_rank_value(previous_hit, miss_rank=max_k + 1)
    next_rank = hit_rank_value(next_hit, miss_rank=max_k + 1)
    if previous_rank > 20 and next_rank <= 20:
        return True
    if previous_rank > 10 and next_rank <= 10:
        return True
    return previous_rank - next_rank >= min_rank_improvement


def make_conclusion_result_for_follow_up(
    *,
    missing_slots: list[str],
    follow_up: HypothesisDocument,
) -> ConclusionResult:
    slots = [slot for slot in dedupe_keep_order(missing_slots) if slot][:5]
    if not slots:
        slots = ["缺少能直接连接问题与答案的桥接证据", "需要补充关键实体或事件上下文"]
    return ConclusionResult(
        next_action="retrieve_more",
        answer="",
        missing_slots=slots,
        clarification_question="",
        follow_up_hypothesis=follow_up,
    )


def make_final_conclusion_payload(
    *,
    question: str,
    answer: str,
    hit: tuple[int, str, float] | None,
    max_round_reached: bool,
) -> dict[str, Any]:
    if hit is not None and answer:
        action = "answer_directly"
        final_answer = compact_text(answer, 420)
        missing_slots: list[str] = []
    else:
        action = "abstain"
        final_answer = (
            "现有检索证据仍不足以稳定确认该问题。"
            if max_round_reached
            else "当前候选召回链没有继续提供可验证的有效检索增益。"
        )
        missing_slots = []
    return {
        "question": question,
        "next_action": action,
        "answer": final_answer,
        "missing_slots": missing_slots,
        "clarification_question": "",
        "follow_up_hypothesis": None,
    }


def process_seed(
    seed: dict[str, Any],
    *,
    api_config: TeacherApiConfig,
    retriever: ArknightsHybridRetriever,
    query_config: QueryConfig,
    output_dir: Path,
    max_rounds: int,
    api_retries: int,
    retry_sleep: float,
    validation_retries: int,
    raw_output: Path,
    max_evidence_items: int,
    max_evidence_chars: int,
    prompt_evidence_top_k: int,
    max_hit_k: int,
    min_rank_improvement: int,
    require_hit_if_gold: bool,
    jaccard_threshold: float,
    overlap_threshold: float,
    min_overlap_grams: int,
    min_candidate_grams: int,
    retrieval_lock: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    system_prompt = build_teacher_system()
    seed_key = str(seed["seed_key"])
    last_error: Exception | None = None
    for attempt in range(validation_retries + 1):
        payload = call_teacher_json(
            api_config,
            system_prompt=system_prompt,
            user_prompt=build_teacher_planned_chain_prompt(seed, max_rounds=max_rounds),
            retries=api_retries,
            retry_sleep=retry_sleep,
            raw_output=raw_output,
            request_meta={
                "question_key": seed_key,
                "task_type": "teacher_planned_retrieval_chain",
                "round": 0,
                "validation_attempt": attempt,
            },
        )
        try:
            planned = parse_planned_chain_payload(payload, seed=seed, max_rounds=max_rounds)
            return build_verified_records_for_plan(
                planned,
                seed=seed,
                retriever=retriever,
                query_config=query_config,
                max_rounds=max_rounds,
                max_evidence_items=max_evidence_items,
                max_evidence_chars=max_evidence_chars,
                prompt_evidence_top_k=prompt_evidence_top_k,
                max_hit_k=max_hit_k,
                min_rank_improvement=min_rank_improvement,
                require_hit_if_gold=require_hit_if_gold,
                jaccard_threshold=jaccard_threshold,
                overlap_threshold=overlap_threshold,
                min_overlap_grams=min_overlap_grams,
                min_candidate_grams=min_candidate_grams,
                retrieval_lock=retrieval_lock,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            append_jsonl(
                output_dir / "rejected_plans.jsonl",
                {
                    "seed_key": seed_key,
                    "attempt": attempt,
                    "error": str(exc),
                    "payload": payload,
                    "created_at": int(time.time()),
                },
            )
    raise last_error or RuntimeError("planned chain validation failed")


def build_verified_records_for_plan(
    planned: dict[str, Any],
    *,
    seed: dict[str, Any],
    retriever: ArknightsHybridRetriever,
    query_config: QueryConfig,
    max_rounds: int,
    max_evidence_items: int,
    max_evidence_chars: int,
    prompt_evidence_top_k: int,
    max_hit_k: int,
    min_rank_improvement: int,
    require_hit_if_gold: bool,
    jaccard_threshold: float,
    overlap_threshold: float,
    min_overlap_grams: int,
    min_candidate_grams: int,
    retrieval_lock: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question = str(planned["question"])
    dialogue_context = str(planned.get("dialogue_context") or "")
    answer = str(planned.get("expected_answer") or seed.get("answer") or "")
    gold_text = build_seed_gold_text(seed)
    if not gold_text:
        raise ValueError("missing verification gold evidence")
    initial: HypothesisDocument = planned["initial_hypothesis"]
    question_item = make_question_item(
        question=question,
        query_type=str(planned.get("query_type") or initial.query_type),
        source=str(seed.get("source") or "teacher_planned"),
        seed_key=str(seed.get("seed_key") or ""),
        dialogue_context=dialogue_context,
        difficulty="hard",
    )
    records: list[dict[str, Any]] = [
        make_sft_record_compat(
            record_id=f"{question_item['question_key']}-{INITIAL_HYPOTHESIS_TASK_TYPE}",
            task_type=INITIAL_HYPOTHESIS_TASK_TYPE,
            student_prompt=build_student_hypothesis_prompt(question, dialogue_context),
            assistant_payload=as_initial_payload(initial),
            question_item=question_item,
            round_index=None,
            retrieval_query=None,
            evidence=None,
        )
    ]
    trace: dict[str, Any] = {
        "question_key": question_item["question_key"],
        "question": question,
        "source_seed_key": seed.get("seed_key"),
        "source": seed.get("source"),
        "rounds": [],
    }
    current = initial
    previous_hit: tuple[int, str, float] | None = None
    previous_evidence: list[dict[str, Any]] = []
    accepted_followups = 0
    planned_rounds: list[dict[str, Any]] = planned["rounds"]

    for round_index in range(1, max_rounds + 1):
        queries = [question, build_retrieval_query(current)] if round_index == 1 else [build_retrieval_query(current)]
        if retrieval_lock is None:
            evidence = retrieve_round(retriever, question=question, hypothesis=current, queries=queries, query_config=query_config)
        else:
            with retrieval_lock:
                evidence = retrieve_round(retriever, question=question, hypothesis=current, queries=queries, query_config=query_config)
        hit = first_hit_rank(
            evidence,
            gold_text,
            max_k=max_hit_k,
            jaccard_threshold=jaccard_threshold,
            overlap_threshold=overlap_threshold,
            min_overlap_grams=min_overlap_grams,
            min_candidate_grams=min_candidate_grams,
        )
        trace["rounds"].append(
            {
                "round": round_index,
                "queries": queries,
                "hypothesis": as_initial_payload(current),
                "hit": {"rank": hit[0], "source": hit[1], "score": hit[2]} if hit else None,
                "top_doc_ids": [(item.get("document") or {}).get("id") for item in evidence[:5]],
            }
        )
        if round_index > 1:
            if not retrieval_gain_ok(
                previous_hit,
                hit,
                max_k=max_hit_k,
                min_rank_improvement=min_rank_improvement,
                require_hit_if_gold=bool(gold_text) and require_hit_if_gold,
            ):
                raise ValueError("follow-up did not improve verified retrieval")
            accepted_followups += 1
        if round_index <= len(planned_rounds):
            follow = planned_rounds[round_index - 1]["follow_up_hypothesis"]
            if not follow_up_has_strong_delta(current, follow):
                raise ValueError("follow-up lacks strong new bridge terms")
            conclusion = make_conclusion_result_for_follow_up(
                missing_slots=planned_rounds[round_index - 1].get("missing_slots") or [],
                follow_up=follow,
            )
            records.append(
                make_sft_record_compat(
                    record_id=f"{question_item['question_key']}-{CONCLUSION_TASK_TYPE}-{round_index:02d}",
                    task_type=CONCLUSION_TASK_TYPE,
                    student_prompt=build_student_conclusion_prompt(
                        question=question,
                        dialogue_context=dialogue_context,
                        hypothesis=current,
                        round_index=round_index,
                        max_rounds=max_rounds,
                        evidence=evidence,
                        max_evidence_items=max_evidence_items,
                        max_evidence_chars=max_evidence_chars,
                    ),
                    assistant_payload=as_conclusion_payload(conclusion, question=question),
                    question_item=question_item,
                    round_index=round_index,
                    retrieval_query="\n\n".join(queries),
                    evidence=evidence,
                )
            )
            records.append(
                make_sft_record_compat(
                    record_id=f"{question_item['question_key']}-{FOLLOW_UP_HYPOTHESIS_TASK_TYPE}-{round_index:02d}",
                    task_type=FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
                    student_prompt=build_student_follow_up_prompt(
                        question=question,
                        dialogue_context=dialogue_context,
                        hypothesis=current,
                        conclusion=conclusion,
                        round_index=round_index,
                        max_rounds=max_rounds,
                        evidence=evidence,
                        max_evidence_items=max_evidence_items,
                        max_evidence_chars=max_evidence_chars,
                    ),
                    assistant_payload=as_follow_up_payload(follow),
                    question_item=question_item,
                    round_index=round_index,
                    retrieval_query="\n\n".join(queries),
                    evidence=evidence,
                )
            )
            previous_hit = hit
            previous_evidence = evidence
            current = merge_hypotheses(current, follow)
            continue

        final_payload = make_final_conclusion_payload(
            question=question,
            answer=answer,
            hit=hit,
            max_round_reached=round_index >= max_rounds,
        )
        if hit is not None and not answer:
            trace["skipped_final_conclusion"] = "hit_without_expected_answer"
            break
        final_conclusion = normalize_conclusion_payload(
            final_payload,
            question=question,
            dialogue_context=dialogue_context,
            current_intent=current.intent,
            max_round_reached=round_index >= max_rounds,
        )
        records.append(
            make_sft_record_compat(
                record_id=f"{question_item['question_key']}-{CONCLUSION_TASK_TYPE}-{round_index:02d}",
                task_type=CONCLUSION_TASK_TYPE,
                student_prompt=build_student_conclusion_prompt(
                    question=question,
                    dialogue_context=dialogue_context,
                    hypothesis=current,
                    round_index=round_index,
                    max_rounds=max_rounds,
                    evidence=evidence,
                    max_evidence_items=max_evidence_items,
                    max_evidence_chars=max_evidence_chars,
                ),
                assistant_payload=as_conclusion_payload(final_conclusion, question=question),
                question_item=question_item,
                round_index=round_index,
                retrieval_query="\n\n".join(queries),
                evidence=evidence,
            )
        )
        break

    if accepted_followups <= 0:
        raise ValueError("no verified follow-up gain")
    trace["records"] = len(records)
    trace["accepted_followups"] = accepted_followups
    trace["final_hit"] = trace["rounds"][-1].get("hit") if trace["rounds"] else None
    return records, trace


def make_sft_record_compat(
    *,
    record_id: str,
    task_type: str,
    student_prompt: str,
    assistant_payload: dict[str, Any],
    question_item: dict[str, str],
    round_index: int | None,
    retrieval_query: str | None,
    evidence: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "category": "tool",
        "task_family": task_type,
        "generation_mode": "teacher_planned_retrieval_chain_v1",
        "short_prompt_schema": "current_pipeline_v1",
        "source_question_key": question_item["question_key"],
        "source_question": question_item["question"],
        "source_dialogue_context": question_item.get("dialogue_context") or "",
        "source_split": question_item.get("source_split") or "",
        "source_task_type": question_item.get("source_task_type") or "",
        "source_seed_key": question_item.get("source_seed_key") or "",
        "source_seed_index": question_item.get("source_seed_index") or "",
    }
    if round_index is not None:
        meta["round"] = round_index
    if retrieval_query:
        meta["retrieval_query"] = retrieval_query
    if evidence is not None:
        meta["evidence_doc_ids"] = [(item.get("document") or {}).get("id") for item in evidence[:12]]
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "tool",
        "system": "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。",
        "tools": "[]",
        "conversations": [
            {"from": "human", "value": student_prompt},
            {"from": "gpt", "value": compact_json(assistant_payload)},
        ],
        "meta": meta,
    }


def load_teacher_api_config(args: argparse.Namespace) -> TeacherApiConfig:
    teacher_cfg: dict[str, Any] = {}
    if args.teacher_config:
        loaded_cfg = load_json(resolve_path(args.teacher_config))
        teacher_cfg = loaded_cfg.get("teacher_api", loaded_cfg) if isinstance(loaded_cfg, dict) else {}
    api_type = str(args.api_type or teacher_cfg.get("api_type") or "chat_completions")
    api_base = str(args.api_base or teacher_cfg.get("base_url") or "")
    api_key_env = str(args.api_key_env or teacher_cfg.get("api_key_env") or "TEACHER_API_KEY")
    auth_header = str(args.auth_header or teacher_cfg.get("auth_header") or "bearer")
    model = str(args.model or teacher_cfg.get("model") or "")
    timeout = int(args.timeout if args.timeout is not None else teacher_cfg.get("timeout_seconds") or 300)
    temperature = float(
        args.temperature
        if args.temperature is not None
        else teacher_cfg.get("temperature") if teacher_cfg.get("temperature") is not None else 0.2
    )
    max_output_tokens = int(
        args.max_output_tokens
        if args.max_output_tokens is not None
        else teacher_cfg.get("max_output_tokens") or 4096
    )
    json_mode = bool(teacher_cfg.get("json_mode", True)) and not args.no_json_mode
    extra_headers = teacher_cfg.get("extra_headers") if isinstance(teacher_cfg.get("extra_headers"), dict) else None
    anthropic_disable_thinking = bool(
        args.anthropic_disable_thinking or teacher_cfg.get("anthropic_disable_thinking", False)
    )
    if not api_base or not model:
        raise SystemExit("--api-base and --model are required unless --export-only/--dry-run is used.")
    import os

    if not os.environ.get(api_key_env):
        raise SystemExit(f"Missing API key environment variable: {api_key_env}")
    return TeacherApiConfig(
        api_type=api_type,
        base_url=api_base,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        extra_headers=extra_headers,
        auth_header=auth_header,
        anthropic_disable_thinking=anthropic_disable_thinking,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate verified hard follow-up retrieval-chain SFT data compatible with previous online-chain datasets."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--listwise", type=Path, default=DEFAULT_LISTWISE)
    parser.add_argument("--eval-misses", type=Path, default=None)
    parser.add_argument("--documents", type=Path, default=INDEX_ROOT / "documents.jsonl")
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--max-eval-seeds", type=int, default=80)
    parser.add_argument("--include-eval-hits", action="store_true")
    parser.add_argument("--max-listwise-seeds", type=int, default=120)
    parser.add_argument("--max-document-seeds", type=int, default=40)
    parser.add_argument("--document-seed-doc-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--teacher-config", type=Path, default=None)
    parser.add_argument("--api-type", choices=("chat_completions", "anthropic_messages", "responses"), default=None)
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--auth-header", choices=("bearer", "x-api-key", "both"), default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument("--validation-retries", type=int, default=2)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--anthropic-disable-thinking", action="store_true")

    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--parallel-retrieval", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=RERANKER_MODEL_DIR)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--minirag-index", type=Path, default=DEFAULT_MINIRAG_INDEX)
    parser.add_argument("--dense-top-k", type=int, default=160)
    parser.add_argument("--sparse-top-k", type=int, default=160)
    parser.add_argument("--minirag-top-k", type=int, default=120)
    parser.add_argument("--fusion-top-k", type=int, default=120)
    parser.add_argument("--rerank-top-k", type=int, default=50)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=120)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--minirag-weight", type=float, default=0.35)
    parser.add_argument("--minirag-mode-weights", type=parse_mode_weights, default={})
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default="score")
    parser.add_argument("--enable-neighbor-expansion", action="store_true")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=24)
    parser.add_argument("--neighbor-story-window", type=int, default=2)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=1)

    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--prompt-evidence-top-k", type=int, default=8)
    parser.add_argument("--max-evidence-items", type=int, default=6)
    parser.add_argument("--max-evidence-chars", type=int, default=220)
    parser.add_argument("--max-hit-k", type=int, default=50)
    parser.add_argument("--min-rank-improvement", type=int, default=10)
    parser.add_argument("--no-require-hit-if-gold", dest="require_hit_if_gold", action="store_false", default=True)
    parser.add_argument("--jaccard-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=float, default=0.32)
    parser.add_argument("--min-overlap-grams", type=int, default=60)
    parser.add_argument("--min-candidate-grams", type=int, default=80)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_jsonl = output_dir / "records.jsonl"
    raw_output = output_dir / "raw_teacher_requests.jsonl"
    failed_output = output_dir / "failed_seeds.jsonl"
    trace_output = output_dir / "verified_traces.jsonl"

    if args.export_only:
        summary = export_splits(
            records_jsonl,
            output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    seeds = load_seed_pool(args)
    if args.dry_run:
        print(json.dumps({"seeds": len(seeds), "preview": seeds[:10]}, ensure_ascii=False, indent=2, default=str))
        return
    if not seeds:
        raise SystemExit("No seed items available.")

    done_keys = {
        str((record.get("meta") or {}).get("source_seed_key") or (record.get("meta") or {}).get("source_question_key") or "")
        for record in load_jsonl(records_jsonl)
    } if args.resume else set()
    if done_keys:
        seeds = [seed for seed in seeds if str(seed.get("seed_key") or "") not in done_keys]
    seeds = seeds[: args.max_seeds]
    log_progress(f"[seeds] pending={len(seeds)} resume_done={len(done_keys)}")

    api_config = load_teacher_api_config(args)
    query_config = QueryConfig(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        minirag_top_k=args.minirag_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_k=args.rerank_top_k,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights=args.minirag_mode_weights,
        minirag_fusion_mode=args.minirag_fusion_mode,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        enable_neighbor_expansion=args.enable_neighbor_expansion,
        neighbor_max_seed_docs=args.neighbor_max_seed_docs,
        neighbor_story_window=args.neighbor_story_window,
        neighbor_activity_story_sort_window=args.neighbor_activity_story_sort_window,
        rerank_batch_size=args.rerank_batch_size,
    )
    reranker_model_path = None if args.no_reranker else resolve_path(args.reranker_model)
    log_progress(
        f"[retriever-load] start index={resolve_path(args.index_dir)} device={args.device} "
        f"reranker={'off' if reranker_model_path is None else reranker_model_path}"
    )
    started = time.time()
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=resolve_path(args.embedding_model),
        reranker_model_path=reranker_model_path,
        documents_path=resolve_path(args.index_dir) / "documents.jsonl",
        faiss_index_path=resolve_path(args.index_dir) / "faiss.index",
        bm25_tokens_path=resolve_path(args.index_dir) / "bm25_tokens.pkl",
        minirag_index_path=resolve_path(args.minirag_index),
        device=args.device,
    )
    log_progress(f"[retriever-load] done latency={time.time() - started:.1f}s")

    stats: Counter[str] = Counter()
    retrieval_lock = None if args.parallel_retrieval or args.parallel <= 1 else Lock()

    def run_seed(seed: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        log_progress(f"[seed-start] source={seed.get('source')} key={str(seed.get('seed_key'))[:16]}")
        records, trace = process_seed(
            seed,
            api_config=api_config,
            retriever=retriever,
            query_config=query_config,
            output_dir=output_dir,
            max_rounds=args.max_rounds,
            api_retries=args.api_retries,
            retry_sleep=args.retry_sleep,
            validation_retries=args.validation_retries,
            raw_output=raw_output,
            max_evidence_items=args.max_evidence_items,
            max_evidence_chars=args.max_evidence_chars,
            prompt_evidence_top_k=args.prompt_evidence_top_k,
            max_hit_k=args.max_hit_k,
            min_rank_improvement=args.min_rank_improvement,
            require_hit_if_gold=args.require_hit_if_gold,
            jaccard_threshold=args.jaccard_threshold,
            overlap_threshold=args.overlap_threshold,
            min_overlap_grams=args.min_overlap_grams,
            min_candidate_grams=args.min_candidate_grams,
            retrieval_lock=retrieval_lock,
        )
        return seed, records, trace

    parallel = max(1, args.parallel)
    if parallel == 1:
        iterator = tqdm(seeds, desc="teacher planned hard chains", unit="seed")
        for seed in iterator:
            try:
                _, records, trace = run_seed(seed)
                for record in records:
                    append_jsonl(records_jsonl, record)
                    stats[f"task:{record['task_type']}"] += 1
                append_jsonl(trace_output, trace)
                stats["completed_seeds"] += 1
                stats["records"] += len(records)
                stats["accepted_followups"] += int(trace.get("accepted_followups") or 0)
            except Exception as exc:  # noqa: BLE001
                stats["failed_seeds"] += 1
                append_jsonl(failed_output, {"seed": seed, "error": str(exc), "created_at": int(time.time())})
            iterator.set_postfix({"ok": stats["completed_seeds"], "failed": stats["failed_seeds"], "records": stats["records"]})
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_seed = {executor.submit(run_seed, seed): seed for seed in seeds}
            progress = tqdm(as_completed(future_to_seed), total=len(future_to_seed), desc="teacher planned hard chains", unit="seed")
            for future in progress:
                seed = future_to_seed[future]
                try:
                    _, records, trace = future.result()
                    for record in records:
                        append_jsonl(records_jsonl, record)
                        stats[f"task:{record['task_type']}"] += 1
                    append_jsonl(trace_output, trace)
                    stats["completed_seeds"] += 1
                    stats["records"] += len(records)
                    stats["accepted_followups"] += int(trace.get("accepted_followups") or 0)
                except Exception as exc:  # noqa: BLE001
                    stats["failed_seeds"] += 1
                    append_jsonl(failed_output, {"seed": seed, "error": str(exc), "created_at": int(time.time())})
                progress.set_postfix({"ok": stats["completed_seeds"], "failed": stats["failed_seeds"], "records": stats["records"]})

    summary = export_splits(
        records_jsonl,
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    summary["run_stats"] = dict(stats)
    summary["output_dir"] = str(output_dir)
    summary["seed_sources"] = dict(Counter(str(seed.get("source") or "") for seed in seeds))
    summary["query_config"] = {
        "dense_top_k": args.dense_top_k,
        "sparse_top_k": args.sparse_top_k,
        "fusion_top_k": args.fusion_top_k,
        "rerank_top_k": args.rerank_top_k,
        "minirag_top_k": args.minirag_top_k,
        "minirag_weight": args.minirag_weight,
        "minirag_fusion_mode": args.minirag_fusion_mode,
    }
    summary["verification"] = {
        "max_hit_k": args.max_hit_k,
        "min_rank_improvement": args.min_rank_improvement,
        "require_hit_if_gold": args.require_hit_if_gold,
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
