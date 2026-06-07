#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
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

from goldenglow.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    INDEX_ROOT,
    MINIRAG_GRAPH_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1"
DEFAULT_TEACHER_RUNTIME_CONFIG = PROJECT_ROOT / "api-mode/runtime_deepseek_api.json"
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_inference_gpu.json"
ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
QUESTION_RE = re.compile(r"(?m)^question:\s*(.+?)\s*$")
ROUND_RE = re.compile(r"(?m)^round:\s*(.+?)\s*$")
HYPOTHESIS_RE = re.compile(r"(?s)^hypothesis:\s*(\{.*?\})\s*(?:^round:|\Z)", re.MULTILINE)
EVIDENCE_BRIEF_RE = re.compile(r"(?s)^evidence_brief:\s*(.*?)(?:\nminirag_hints:|\noutput_schema:|\nfields:|\Z)", re.MULTILINE)
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_.\-]{1,31}")


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_path(value: Any, *, default: Path | None = None) -> Path | None:
    selected = value if value not in (None, "") else default
    if selected in (None, ""):
        return None
    path = Path(selected)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def config_value(cli_value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def build_query_config(
    args: argparse.Namespace,
    retrieval_cfg: dict[str, Any],
    *,
    rerank_top_k: int,
) -> QueryConfig:
    configured_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    minirag_mode_weights = (
        {str(key): float(value) for key, value in configured_weights.items()}
        if isinstance(configured_weights, dict)
        else {}
    )
    return QueryConfig(
        dense_top_k=int(config_value(args.dense_top_k, retrieval_cfg, "dense_top_k", 120)),
        sparse_top_k=int(config_value(args.sparse_top_k, retrieval_cfg, "sparse_top_k", 120)),
        minirag_top_k=int(config_value(args.minirag_top_k, retrieval_cfg, "minirag_top_k", 120)),
        fusion_top_k=int(config_value(args.fusion_top_k, retrieval_cfg, "fusion_top_k", 80)),
        rerank_top_k=rerank_top_k,
        minirag_weight=float(config_value(args.minirag_weight, retrieval_cfg, "minirag_weight", 0.35)),
        minirag_mode_weights=minirag_mode_weights,
        minirag_fusion_mode=str(config_value(args.minirag_fusion_mode, retrieval_cfg, "minirag_fusion_mode", "score")),
        minirag_chapter_isolation=bool(config_value(args.minirag_chapter_isolation, retrieval_cfg, "minirag_chapter_isolation", True)),
        minirag_auto_second_retrieval=bool(
            config_value(args.minirag_auto_second_retrieval, retrieval_cfg, "minirag_auto_second_retrieval", True)
        ),
        minirag_scope_seed_top_k=int(config_value(args.minirag_scope_seed_top_k, retrieval_cfg, "minirag_scope_seed_top_k", 40)),
        minirag_expansion_query_top_k=int(
            config_value(args.minirag_expansion_query_top_k, retrieval_cfg, "minirag_expansion_query_top_k", 8)
        ),
        minirag_graph_scope_min_ratio=float(
            config_value(args.minirag_graph_scope_min_ratio, retrieval_cfg, "minirag_graph_scope_min_ratio", 1.0)
        ),
        minirag_second_pass_scope_min_ratio=float(
            config_value(args.minirag_second_pass_scope_min_ratio, retrieval_cfg, "minirag_second_pass_scope_min_ratio", 2.5)
        ),
        enable_storyline_sparse_scope=bool(
            config_value(args.enable_storyline_sparse_scope, retrieval_cfg, "enable_storyline_sparse_scope", True)
        ),
        storyline_scope_seed_top_k=int(config_value(args.storyline_scope_seed_top_k, retrieval_cfg, "storyline_scope_seed_top_k", 40)),
        storyline_sparse_scope_min_ratio=float(
            config_value(args.storyline_sparse_scope_min_ratio, retrieval_cfg, "storyline_sparse_scope_min_ratio", 1.5)
        ),
        enable_scoped_chapter_search=bool(
            config_value(getattr(args, "enable_scoped_chapter_search", None), retrieval_cfg, "enable_scoped_chapter_search", True)
        ),
        scoped_chapter_dense_top_k=int(
            config_value(getattr(args, "scoped_chapter_dense_top_k", None), retrieval_cfg, "scoped_chapter_dense_top_k", 160)
        ),
        scoped_chapter_sparse_top_k=int(
            config_value(getattr(args, "scoped_chapter_sparse_top_k", None), retrieval_cfg, "scoped_chapter_sparse_top_k", 160)
        ),
        reranker_candidate_top_k=int(config_value(args.reranker_candidate_top_k, retrieval_cfg, "reranker_candidate_top_k", 120)),
        enable_neighbor_expansion=bool(config_value(args.enable_neighbor_expansion, retrieval_cfg, "enable_neighbor_expansion", False)),
        neighbor_max_seed_docs=int(config_value(args.neighbor_max_seed_docs, retrieval_cfg, "neighbor_max_seed_docs", 24)),
        neighbor_story_window=int(config_value(args.neighbor_story_window, retrieval_cfg, "neighbor_story_window", 2)),
        neighbor_activity_story_sort_window=int(
            config_value(args.neighbor_activity_story_sort_window, retrieval_cfg, "neighbor_activity_story_sort_window", 1)
        ),
        enable_same_story_sweep=bool(
            config_value(getattr(args, "enable_same_story_sweep", None), retrieval_cfg, "enable_same_story_sweep", True)
        ),
        same_story_sweep_max_seed_docs=int(
            config_value(getattr(args, "same_story_sweep_max_seed_docs", None), retrieval_cfg, "same_story_sweep_max_seed_docs", 8)
        ),
        same_story_sweep_max_docs_per_story=int(
            config_value(
                getattr(args, "same_story_sweep_max_docs_per_story", None),
                retrieval_cfg,
                "same_story_sweep_max_docs_per_story",
                24,
            )
        ),
        same_story_sweep_extra_candidates=int(
            config_value(
                getattr(args, "same_story_sweep_extra_candidates", None),
                retrieval_cfg,
                "same_story_sweep_extra_candidates",
                80,
            )
        ),
        rerank_batch_size=int(config_value(args.rerank_batch_size, retrieval_cfg, "rerank_batch_size", 8)),
    )


def load_api_mode_module() -> Any:
    module_path = PROJECT_ROOT / "api-mode/run_api_inference.py"
    spec = importlib.util.spec_from_file_location("goldenglow_api_mode_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API mode runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_known_api_key_prefix(api_key: str) -> str:
    key = str(api_key or "").strip()
    if key.startswith("ds:sk-"):
        return key.split(":", 1)[1]
    return key


def resolve_local_path(path: Path | None, default: Path | None = None) -> Path | None:
    selected = path if path not in (None, "") else default
    if selected in (None, ""):
        return None
    selected = Path(selected)
    return selected if selected.is_absolute() else PROJECT_ROOT / selected


def prompt_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[0].get("value") or "")
    return ""


def response_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return str(conversations[-1].get("value") or "").strip()
    return ""


def parse_jsonish(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(raw[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_round(prompt: str) -> str:
    match = ROUND_RE.search(prompt or "")
    return match.group(1).strip() if match else ""


def extract_hypothesis(prompt: str) -> dict[str, Any]:
    match = HYPOTHESIS_RE.search(prompt or "")
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def extract_evidence_brief(prompt: str) -> str:
    match = EVIDENCE_BRIEF_RE.search(prompt or "")
    if not match:
        return ""
    return match.group(1).strip()


def infer_task_type(record: dict[str, Any]) -> str:
    return str(record.get("task_type") or "")


def action(payload: dict[str, Any] | None) -> str:
    return str((payload or {}).get("next_action") or "").strip()


def tokenize(text: str) -> list[str]:
    seen: set[str] = set()
    tokens: list[str] = []
    for token in TOKEN_RE.findall(text or ""):
        normalized = re.sub(r"\s+", "", token).strip("，。！？；：、（）()[]【】《》“”\"'")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tokens


def build_fallback_follow_up(question: str, prompt: str, missing_slots: list[str]) -> dict[str, Any]:
    hypothesis = extract_hypothesis(prompt)
    entities = [str(item).strip() for item in hypothesis.get("entities", []) if str(item).strip()]
    keywords = [str(item).strip() for item in hypothesis.get("keywords", []) if str(item).strip()]
    for slot in missing_slots:
        keywords.extend(tokenize(slot)[:6])
    if not entities:
        entities = tokenize(question)[:6]
    if not keywords:
        keywords = tokenize(question)[:12]
    return {
        "question": question,
        "query_type": str(hypothesis.get("query_type") or "reasoning"),
        "entities": entities[:12],
        "keywords": list(dict.fromkeys(keywords))[:24],
        "expected_answer_type": str(hypothesis.get("expected_answer_type") or "剧情问答"),
        "dialogue_context": str(hypothesis.get("dialogue_context") or ""),
    }


def normalize_string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        items = re.split(r"[、,，;；]\s*", value)
    elif isinstance(value, list):
        items = value
    else:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
        if len(output) >= limit:
            break
    return output


def normalize_follow_up_payload(payload: dict[str, Any] | None, *, question: str, prompt: str) -> dict[str, Any]:
    hypothesis = extract_hypothesis(prompt)
    source = payload if isinstance(payload, dict) else {}
    entities = normalize_string_list(source.get("entities"), limit=12) or normalize_string_list(hypothesis.get("entities"), limit=12)
    keywords = normalize_string_list(source.get("keywords"), limit=24) or normalize_string_list(hypothesis.get("keywords"), limit=24)
    if not entities:
        entities = tokenize(question)[:6]
    if not keywords:
        keywords = tokenize(question)[:12]
    return {
        "question": str(source.get("question") or question).strip(),
        "query_type": str(source.get("query_type") or hypothesis.get("query_type") or "reasoning").strip(),
        "entities": entities[:12],
        "keywords": list(dict.fromkeys(keywords))[:24],
        "expected_answer_type": str(source.get("expected_answer_type") or hypothesis.get("expected_answer_type") or "剧情问答").strip(),
        "dialogue_context": str(source.get("dialogue_context") or hypothesis.get("dialogue_context") or "").strip(),
    }


def iter_evidence_items(prompt: str) -> list[tuple[str, str]]:
    evidence = extract_evidence_brief(prompt)
    if not evidence:
        return []
    pattern = re.compile(r"(?ms)^\s*\d+\.\s+([^:\n]+):\s*(.*?)(?=^\s*\d+\.\s+[^:\n]+:|\Z)")
    items: list[tuple[str, str]] = []
    for match in pattern.finditer(evidence):
        evidence_id = match.group(1).strip()
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        if evidence_id and text:
            items.append((evidence_id, text))
    return items


def quote_candidates(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"(?<=[。！？；;.!?])\s*|\s{2,}", text) if part.strip()]
    if not parts:
        parts = [text]
    candidates: list[str] = []
    for part in parts:
        if len(part) <= 80:
            candidates.append(part)
            continue
        for start in range(0, len(part), 60):
            chunk = part[start : start + 80].strip()
            if chunk:
                candidates.append(chunk)
    return candidates


def quote_score(answer: str, quote: str) -> float:
    answer_tokens = set(tokenize(answer))
    quote_tokens = set(tokenize(quote))
    token_overlap = len(answer_tokens & quote_tokens)
    answer_chars = {char for char in answer if "\u4e00" <= char <= "\u9fff" or char.isalnum()}
    quote_chars = {char for char in quote if "\u4e00" <= char <= "\u9fff" or char.isalnum()}
    char_overlap = len(answer_chars & quote_chars)
    return token_overlap * 3.0 + char_overlap * 0.2


def select_evidence_refs(answer: str, prompt: str, *, max_refs: int = 2) -> list[dict[str, str]]:
    scored: list[tuple[float, str, str]] = []
    for evidence_id, text in iter_evidence_items(prompt):
        for quote in quote_candidates(text):
            if len(quote) < 8:
                continue
            scored.append((quote_score(answer, quote), evidence_id, quote[:80]))
    scored.sort(key=lambda item: item[0], reverse=True)
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_quotes: set[str] = set()
    for score, evidence_id, quote in scored:
        if score <= 0:
            continue
        key = (evidence_id, quote)
        normalized_quote = re.sub(r"\s+", "", quote)
        if key in seen or normalized_quote in seen_quotes:
            continue
        refs.append({"evidence_id": evidence_id, "quote": quote})
        seen.add(key)
        seen_quotes.add(normalized_quote)
        if len(refs) >= max_refs:
            break
    return refs


def normalize_evidence_refs(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, str]] = []
    total = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        quote = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip()
        if not evidence_id or not quote:
            continue
        quote = quote[:80]
        if total + len(quote) > 160:
            break
        refs.append({"evidence_id": evidence_id, "quote": quote})
        total += len(quote)
        if len(refs) >= 2:
            break
    return refs


def normalize_supported_facts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    facts: list[dict[str, Any]] = []
    quote_budget = 400
    for item in value:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        refs = normalize_evidence_refs(item.get("evidence_refs"))
        if not fact:
            continue
        kept_refs: list[dict[str, str]] = []
        for ref in refs:
            quote_len = len(ref["quote"])
            if quote_len > quote_budget:
                continue
            kept_refs.append(ref)
            quote_budget -= quote_len
        facts.append({"fact": fact, "evidence_refs": kept_refs})
        if len(facts) >= 6 or quote_budget <= 0:
            break
    return facts


def normalize_inferred_facts(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    output: list[Any] = []
    for item in value[:6]:
        if isinstance(item, dict):
            fact = str(item.get("fact") or "").strip()
            if fact:
                normalized = {"fact": fact}
                premise_ids = normalize_string_list(item.get("premise_fact_ids"), limit=6)
                if premise_ids:
                    normalized["premise_fact_ids"] = premise_ids
                output.append(normalized)
        else:
            fact = str(item or "").strip()
            if fact:
                output.append(fact)
    return output


def normalize_grounded_action_payload(
    payload: dict[str, Any] | None,
    *,
    question: str,
    prompt: str,
    force_action: str | None = None,
    fallback_answer: str = "",
    missing_slots: list[str] | None = None,
) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    next_action = str(force_action or source.get("next_action") or "retrieve_more").strip()
    if next_action not in {"answer_directly", "retrieve_more", "abstain", "clarify_user"}:
        next_action = "retrieve_more"
    if next_action == "clarify_user":
        next_action = "abstain"

    if next_action == "answer_directly":
        final_answer = str(source.get("final_answer") or source.get("answer") or fallback_answer or "").strip()
        supported_facts = normalize_supported_facts(source.get("supported_facts"))
        return {
            "next_action": "answer_directly",
            "supported_facts": supported_facts,
            "inferred_facts": normalize_inferred_facts(source.get("inferred_facts")),
            "final_answer": final_answer or "现有证据不足以确认。",
        }

    if next_action == "retrieve_more":
        follow_up = source.get("follow_up_hypothesis") if isinstance(source.get("follow_up_hypothesis"), dict) else None
        if follow_up is None:
            follow_up = build_fallback_follow_up(question, prompt, missing_slots or [])
        return {
            "next_action": "retrieve_more",
            "follow_up_hypothesis": normalize_follow_up_payload(follow_up, question=question, prompt=prompt),
        }

    return {
        "next_action": "abstain",
        "final_answer": str(source.get("final_answer") or source.get("answer") or fallback_answer or "现有证据不足以确认。").strip(),
    }


def normalize_conclusion_payload(payload: dict[str, Any], *, question: str, prompt: str) -> dict[str, Any]:
    next_action = str(payload.get("next_action") or "retrieve_more").strip()
    if next_action not in {"answer_directly", "retrieve_more", "clarify_user", "abstain"}:
        next_action = "retrieve_more"
    missing_slots = payload.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = []
    missing_slots = [str(item).strip() for item in missing_slots if str(item).strip()][:8]
    answer = str(payload.get("answer") or "").strip()
    clarification_question = str(payload.get("clarification_question") or "").strip()
    follow_up = payload.get("follow_up_hypothesis") if isinstance(payload.get("follow_up_hypothesis"), dict) else None
    if next_action == "retrieve_more" and follow_up is None:
        follow_up = build_fallback_follow_up(question, prompt, missing_slots)
    if next_action == "retrieve_more":
        answer = ""
    if next_action in {"answer_directly", "abstain"}:
        follow_up = None
        if not answer:
            answer = "现有证据不足以确认。"
    if next_action == "clarify_user":
        follow_up = None
        if not clarification_question:
            clarification_question = "请补充你想确认的具体剧情范围。"
    return {
        "question": question,
        "next_action": next_action,
        "answer": answer,
        "missing_slots": missing_slots,
        "clarification_question": clarification_question,
        "follow_up_hypothesis": follow_up,
    }


def build_verifier_prompt(
    *,
    prompt_key: str,
    task_prompt: str,
    student_output: str,
    teacher_output: str,
) -> str:
    question = extract_question(task_prompt)
    round_id = extract_round(task_prompt)
    evidence_brief = extract_evidence_brief(task_prompt)
    return (
        "<|im_start|>system\n"
        "你是《明日方舟》RAG 轨迹的 evidence-only verifier。只输出 JSON，不要输出思维过程。\n"
        "你只能依据用户给出的 allowed_evidence_brief 判断，不能使用你自己的剧情知识补证据。\n"
        "teacher_policy_output 只是候选输出，不是标准答案；不要因为它看起来合理就采纳。\n"
        "目标是判断当前 evidence state 下应该 answer_directly、retrieve_more、clarify_user 还是 abstain。\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + "\n".join(
            [
                "请校验同一个 evidence state 下 student_output 与 teacher_policy_output 哪个动作可训练。",
                "",
                f"prompt_key: {prompt_key}",
                f"question: {question}",
                f"round: {round_id}",
                "",
                "allowed_evidence_brief（唯一可用于事实支撑的证据）:",
                evidence_brief or "<empty>",
                "",
                "student_output:",
                student_output.strip() or "<empty>",
                "",
                "teacher_policy_output:",
                teacher_output.strip() or "<empty>",
                "",
                "输出 JSON 字段:",
                "evidence_sufficient: boolean",
                "correct_action: answer_directly | retrieve_more | clarify_user | abstain",
                "supported_answer: string，只有 evidence 足以回答时填写，只写证据支持内容",
                "missing_slots: string[]，证据不足时写具体缺口",
                "student_action_error: none | premature_answer | over_retrieve | over_abstain | unsupported_answer | invalid_output",
                "teacher_action_error: none | premature_answer | over_retrieve | over_abstain | unsupported_answer | teacher_prior_knowledge | invalid_output",
                "teacher_answer_uses_prior_knowledge: boolean",
                "use_for_training: boolean",
                "label_reason: string，短句说明，不要写长推理",
                "",
                "规则:",
                "1. 如果答案中的关键实体、身份、因果、动机或结果无法在 evidence 中找到支撑，标为 unsupported/teacher_prior_knowledge。",
                "2. 如果当前 evidence 只能支持部分答案，supported_answer 只写可确认部分；不要补全。",
                "3. 如果证据不足但有明确检索方向，correct_action=retrieve_more。",
                "4. 如果证据足够回答核心问题，correct_action=answer_directly。",
                "5. 如果答案只是常识/剧情先验/模型记忆支持，而不是 evidence 明确支持，仍然判为证据不足。",
                "6. hypothesis、entities、keywords、minirag_hints、student_output、teacher_policy_output 都不是证据，不能用来支撑事实。",
                "7. evidence_brief 中的网页摘要、玩家总结也只能按其明文内容使用，不能外推隐藏剧情。",
                "8. 问题若包含特定实体/别名/引号词/事件名，answer_directly 需要 evidence 明确出现该实体或明确说明等价关系；只出现疑似相关人物不够。",
                "9. 例如问题问“朔的幻象”，evidence 只写“重岳/宗师/比武”但没有“朔/幻象/二者等同”，必须判证据不足。",
                "10. 只要 correct_action 能明确判定，就应 use_for_training=true；证据不足但应继续检索的早答负例尤其要保留。",
                "11. 只有 prompt/evidence 严重损坏、问题无法判断、或无法构造安全 chosen/rejected 时，才 use_for_training=false。",
                "12. 输出必须是单个 JSON 对象。",
            ]
        )
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def build_teacher_generator(args: argparse.Namespace, output_dir: Path) -> Any:
    teacher_runtime = load_runtime_config(resolve_local_path(args.teacher_runtime_config, DEFAULT_TEACHER_RUNTIME_CONFIG) or DEFAULT_TEACHER_RUNTIME_CONFIG)
    generator_cfg = teacher_runtime.get("generator", {}) if isinstance(teacher_runtime.get("generator"), dict) else {}
    api_mode = load_api_mode_module()
    backend = str(args.teacher_backend or generator_cfg.get("backend") or "chat_completions")
    if backend in {"openai_compatible_api", "chat_completions"}:
        generator_cls = api_mode.OpenAICompatibleAPIRunner
    elif backend in {"responses_api", "responses"}:
        generator_cls = api_mode.ResponsesAPIRunner
    else:
        raise SystemExit(f"Unsupported teacher backend: {backend}")
    api_key_env = str(args.api_key_env or generator_cfg.get("api_key_env") or "DEEPSEEK_API_KEY")
    api_key = strip_known_api_key_prefix(args.api_key or os.environ.get(api_key_env, ""))
    if not api_key:
        raise SystemExit(f"Missing API key. Set {api_key_env} or pass --api-key.")
    api_mode.validate_api_key(api_key, api_key_env)
    request_log_dir = output_dir / "api_request_logs" if args.save_api_request_logs else None
    return generator_cls(
        api_base_url=str(args.api_base_url or generator_cfg.get("api_base_url") or "https://api.deepseek.com"),
        api_key=api_key,
        api_key_env=api_key_env,
        model=str(args.api_model or generator_cfg.get("model") or "deepseek-v4-flash"),
        timeout=float(args.api_timeout if args.api_timeout is not None else generator_cfg.get("timeout", 120)),
        max_tokens=max(int(args.teacher_max_tokens if args.teacher_max_tokens is not None else generator_cfg.get("max_tokens", 4096)), 4096),
        temperature=float(args.teacher_temperature if args.teacher_temperature is not None else generator_cfg.get("temperature", 0.1)),
        top_p=float(args.teacher_top_p if args.teacher_top_p is not None else generator_cfg.get("top_p", 0.9)),
        response_format_json=True,
        extra_body=generator_cfg.get("extra_body") if isinstance(generator_cfg.get("extra_body"), dict) else None,
        request_log_dir=request_log_dir,
    )


def load_audit_groups(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records = read_jsonl(input_dir / "audit_records.jsonl")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        prompt_key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        groups[prompt_key].append(record)
    return groups


def load_existing_verifier(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_key = str(record.get("prompt_key") or "")
            if prompt_key:
                output[prompt_key] = record
    return output


def record_for_role(group: list[dict[str, Any]], *, kto_tag: bool) -> dict[str, Any] | None:
    return next((record for record in group if bool(record.get("kto_tag")) is kto_tag), None)


def make_kto_record(base: dict[str, Any], *, record_id: str, response: dict[str, Any] | str, kto_tag: bool, reason: str, verifier: dict[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(base, ensure_ascii=False))
    output["id"] = record_id
    output["kto_tag"] = bool(kto_tag)
    conversations = output.get("conversations")
    if isinstance(conversations, list) and conversations:
        conversations[-1]["value"] = compact_json(response) if isinstance(response, dict) else str(response)
    meta = output.setdefault("meta", {})
    meta["api_verifier_reason"] = reason
    meta["api_verifier"] = verifier
    meta["label_reason"] = str(verifier.get("label_reason") or reason)
    meta["verifier_correct_action"] = str(verifier.get("correct_action") or "")
    meta["verifier_evidence_sufficient"] = verifier.get("evidence_sufficient")
    return output


def relabel_group(prompt_key: str, group: list[dict[str, Any]], verifier_record: dict[str, Any], stats: Counter) -> list[dict[str, Any]]:
    teacher = record_for_role(group, kto_tag=True)
    student = record_for_role(group, kto_tag=False)
    if teacher is None:
        stats["drop:no_teacher_record"] += 1
        return []
    if infer_task_type(teacher) != "conclusion_generation":
        stats["keep:non_conclusion"] += len(group)
        return group
    verdict = verifier_record.get("verifier") if isinstance(verifier_record.get("verifier"), dict) else {}
    if not verdict:
        stats["drop:empty_verifier"] += len(group)
        return []

    prompt = prompt_text(teacher)
    question = extract_question(prompt)
    correct_action = str(verdict.get("correct_action") or "").strip()
    if correct_action not in {"answer_directly", "retrieve_more", "clarify_user", "abstain"}:
        stats["drop:invalid_correct_action"] += len(group)
        return []

    teacher_payload = parse_jsonish(response_text(teacher)) or {}
    student_payload = parse_jsonish(response_text(student)) if student is not None else None
    missing_slots = verdict.get("missing_slots")
    if not isinstance(missing_slots, list):
        missing_slots = []
    missing_slots = [str(item).strip() for item in missing_slots if str(item).strip()][:8]
    use_for_training = bool(verdict.get("use_for_training", True))
    actionable_false = (
        not use_for_training
        and (
            correct_action == "retrieve_more"
            or (correct_action == "answer_directly" and str(verdict.get("supported_answer") or "").strip())
            or correct_action in {"clarify_user", "abstain"}
        )
    )
    if not use_for_training and not actionable_false:
        stats["drop:verifier_use_for_training_false"] += len(group)
        return []
    if actionable_false:
        stats["override:actionable_use_for_training_false"] += 1
    chosen_payload: dict[str, Any]
    if correct_action == "answer_directly":
        candidate = None
        for payload in (teacher_payload, student_payload):
            if isinstance(payload, dict) and action(payload) == "answer_directly" and payload.get("supported_facts"):
                candidate = payload
                break
        if candidate is None:
            stats["skip:answer_directly_no_grounded_candidate"] += 1
            chosen_payload = {}
        else:
            chosen_payload = normalize_grounded_action_payload(
                candidate,
                question=question,
                prompt=prompt,
                force_action="answer_directly",
            )
    elif correct_action == "retrieve_more":
        teacher_error = str(verdict.get("teacher_action_error") or "none")
        student_error = str(verdict.get("student_action_error") or "none")
        teacher_prior = bool(verdict.get("teacher_answer_uses_prior_knowledge"))
        candidate = None
        if (
            action(teacher_payload) == "retrieve_more"
            and teacher_error in {"", "none"}
            and not teacher_prior
        ):
            candidate = teacher_payload
        elif (
            isinstance(student_payload, dict)
            and action(student_payload) == "retrieve_more"
            and student_error in {"", "none"}
        ):
                candidate = student_payload
        if not isinstance(candidate, dict) or action(candidate) != "retrieve_more":
            candidate = {
                "next_action": "retrieve_more",
                "follow_up_hypothesis": build_fallback_follow_up(question, prompt, missing_slots),
            }
        chosen_payload = normalize_grounded_action_payload(
            candidate,
            question=question,
            prompt=prompt,
            force_action="retrieve_more",
            missing_slots=missing_slots,
        )
    elif correct_action == "abstain":
        chosen_payload = normalize_grounded_action_payload(
            {
                "next_action": "abstain",
                "final_answer": str(verdict.get("supported_answer") or "现有证据不足以确认。"),
            },
            question=question,
            prompt=prompt,
            force_action="abstain",
        )
    else:
        chosen_payload = normalize_grounded_action_payload(
            {
                "next_action": "abstain",
                "final_answer": "现有证据不足以确认。",
            },
            question=question,
            prompt=prompt,
            force_action="abstain",
        )

    output = []
    if chosen_payload:
        output.append(
            make_kto_record(
                teacher,
                record_id=f"{prompt_key}-verifier-chosen",
                response=chosen_payload,
                kto_tag=True,
                reason=f"verifier_chosen_{correct_action}",
                verifier=verdict,
            )
        )
    for role, record, payload, error_field in (
        ("student", student, student_payload, "student_action_error"),
        ("teacher", teacher, teacher_payload, "teacher_action_error"),
    ):
        if record is None or not isinstance(payload, dict):
            continue
        record_action = action(payload)
        error = str(verdict.get(error_field) or "none")
        prior = bool(verdict.get("teacher_answer_uses_prior_knowledge")) and role == "teacher"
        if record_action == correct_action and error in {"", "none"} and not prior:
            continue
        rejected_payload = normalize_grounded_action_payload(
            payload,
            question=question,
            prompt=prompt,
            force_action=record_action if record_action in {"answer_directly", "retrieve_more", "abstain"} else None,
            missing_slots=missing_slots,
        )
        output.append(
            make_kto_record(
                record,
                record_id=f"{prompt_key}-{role}-verifier-rejected",
                response=rejected_payload,
                kto_tag=False,
                reason=f"reject_{role}_{error or 'wrong_action'}",
                verifier=verdict,
            )
        )
    stats[f"verifier_correct_action:{correct_action}"] += 1
    stats[f"records_out:{len(output)}"] += 1
    return output


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        by_prompt[key].append(record)
    keys = list(by_prompt)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = int(round(len(records) * val_ratio)) if val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for key in keys:
        bucket = by_prompt[key]
        if val_count < target_val:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def dataset_info(dataset_name: str) -> dict[str, Any]:
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
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def run_verifier(args: argparse.Namespace, output_dir: Path, groups: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    verifier_path = output_dir / "api_verifier_records.jsonl"
    if args.overwrite_verifier and verifier_path.exists():
        verifier_path.unlink()
    existing = load_existing_verifier(verifier_path)
    pending: list[tuple[str, list[dict[str, Any]]]] = []
    for prompt_key, group in groups.items():
        teacher = record_for_role(group, kto_tag=True)
        if teacher is None or infer_task_type(teacher) != "conclusion_generation":
            continue
        if prompt_key in existing and not args.overwrite_verifier:
            continue
        pending.append((prompt_key, group))
    if args.max_verifier_prompts is not None:
        pending = pending[: max(0, args.max_verifier_prompts)]
    if not pending:
        return existing

    teacher_api = build_teacher_generator(args, output_dir)
    progress = tqdm(pending, desc="api verifier", unit="prompt")
    for prompt_key, group in progress:
        teacher = record_for_role(group, kto_tag=True)
        student = record_for_role(group, kto_tag=False)
        assert teacher is not None
        verifier_prompt = build_verifier_prompt(
            prompt_key=prompt_key,
            task_prompt=prompt_text(teacher),
            student_output=response_text(student or {}),
            teacher_output=response_text(teacher),
        )
        started = time.perf_counter()
        try:
            raw = teacher_api.generate(
                verifier_prompt,
                max_tokens=args.verifier_max_tokens,
                temperature=args.teacher_temperature,
                top_p=args.teacher_top_p,
                repeat_penalty=1.0,
            )
            payload = parse_jsonish(raw) or {}
            record = {
                "prompt_key": prompt_key,
                "question": extract_question(prompt_text(teacher)),
                "round": extract_round(prompt_text(teacher)),
                "raw": raw,
                "verifier": payload,
                "error": "",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            record = {
                "prompt_key": prompt_key,
                "question": extract_question(prompt_text(teacher)),
                "round": extract_round(prompt_text(teacher)),
                "raw": "",
                "verifier": {},
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        append_jsonl(verifier_path, record)
        existing[prompt_key] = record
    return existing


def build_teacher_full_chain_pipeline(args: argparse.Namespace, output_dir: Path) -> CPUInferencePipeline:
    from goldenglow.inference import CPUInferencePipeline
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever

    runtime_config_path = resolve_local_path(args.runtime_config, DEFAULT_RUNTIME_CONFIG_PATH) or DEFAULT_RUNTIME_CONFIG_PATH
    runtime_config = load_runtime_config(runtime_config_path)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    index_dir = resolve_local_path(args.index_dir, INDEX_ROOT) or INDEX_ROOT
    device = str(config_value(args.device, retrieval_cfg, "device", "cuda"))
    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True)) and not args.no_reranker
    configured_reranker = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
    reranker_model = (
        resolve_path(args.reranker_model if args.reranker_model is not None else configured_reranker, default=RERANKER_MODEL_DIR)
        if enable_reranker
        else None
    )
    minirag_index = resolve_path(
        args.minirag_index if args.minirag_index is not None else retrieval_cfg.get("minirag_index_path"),
        default=MINIRAG_GRAPH_PATH if bool(retrieval_cfg.get("enable_minirag", True)) else None,
    )
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=int(config_value(args.reranker_max_length, retrieval_cfg, "reranker_max_length", 1024)),
        documents_path=index_dir / "documents.jsonl" if (index_dir / "documents.jsonl").exists() else DOCUMENTS_PATH,
        faiss_index_path=index_dir / "faiss.index" if (index_dir / "faiss.index").exists() else FAISS_INDEX_PATH,
        bm25_tokens_path=index_dir / "bm25_tokens.pkl" if (index_dir / "bm25_tokens.pkl").exists() else BM25_TOKENS_PATH,
        minirag_index_path=minirag_index,
        device=device,
    )
    query_config_args = argparse.Namespace(
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        fusion_top_k=args.fusion_top_k,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        rerank_batch_size=args.rerank_batch_size,
        minirag_top_k=args.minirag_top_k,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights=None,
        minirag_index=args.minirag_index,
        minirag_fusion_mode=args.minirag_fusion_mode,
        minirag_chapter_isolation=args.minirag_chapter_isolation,
        minirag_auto_second_retrieval=args.minirag_auto_second_retrieval,
        minirag_scope_seed_top_k=args.minirag_scope_seed_top_k,
        minirag_expansion_query_top_k=args.minirag_expansion_query_top_k,
        minirag_graph_scope_min_ratio=args.minirag_graph_scope_min_ratio,
        minirag_second_pass_scope_min_ratio=args.minirag_second_pass_scope_min_ratio,
        enable_storyline_sparse_scope=args.enable_storyline_sparse_scope,
        storyline_scope_seed_top_k=args.storyline_scope_seed_top_k,
        storyline_sparse_scope_min_ratio=args.storyline_sparse_scope_min_ratio,
        enable_scoped_chapter_search=args.enable_scoped_chapter_search,
        scoped_chapter_dense_top_k=args.scoped_chapter_dense_top_k,
        scoped_chapter_sparse_top_k=args.scoped_chapter_sparse_top_k,
        enable_neighbor_expansion=args.enable_neighbor_expansion,
        neighbor_max_seed_docs=args.neighbor_max_seed_docs,
        neighbor_story_window=args.neighbor_story_window,
        neighbor_activity_story_sort_window=args.neighbor_activity_story_sort_window,
        enable_same_story_sweep=args.enable_same_story_sweep,
        same_story_sweep_max_seed_docs=args.same_story_sweep_max_seed_docs,
        same_story_sweep_max_docs_per_story=args.same_story_sweep_max_docs_per_story,
        same_story_sweep_extra_candidates=args.same_story_sweep_extra_candidates,
    )
    rerank_top_k = int(config_value(args.rerank_top_k, retrieval_cfg, "rerank_top_k", 32))
    generator = build_teacher_generator(args, output_dir)
    return CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=build_query_config(query_config_args, retrieval_cfg, rerank_top_k=rerank_top_k),
        max_retrieval_rounds=int(config_value(args.max_rounds, inference_cfg, "max_retrieval_rounds", 2)),
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(config_value(args.prompt_evidence_max_chars_per_doc, inference_cfg, "prompt_evidence_max_chars_per_doc", 900)),
        prompt_conclusion_evidence_max_total_chars=int(
            config_value(args.prompt_conclusion_evidence_max_total_chars, inference_cfg, "prompt_conclusion_evidence_max_total_chars", 9000)
        ),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_evidence_pinning=bool(config_value(args.enable_evidence_pinning, inference_cfg, "enable_evidence_pinning", False)),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)),
        self_consistency_samples=1,
        self_consistency_temperature=0.1,
        answer_grounding_mode=str(config_value(args.answer_grounding_mode, inference_cfg, "answer_grounding_mode", "weak")),
        conclusion_prompt_mode=str(config_value(args.conclusion_prompt_mode, inference_cfg, "conclusion_prompt_mode", "minimal")),
        use_model_hypothesis=bool(inference_cfg.get("use_model_hypothesis", True)),
        use_model_conclusion_generation=bool(inference_cfg.get("use_model_conclusion_generation", True)),
        web_context_config=inference_cfg.get("web_context") if isinstance(inference_cfg.get("web_context"), dict) else None,
    )


def run_teacher_full_chain(args: argparse.Namespace, output_dir: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    output_path = output_dir / "teacher_full_chain.jsonl"
    existing_questions: set[str] = set()
    if output_path.exists() and not args.overwrite_teacher_full_chain:
        with output_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if payload.get("question"):
                        existing_questions.add(str(payload["question"]))
    questions: list[str] = []
    seen: set[str] = set()
    for group in groups.values():
        record = group[0]
        question = extract_question(prompt_text(record))
        if question and question not in seen and question not in existing_questions:
            questions.append(question)
            seen.add(question)
    if args.max_teacher_full_chain_questions is not None:
        questions = questions[: max(0, args.max_teacher_full_chain_questions)]
    if not questions:
        return
    pipeline = build_teacher_full_chain_pipeline(args, output_dir)
    progress = tqdm(questions, desc="teacher full-chain", unit="question")
    for question in progress:
        started = time.perf_counter()
        try:
            result = pipeline.run(question)
            payload = asdict(result)
            payload["error"] = ""
            payload["elapsed_sec"] = round(time.perf_counter() - started, 3)
        except Exception as exc:
            payload = {
                "question": question,
                "answer": "",
                "retrieval_trace": [],
                "evidence": [],
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        append_jsonl(output_path, payload)


def build_dataset(args: argparse.Namespace, output_dir: Path, groups: dict[str, list[dict[str, Any]]], verifier_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    stats: Counter = Counter()
    output_records: list[dict[str, Any]] = []
    for prompt_key, group in groups.items():
        teacher = record_for_role(group, kto_tag=True)
        if teacher is None:
            stats["drop:no_teacher"] += 1
            continue
        if infer_task_type(teacher) != "conclusion_generation":
            output_records.extend(group)
            stats["keep:non_conclusion"] += len(group)
            continue
        verifier_record = verifier_records.get(prompt_key)
        if not verifier_record or verifier_record.get("error"):
            if args.keep_unverified_conclusion:
                output_records.extend(group)
                stats["keep:unverified_conclusion_original"] += len(group)
            else:
                stats["drop:unverified_conclusion"] += len(group)
            continue
        output_records.extend(relabel_group(prompt_key, group, verifier_record, stats))
    train, val = split_records(output_records, seed=args.seed, val_ratio=args.val_ratio)
    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "prompt_groups": len(groups),
        "verifier_records": len(verifier_records),
        "output_records": len(output_records),
        "train_records": len(train),
        "val_records": len(val),
        "stats": dict(sorted(stats.items())),
    }
    write_json(output_dir / "build_summary.json", summary)
    report = [
        "# SODA API Verifier Build Report",
        "",
        f"- prompt_groups: {len(groups)}",
        f"- verifier_records: {len(verifier_records)}",
        f"- output_records: {len(output_records)}",
        f"- train_records: {len(train)}",
        f"- val_records: {len(val)}",
        "",
        "## Stats",
        "",
    ]
    report.extend(f"- {key}: {value}" for key, value in sorted(stats.items()))
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API evidence-only verifier on existing SODA student-state rollouts and build relabeled KTO data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1")
    parser.add_argument("--teacher-runtime-config", type=Path, default=DEFAULT_TEACHER_RUNTIME_CONFIG)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--max-verifier-prompts", type=int, default=None)
    parser.add_argument("--verifier-max-tokens", type=int, default=2048)
    parser.add_argument("--keep-unverified-conclusion", action="store_true")
    parser.add_argument("--overwrite-verifier", action="store_true")
    parser.add_argument("--run-teacher-full-chain", action="store_true")
    parser.add_argument("--overwrite-teacher-full-chain", action="store_true")
    parser.add_argument("--max-teacher-full-chain-questions", type=int, default=None)
    parser.add_argument("--save-api-request-logs", action="store_true")

    parser.add_argument("--teacher-backend", choices=("chat_completions", "openai_compatible_api", "responses_api", "responses"), default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-timeout", type=float, default=None)
    parser.add_argument("--teacher-max-tokens", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-top-p", type=float, default=0.8)
    parser.add_argument("--no-json-response-format", action="store_true")

    parser.add_argument("--device", default=None)
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--reranker-max-length", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--minirag-top-k", type=int, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--minirag-fusion-mode", choices=("score", "append"), default=None)
    parser.add_argument("--enable-minirag-chapter-isolation", dest="minirag_chapter_isolation", action="store_true", default=None)
    parser.add_argument("--disable-minirag-chapter-isolation", dest="minirag_chapter_isolation", action="store_false")
    parser.add_argument("--enable-minirag-auto-second-retrieval", dest="minirag_auto_second_retrieval", action="store_true", default=None)
    parser.add_argument("--disable-minirag-auto-second-retrieval", dest="minirag_auto_second_retrieval", action="store_false")
    parser.add_argument("--minirag-scope-seed-top-k", type=int, default=None)
    parser.add_argument("--minirag-expansion-query-top-k", type=int, default=None)
    parser.add_argument("--minirag-graph-scope-min-ratio", type=float, default=None)
    parser.add_argument("--minirag-second-pass-scope-min-ratio", type=float, default=None)
    parser.add_argument("--enable-storyline-sparse-scope", dest="enable_storyline_sparse_scope", action="store_true", default=None)
    parser.add_argument("--disable-storyline-sparse-scope", dest="enable_storyline_sparse_scope", action="store_false")
    parser.add_argument("--storyline-scope-seed-top-k", type=int, default=None)
    parser.add_argument("--storyline-sparse-scope-min-ratio", type=float, default=None)
    parser.add_argument("--enable-scoped-chapter-search", dest="enable_scoped_chapter_search", action="store_true", default=None)
    parser.add_argument("--disable-scoped-chapter-search", dest="enable_scoped_chapter_search", action="store_false")
    parser.add_argument("--scoped-chapter-dense-top-k", type=int, default=None)
    parser.add_argument("--scoped-chapter-sparse-top-k", type=int, default=None)
    parser.add_argument("--enable-neighbor-expansion", action="store_true", default=None)
    parser.add_argument("--disable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_false")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=None)
    parser.add_argument("--neighbor-story-window", type=int, default=None)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=None)
    parser.add_argument("--enable-same-story-sweep", dest="enable_same_story_sweep", action="store_true", default=None)
    parser.add_argument("--disable-same-story-sweep", dest="enable_same_story_sweep", action="store_false")
    parser.add_argument("--same-story-sweep-max-seed-docs", type=int, default=None)
    parser.add_argument("--same-story-sweep-max-docs-per-story", type=int, default=None)
    parser.add_argument("--same-story-sweep-extra-candidates", type=int, default=None)
    parser.add_argument("--prompt-evidence-top-k", type=int, default=None)
    parser.add_argument("--prompt-evidence-max-chars-per-doc", type=int, default=None)
    parser.add_argument("--prompt-conclusion-evidence-max-total-chars", type=int, default=None)
    parser.add_argument("--enable-mmr", action="store_true", default=None)
    parser.add_argument("--disable-mmr", dest="enable_mmr", action="store_false")
    parser.add_argument("--mmr-lambda", type=float, default=None)
    parser.add_argument("--enable-pyramid-order", action="store_true", default=None)
    parser.add_argument("--disable-pyramid-order", dest="enable_pyramid_order", action="store_false")
    parser.add_argument("--enable-evidence-pinning", action="store_true", default=None)
    parser.add_argument("--disable-evidence-pinning", dest="enable_evidence_pinning", action="store_false")
    parser.add_argument("--enable-crag-refinement", action="store_true", default=None)
    parser.add_argument("--disable-crag-refinement", dest="enable_crag_refinement", action="store_false")
    parser.add_argument("--crag-refine-top-sentences", type=int, default=None)
    parser.add_argument("--crag-refine-max-sentences", type=int, default=None)
    parser.add_argument("--conclusion-prompt-mode", choices=("full", "minimal"), default=None)
    parser.add_argument("--answer-grounding-mode", choices=("off", "weak", "strict"), default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_audit_groups(args.input_dir)
    verifier_records = run_verifier(args, output_dir, groups)
    if args.run_teacher_full_chain:
        run_teacher_full_chain(args, output_dir, groups)
    summary = build_dataset(args, output_dir, groups, verifier_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
