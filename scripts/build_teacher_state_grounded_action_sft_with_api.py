#!/usr/bin/env python3
"""Build teacher-state grounded-action SFT data without student rollout.

This script is intended for cold-starting a raw 4B model. It does not use the
student model output. For each question it:
1. asks the API teacher to build the first retrieval hypothesis;
2. runs the local retriever for round 1;
3. asks the API teacher/verifier to choose answer_directly/retrieve_more/abstain
   based only on the current evidence;
4. if needed, runs one follow-up retrieval round;
5. writes ShareGPT SFT records in grounded_action_v1 format.

No missing_slots or clarify_user fields are emitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
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
    RERANKER_MODEL_DIR,
)
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    COMMON_NON_ENTITY_WORDS,
    CPUInferencePipeline,
    HypothesisDocument,
    NOISY_RETRIEVAL_TOKENS,
    PRONOUN_REFERENCES,
    build_follow_up_hypothesis_queries,
    build_hypothesis_prompt,
    build_retrieval_query,
    expand_queries_with_main_chapter_terms,
    extract_json_object,
    merge_evidence_keep_order,
    merge_hypotheses,
    normalize_hypothesis_payload,
    render_evidence_blocks,
    render_minirag_hints_for_prompt,
    render_short_evidence_brief,
    repair_json_like_output,
    summarize_evidence_for_trace,
    _extract_content_tokens,
    _is_entity_candidate,
    _resolve_referential_question,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.evaluate_multiround_retrieval_recall import (  # noqa: E402
    DEFAULT_RUNTIME_CONFIG_PATH,
    build_query_config,
    config_value,
    load_runtime_config,
    parse_mode_weights,
    resolve_path,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/teacher_state_grounded_action_sft_v1"
DEFAULT_TEACHER_RUNTIME_CONFIG = PROJECT_ROOT / "api-mode/runtime_deepseek_api.json"

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
SYSTEM_PROMPT = "你是《明日方舟》剧情 RAG 的证据约束动作与回答模块。只输出 JSON。"
ALLOWED_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}
QUERY_TYPES = {"fact", "relation", "causality", "reasoning", "reveal", "mystery", "answerability"}
FOLLOW_UP_FIELDS = ("question", "query_type", "entities", "keywords", "expected_answer_type", "dialogue_context")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"([。！？；;!?])")
DEFAULT_MAX_QUOTE_CHARS = 80
DEFAULT_MAX_QUOTES_PER_FACT = 2
DEFAULT_MAX_FACT_QUOTE_TOTAL_CHARS = 160
DEFAULT_MAX_SUPPORTED_FACTS = 6
DEFAULT_MAX_ANSWER_QUOTE_TOTAL_CHARS = 400
HIGH_RISK_GROUNDING_TERMS = {
    "开发",
    "制造",
    "建造",
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
    "真正原因",
    "幕后主使",
    "亲生",
    "父亲",
    "母亲",
    "未婚夫",
    "未婚妻",
}


class DummyGenerator:
    max_tokens = 4096

    def describe_runtime(self) -> dict[str, Any]:
        return {"generator_backend": "none", "runtime_mode": "teacher_state_sft_no_student"}

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("DummyGenerator cannot generate. This script uses API teacher calls directly.")


def stable_key(*parts: str) -> str:
    return hashlib.sha1("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_local_path(path: str | Path | None, default: Path | None = None) -> Path | None:
    selected = Path(path) if path not in (None, "") else default
    if selected in (None, ""):
        return None
    assert selected is not None
    return selected if selected.is_absolute() else PROJECT_ROOT / selected


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_jsonish(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    block = JSON_BLOCK_RE.search(raw)
    if block:
        raw = block.group(1).strip()
    raw = repair_json_like_output(raw)
    payload = extract_json_object(raw)
    if isinstance(payload, dict):
        return payload
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def normalize_for_match(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text or ""))


def load_api_mode_module() -> Any:
    module_path = PROJECT_ROOT / "api-mode/run_api_inference.py"
    spec = importlib.util.spec_from_file_location("goldenglow_api_mode_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_known_api_key_prefix(api_key: str) -> str:
    key = str(api_key or "").strip()
    if key.startswith("ds:sk-"):
        return key.split(":", 1)[1]
    return key


def build_teacher_generator(args: argparse.Namespace, output_dir: Path) -> Any:
    teacher_runtime = load_runtime_config(
        resolve_local_path(args.teacher_runtime_config, DEFAULT_TEACHER_RUNTIME_CONFIG) or DEFAULT_TEACHER_RUNTIME_CONFIG
    )
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
        response_format_json=bool(generator_cfg.get("response_format_json", True)) and not args.no_json_response_format,
        extra_body=generator_cfg.get("extra_body") if isinstance(generator_cfg.get("extra_body"), dict) else None,
        request_log_dir=request_log_dir,
    )


def load_questions(args: argparse.Namespace) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if args.questions_file:
        path = resolve_local_path(args.questions_file)
        if path is None or not path.exists():
            raise SystemExit(f"Missing questions file: {args.questions_file}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise SystemExit(f"Questions JSON must be a list: {path}")
            for item in payload:
                if isinstance(item, str):
                    question = item.strip()
                    if question:
                        items.append({"question": question, "question_key": stable_key("json", question)})
                elif isinstance(item, dict):
                    question = str(item.get("question") or item.get("query") or "").strip()
                    if question:
                        items.append({"question": question, "question_key": str(item.get("question_key") or stable_key("json", question))})
        else:
            for line in text.splitlines():
                if not line.strip():
                    continue
                if path.suffix.lower() == ".jsonl":
                    payload = json.loads(line)
                    question = str(payload.get("question") or payload.get("query") or "").strip()
                    if question:
                        items.append({"question": question, "question_key": str(payload.get("question_key") or stable_key("jsonl", question))})
                else:
                    question = line.strip()
                    items.append({"question": question, "question_key": stable_key("txt", question)})
    for question in args.question or []:
        question = str(question).strip()
        if question:
            items.append({"question": question, "question_key": stable_key("inline", question)})
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    return items


def clean_string_list(value: Any, *, limit: int) -> list[str]:
    if isinstance(value, str):
        raw_items = re.split(r"[、,，;；]\s*", value)
    elif isinstance(value, list):
        raw_items = value
    else:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def sanitize_follow_up_hypothesis(payload: Any, *, question: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    query_type = str(payload.get("query_type") or "reasoning").strip()
    if query_type not in QUERY_TYPES:
        query_type = "reasoning"
    follow_up = {
        "question": str(payload.get("question") or question).strip(),
        "query_type": query_type,
        "entities": clean_string_list(payload.get("entities"), limit=12),
        "keywords": clean_string_list(payload.get("keywords"), limit=24),
        "expected_answer_type": str(payload.get("expected_answer_type") or "剧情问答").strip(),
        "dialogue_context": str(payload.get("dialogue_context") or "").strip(),
    }
    if not follow_up["question"]:
        follow_up["question"] = question
    if not follow_up["entities"] and not follow_up["keywords"]:
        return None
    return follow_up


def validate_grounded_answer(
    payload: dict[str, Any],
    *,
    allowed_evidence: str,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_fact_quote_total_chars: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    supported = payload.get("supported_facts")
    if not isinstance(supported, list) or not supported:
        return False, ["answer_directly requires non-empty supported_facts"]
    if max_supported_facts > 0 and len(supported) > max_supported_facts:
        errors.append(f"too many supported_facts ({len(supported)}>{max_supported_facts})")
    evidence_norm = normalize_for_match(allowed_evidence)
    quote_texts: list[str] = []
    for fact_index, fact in enumerate(supported, start=1):
        if not isinstance(fact, dict):
            errors.append(f"supported_facts[{fact_index}] is not object")
            continue
        if not str(fact.get("fact") or "").strip():
            errors.append(f"supported_facts[{fact_index}].fact is empty")
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"supported_facts[{fact_index}].evidence_refs is empty")
            continue
        if max_quotes_per_fact > 0 and len(refs) > max_quotes_per_fact:
            errors.append(f"supported_facts[{fact_index}] has too many quotes ({len(refs)}>{max_quotes_per_fact})")
        fact_quote_total = 0
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                errors.append(f"evidence_refs[{fact_index}.{ref_index}] is not object")
                continue
            quote = str(ref.get("quote") or "").strip()
            if not quote:
                errors.append(f"evidence_refs[{fact_index}.{ref_index}].quote is empty")
            elif max_quote_chars > 0 and len(quote) > max_quote_chars:
                errors.append(f"quote too long ({len(quote)}>{max_quote_chars}): {quote[:80]}")
            elif normalize_for_match(quote) not in evidence_norm:
                errors.append(f"quote not found in allowed_evidence: {quote[:80]}")
            else:
                quote_texts.append(quote)
                fact_quote_total += len(quote)
        if max_fact_quote_total_chars > 0 and fact_quote_total > max_fact_quote_total_chars:
            errors.append(
                f"supported_facts[{fact_index}] quote total too long ({fact_quote_total}>{max_fact_quote_total_chars})"
            )
    answer_quote_total = sum(len(quote) for quote in quote_texts)
    if max_answer_quote_total_chars > 0 and answer_quote_total > max_answer_quote_total_chars:
        errors.append(f"answer quote total too long ({answer_quote_total}>{max_answer_quote_total_chars})")
    if not str(payload.get("final_answer") or "").strip():
        errors.append("final_answer is empty")
    inferred = payload.get("inferred_facts")
    if inferred is None:
        payload["inferred_facts"] = []
    elif not isinstance(inferred, list):
        errors.append("inferred_facts must be a list")

    quote_pool = normalize_for_match("\n".join(quote_texts))
    question_tokens = set(_extract_content_tokens(str(payload.get("question") or "")))

    def unsupported_api_or_acronym_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        for token in _extract_content_tokens(text):
            if (
                token in question_tokens
                or token in COMMON_NON_ENTITY_WORDS
                or token in NOISY_RETRIEVAL_TOKENS
                or token in PRONOUN_REFERENCES
                or not _is_entity_candidate(token)
            ):
                continue
            if not token.isascii():
                continue
            if len(token) < 3:
                continue
            token_norm = normalize_for_match(token)
            if token_norm and token_norm not in quote_pool:
                tokens.append(token)
        return list(dict.fromkeys(tokens))

    for fact_index, fact in enumerate(supported, start=1):
        if isinstance(fact, dict):
            missing = unsupported_api_or_acronym_tokens(str(fact.get("fact") or ""))
            if missing:
                errors.append(f"supported_facts[{fact_index}].fact has API/acronym tokens not supported by quotes: {','.join(missing[:8])}")

    final_missing = unsupported_api_or_acronym_tokens(str(payload.get("final_answer") or ""))
    if final_missing:
        errors.append(f"final_answer has API/acronym tokens not supported by quotes: {','.join(final_missing[:10])}")

    for term in HIGH_RISK_GROUNDING_TERMS:
        if term in str(payload.get("final_answer") or "") and normalize_for_match(term) not in quote_pool:
            errors.append(f"high-risk term not supported by quotes: {term}")
        for fact_index, fact in enumerate(supported, start=1):
            if isinstance(fact, dict) and term in str(fact.get("fact") or "") and normalize_for_match(term) not in quote_pool:
                errors.append(f"supported_facts[{fact_index}] high-risk term not supported by quotes: {term}")
    return not errors, errors


def split_quote_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in SENTENCE_SPLIT_RE.finditer(str(text or "")):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
        start = end
    tail = str(text or "")[start:].strip()
    if tail:
        parts.append(tail)
    return parts or ([str(text or "").strip()] if str(text or "").strip() else [])


def repair_quote_span(quote: str, *, allowed_evidence: str, max_quote_chars: int) -> str | None:
    quote = str(quote or "").strip()
    if not quote:
        return None
    evidence_norm = normalize_for_match(allowed_evidence)
    quote_norm = normalize_for_match(quote)
    if not quote_norm or quote_norm not in evidence_norm:
        return None
    if max_quote_chars <= 0 or len(quote) <= max_quote_chars:
        return quote
    for sentence in split_quote_sentences(quote):
        if 0 < len(sentence) <= max_quote_chars and normalize_for_match(sentence) in evidence_norm:
            return sentence
    window = min(max_quote_chars, 60)
    step = max(1, window // 3)
    for start in range(0, max(1, len(quote) - window + 1), step):
        candidate = quote[start : start + window].strip()
        if candidate and normalize_for_match(candidate) in evidence_norm:
            return candidate
    return None


def discard_unfixable_quotes(
    payload: dict[str, Any],
    *,
    allowed_evidence: str,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or str(payload.get("next_action") or "") != "answer_directly":
        return payload if isinstance(payload, dict) else None
    supported = payload.get("supported_facts")
    if not isinstance(supported, list):
        return None

    output = dict(payload)
    repaired_facts: list[dict[str, Any]] = []
    total_quote_chars = 0
    for fact in supported:
        if len(repaired_facts) >= max_supported_facts:
            break
        if not isinstance(fact, dict):
            continue
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list):
            continue
        repaired_refs: list[dict[str, str]] = []
        fact_quote_total = 0
        seen_quotes: set[str] = set()
        for ref in refs:
            if max_quotes_per_fact > 0 and len(repaired_refs) >= max_quotes_per_fact:
                break
            if not isinstance(ref, dict):
                continue
            repaired_quote = repair_quote_span(str(ref.get("quote") or ""), allowed_evidence=allowed_evidence, max_quote_chars=max_quote_chars)
            if repaired_quote is None:
                continue
            quote_norm = normalize_for_match(repaired_quote)
            if quote_norm in seen_quotes:
                continue
            if max_answer_quote_total_chars > 0 and total_quote_chars + len(repaired_quote) > max_answer_quote_total_chars:
                continue
            seen_quotes.add(quote_norm)
            repaired_refs.append({"evidence_id": str(ref.get("evidence_id") or "").strip(), "quote": repaired_quote})
            fact_quote_total += len(repaired_quote)
            total_quote_chars += len(repaired_quote)
        if repaired_refs and fact_quote_total > 0:
            repaired_fact = dict(fact)
            repaired_fact["evidence_refs"] = repaired_refs
            repaired_facts.append(repaired_fact)
    if not repaired_facts:
        return None
    output["supported_facts"] = repaired_facts
    return output


def normalize_action_payload(
    payload: dict[str, Any] | None,
    *,
    question: str,
    round_index: int,
    max_rounds: int,
    allowed_evidence: str,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_fact_quote_total_chars: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(payload, dict):
        return None, ["payload is not object"]
    action = str(payload.get("next_action") or "").strip()
    if action not in ALLOWED_ACTIONS:
        return None, [f"invalid next_action: {action or '<empty>'}"]
    if action == "retrieve_more" and round_index >= max_rounds:
        return None, ["retrieve_more is not allowed at max round"]

    if action == "answer_directly":
        output = {
            "question": question,
            "next_action": "answer_directly",
            "follow_up_hypothesis": None,
            "supported_facts": payload.get("supported_facts") or [],
            "inferred_facts": payload.get("inferred_facts") or [],
            "final_answer": str(payload.get("final_answer") or "").strip(),
        }
        ok, errors = validate_grounded_answer(
            output,
            allowed_evidence=allowed_evidence,
            max_quote_chars=max_quote_chars,
            max_quotes_per_fact=max_quotes_per_fact,
            max_fact_quote_total_chars=max_fact_quote_total_chars,
            max_supported_facts=max_supported_facts,
            max_answer_quote_total_chars=max_answer_quote_total_chars,
        )
        return (output if ok else None), errors

    if action == "retrieve_more":
        follow_up = sanitize_follow_up_hypothesis(payload.get("follow_up_hypothesis"), question=question)
        if follow_up is None:
            return None, ["retrieve_more requires valid follow_up_hypothesis"]
        return {
            "question": question,
            "next_action": "retrieve_more",
            "follow_up_hypothesis": follow_up,
            "supported_facts": [],
            "inferred_facts": [],
            "final_answer": "",
        }, []

    final_answer = str(payload.get("final_answer") or payload.get("answer") or "现有证据不足以确认。").strip()
    return {
        "question": question,
        "next_action": "abstain",
        "follow_up_hypothesis": None,
        "supported_facts": [],
        "inferred_facts": [],
        "final_answer": final_answer or "现有证据不足以确认。",
    }, []


def dataset_info(dataset_name: str) -> dict[str, Any]:
    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": ROLE_TAGS,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    if len(shuffled) <= 10 or val_ratio <= 0:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_count:], shuffled[:val_count]


def build_retrieval_pipeline(args: argparse.Namespace, output_dir: Path) -> tuple[CPUInferencePipeline, dict[str, Any]]:
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
    print(f"[load] retriever index={index_dir} device={device}", flush=True)
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
    rerank_top_k = int(config_value(args.rerank_top_k, retrieval_cfg, "rerank_top_k", 32))
    query_config = build_query_config(args, retrieval_cfg, rerank_top_k=rerank_top_k)
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=DummyGenerator(),
        query_config=query_config,
        max_retrieval_rounds=int(config_value(args.max_rounds, inference_cfg, "max_retrieval_rounds", 2)),
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(
            config_value(args.prompt_evidence_max_chars_per_doc, inference_cfg, "prompt_evidence_max_chars_per_doc", 1800)
        ),
        prompt_conclusion_evidence_max_total_chars=int(
            config_value(args.prompt_conclusion_evidence_max_total_chars, inference_cfg, "prompt_conclusion_evidence_max_total_chars", 24000)
        ),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_evidence_pinning=bool(config_value(args.enable_evidence_pinning, inference_cfg, "enable_evidence_pinning", False)),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)),
        answer_grounding_mode=str(config_value(args.answer_grounding_mode, inference_cfg, "answer_grounding_mode", "weak")),
        conclusion_prompt_mode="minimal",
        web_context_config={"enabled": False},
    )
    metadata = {
        "runtime_config": str(runtime_config_path),
        "query_config": asdict(query_config),
        "inference_config": {
            "max_retrieval_rounds": pipeline.max_retrieval_rounds,
            "prompt_evidence_top_k": pipeline.prompt_evidence_top_k,
            "prompt_evidence_max_chars_per_doc": pipeline.prompt_evidence_max_chars_per_doc,
            "prompt_conclusion_evidence_max_total_chars": pipeline.prompt_conclusion_evidence_max_total_chars,
            "web_context_enabled": False,
        },
    }
    write_json(output_dir / "aligned_runtime.json", metadata)
    return pipeline, metadata


def build_initial_queries(question: str, hypothesis: HypothesisDocument) -> list[str]:
    queries = [
        _resolve_referential_question(question, hypothesis.entities),
        build_retrieval_query(hypothesis),
    ]
    queries.extend(build_follow_up_hypothesis_queries(question, hypothesis))
    return expand_queries_with_main_chapter_terms(queries)


def retrieve_round(
    *,
    pipeline: CPUInferencePipeline,
    question: str,
    hypothesis: HypothesisDocument,
    pending_queries: list[str],
    round_index: int,
    scope_state: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    minirag_expansion_record: dict[str, Any] | None = None
    if (
        round_index == 1
        and pipeline.query_config.minirag_chapter_isolation
        and pipeline.query_config.minirag_auto_second_retrieval
    ):
        _, _, evidence, minirag_expansion_record = pipeline._retrieve_first_round_with_scoped_minirag_expansion(
            question,
            hypothesis,
            pending_queries,
        )
        if minirag_expansion_record is not None:
            retained_chapter_scope = str(minirag_expansion_record.get("chapter_scope") or "").strip() or None
            retained_storyline_scope = str(minirag_expansion_record.get("storyline_scope") or "").strip() or None
            scope_state["retained_chapter_scope"] = retained_chapter_scope
            scope_state["retained_storyline_scope"] = retained_storyline_scope
            scope_state["scope_retention_enabled"] = bool(
                minirag_expansion_record.get("use_scoped_candidates") and retained_chapter_scope
            )
    else:
        _, _, evidence = pipeline._retrieve_round(
            question,
            hypothesis,
            pending_queries,
            minirag_chapter_scope=scope_state.get("retained_chapter_scope") if scope_state.get("scope_retention_enabled") else None,
            candidate_chapter_scope=scope_state.get("retained_chapter_scope") if scope_state.get("scope_retention_enabled") else None,
            sparse_storyline_scope=scope_state.get("retained_storyline_scope") if scope_state.get("scope_retention_enabled") else None,
        )
    if scope_state.get("scope_retention_enabled") and scope_state.get("retained_scope_evidence") and round_index > 1:
        evidence = merge_evidence_keep_order(
            scope_state["retained_scope_evidence"],
            evidence,
            limit=max(pipeline.query_config.reranker_candidate_top_k, pipeline.prompt_evidence_top_k * 2),
        )
    if round_index == 1 and scope_state.get("scope_retention_enabled"):
        scope_state["retained_scope_evidence"] = list(evidence)
    prompt_evidence = pipeline.prepare_prompt_evidence(question, hypothesis, evidence)
    trace = {
        "round": round_index,
        "queries": list(pending_queries),
        "hypothesis": asdict(hypothesis),
        "evidence_summary": summarize_evidence_for_trace(evidence),
        "prompt_evidence_summary": summarize_evidence_for_trace(prompt_evidence),
        "retained_chapter_scope": scope_state.get("retained_chapter_scope") or "",
        "retained_storyline_scope": scope_state.get("retained_storyline_scope") or "",
        "scope_retention_enabled": bool(scope_state.get("scope_retention_enabled")),
    }
    if minirag_expansion_record is not None:
        trace["minirag_chapter_expansion"] = minirag_expansion_record
    return evidence, prompt_evidence, trace


def evidence_doc_id(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    return str(doc.get("id") or item.get("doc_index") or "").strip()


def prefer_novel_prompt_evidence(
    prompt_evidence: list[dict[str, Any]],
    *,
    seen_doc_ids: set[str],
    min_new: int,
) -> list[dict[str, Any]]:
    if not seen_doc_ids or min_new <= 0:
        return prompt_evidence
    novel: list[dict[str, Any]] = []
    repeated: list[dict[str, Any]] = []
    for item in prompt_evidence:
        doc_id = evidence_doc_id(item)
        if doc_id and doc_id in seen_doc_ids:
            repeated.append(item)
        else:
            novel.append(item)
    if len(novel) < min_new:
        return prompt_evidence
    return [*novel, *repeated][: len(prompt_evidence)]


def build_novel_first_prompt_evidence(
    *,
    pipeline: CPUInferencePipeline,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    seen_doc_ids: set[str],
    min_new: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not seen_doc_ids or min_new <= 0:
        prompt_evidence = pipeline.prepare_prompt_evidence(question, hypothesis, evidence)
        return prompt_evidence, {"novel_mode": "disabled"}
    novel = [item for item in evidence if evidence_doc_id(item) not in seen_doc_ids]
    repeated = [item for item in evidence if evidence_doc_id(item) in seen_doc_ids]
    if len(novel) < min_new:
        prompt_evidence = pipeline.prepare_prompt_evidence(question, hypothesis, evidence)
        return prompt_evidence, {
            "novel_mode": "fallback_not_enough_new_evidence",
            "novel_candidates": len(novel),
            "repeated_candidates": len(repeated),
        }
    novel_prompt = pipeline.prepare_prompt_evidence(question, hypothesis, novel)
    remaining = max(0, pipeline.prompt_evidence_top_k - len(novel_prompt))
    if remaining > 0 and repeated:
        repeated_prompt = pipeline.prepare_prompt_evidence(question, hypothesis, repeated)[:remaining]
    else:
        repeated_prompt = []
    prompt_evidence = [*novel_prompt, *repeated_prompt]
    return prompt_evidence, {
        "novel_mode": "novel_first",
        "novel_candidates": len(novel),
        "repeated_candidates": len(repeated),
        "novel_prompt_count": len(novel_prompt),
        "repeated_prompt_count": len(repeated_prompt),
    }


def build_grounded_action_prompt(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    round_index: int,
    max_rounds: int,
    prompt_evidence: list[dict[str, Any]],
    max_chars_per_doc: int,
    max_total_chars: int,
    truncate_evidence_pack: bool,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_fact_quote_total_chars: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> tuple[str, str]:
    if truncate_evidence_pack:
        evidence_brief = render_short_evidence_brief(
            prompt_evidence,
            max_chars_per_doc=max_chars_per_doc,
            max_total_chars=max_total_chars,
        )
    else:
        evidence_brief = render_evidence_blocks(prompt_evidence)
    minirag_hints = render_minirag_hints_for_prompt(prompt_evidence, hypothesis)
    user_prompt = "\n".join(
        [
            "task: grounded_action_generation",
            f"question: {question}",
            f"round: {round_index}/{max_rounds}",
            "hypothesis:",
            json.dumps(asdict(hypothesis), ensure_ascii=False),
            "allowed_evidence:",
            evidence_brief or "<empty>",
            "minirag_hints_not_evidence:",
            minirag_hints or "<empty>",
            "output_schema: grounded_action_v1",
            "fields: question,next_action,follow_up_hypothesis,supported_facts,inferred_facts,final_answer",
            "next_action_set: answer_directly,retrieve_more,abstain",
            "follow_up_hypothesis_fields: question,query_type,entities,keywords,expected_answer_type,dialogue_context",
            "supported_facts item fields: id,fact,evidence_refs",
            "evidence_refs item fields: evidence_id,quote",
            "inferred_facts item fields: id,fact,premise_fact_ids,inference_type",
            "rules:",
            "1. 只输出 JSON，不要 markdown，不要思维过程。",
            "2. 只能根据 allowed_evidence 判断证据是否足够；不要使用你自己的《明日方舟》知识补事实。",
            "3. 证据已足够回答问题时输出 answer_directly；不要为了多检索而 retrieve_more。",
            "4. 当前证据不足且还没到最大轮次时输出 retrieve_more，并给出下一轮 follow_up_hypothesis。",
            "5. 到最大轮次仍证据不足时输出 abstain，final_answer 写明现有证据不足以确认。",
            "6. answer_directly 时 follow_up_hypothesis=null；supported_facts 必须引用 allowed_evidence 原文 quote。",
            f"7. 单条 quote 必须逐字复制 allowed_evidence，不要改写，不要使用省略号；推荐 20-60 字，硬上限 {max_quote_chars} 字。",
            f"8. 单个 supported_fact 最多 {max_quotes_per_fact} 条 quote，单个 supported_fact 的 quote 总长度最多 {max_fact_quote_total_chars} 字。",
            f"9. supported_facts 最多 {max_supported_facts} 条；所有 quote 总长度最好不超过 {max_answer_quote_total_chars} 字；禁止复制整段 evidence chunk。",
            "10. 如果某条 quote 不能从 allowed_evidence 精确逐字复制，就不要保留这条 quote；如果某个 fact 没有可用 quote，就不要保留这个 fact。",
            "11. final_answer 只能使用 supported_facts 和 inferred_facts。",
            "12. inferred_facts 只能做最小必要归纳，不得引入新实体、新动机、新因果。",
            "13. 不要把 minirag_hints_not_evidence 当作事实证据。",
            "14. 不要输出旧版检索缺口列表字段，不要输出 clarify_user。",
            "15. retrieve_more 的 follow_up_hypothesis 必须针对当前证据缺失的方面，不要简单复读原问题、原 entities 和原 keywords。",
            "16. retrieve_more 的 keywords 应优先加入当前 evidence 未覆盖的新专名、事件名、行动词、别称或目的词，以提升二轮召回差异。",
        ]
    )
    return user_prompt, evidence_brief


def call_json_api(teacher: Any, prompt: str, *, max_tokens: int, temperature: float = 0.1) -> dict[str, Any] | None:
    raw = teacher.generate(prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.8, repeat_penalty=1.0)
    return parse_jsonish(raw)


def coerce_scalar_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    output = dict(payload)
    for field in fields:
        value = output.get(field)
        if isinstance(value, list):
            selected = next((str(item).strip() for item in value if str(item).strip()), "")
            if selected:
                output[field] = selected
    return output


def repair_action_payload(
    teacher: Any,
    *,
    original_prompt: str,
    invalid_payload: dict[str, Any] | None,
    errors: list[str],
    max_tokens: int,
    temperature: float,
    max_quote_chars: int,
    max_quotes_per_fact: int,
    max_fact_quote_total_chars: int,
    max_supported_facts: int,
    max_answer_quote_total_chars: int,
) -> dict[str, Any] | None:
    repair_prompt = "\n".join(
        [
            original_prompt,
            "",
            "校验失败，需要重写 JSON。",
            "上一版 JSON:",
            json.dumps(invalid_payload or {}, ensure_ascii=False),
            "校验错误:",
            json.dumps(errors, ensure_ascii=False),
            "修复要求:",
            f"1. 如果 next_action=answer_directly，每个 quote 必须从 allowed_evidence 逐字复制短句，不要使用省略号；推荐 20-60 字，硬上限 {max_quote_chars} 字。",
            f"2. 单个 supported_fact 最多 {max_quotes_per_fact} 条 quote，quote 总长度最多 {max_fact_quote_total_chars} 字。",
            f"3. supported_facts 最多 {max_supported_facts} 条；所有 quote 总长度最好不超过 {max_answer_quote_total_chars} 字；禁止复制整段 evidence chunk。",
            "4. 无法精确逐字定位的 quote 必须删除；删除后没有 quote 的 fact 也必须删除。",
            "5. 如果无法找到逐字 quote 支撑答案，改为 retrieve_more；若已到最大轮次则改为 abstain。",
            "6. 只输出修复后的 grounded_action_v1 JSON。",
        ]
    )
    return call_json_api(teacher, repair_prompt, max_tokens=max_tokens, temperature=temperature)


def teacher_hypothesis(teacher: Any, question: str, dialogue_context: str) -> HypothesisDocument:
    prompt = build_hypothesis_prompt(question, dialogue_context)
    payload = call_json_api(teacher, prompt, max_tokens=512, temperature=0.1)
    if payload is None:
        raise RuntimeError("teacher returned invalid hypothesis JSON")
    payload = coerce_scalar_fields(payload, ("intent", "query_type", "expected_answer_type"))
    return normalize_hypothesis_payload(payload, question=question, dialogue_context=dialogue_context)


def sft_record(
    *,
    record_id: str,
    user_prompt: str,
    assistant_payload: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": "grounded_action_generation",
        "bucket": "teacher_state_sft",
        "system": SYSTEM_PROMPT,
        "tools": [],
        "conversations": [
            {"from": "human", "value": user_prompt},
            {"from": "gpt", "value": compact_json(assistant_payload)},
        ],
        "meta": meta,
    }


def process_question(
    *,
    item: dict[str, str],
    teacher: Any,
    pipeline: CPUInferencePipeline,
    args: argparse.Namespace,
    audit_path: Path,
    failures_path: Path,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    question = item["question"]
    question_key = item["question_key"]
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    try:
        current_hypothesis = teacher_hypothesis(teacher, question, args.dialogue_context)
    except Exception as exc:
        stats["fail:hypothesis"] += 1
        append_jsonl(failures_path, {"question_key": question_key, "question": question, "stage": "hypothesis", "error": f"{type(exc).__name__}: {exc}"})
        return records

    scope_state: dict[str, Any] = {
        "retained_chapter_scope": None,
        "retained_storyline_scope": None,
        "retained_scope_evidence": [],
        "scope_retention_enabled": False,
    }
    pending_queries = build_initial_queries(question, current_hypothesis)
    seen_prompt_doc_ids: set[str] = set()

    for round_index in range(1, pipeline.max_retrieval_rounds + 1):
        try:
            evidence, prompt_evidence, retrieval_trace = retrieve_round(
                pipeline=pipeline,
                question=question,
                hypothesis=current_hypothesis,
                pending_queries=pending_queries,
                round_index=round_index,
                scope_state=scope_state,
            )
            if round_index > 1:
                prompt_evidence, novelty_record = build_novel_first_prompt_evidence(
                    pipeline=pipeline,
                    question=question,
                    hypothesis=current_hypothesis,
                    evidence=evidence,
                    seen_doc_ids=seen_prompt_doc_ids,
                    min_new=args.min_second_round_new_evidence,
                )
                retrieval_trace["second_round_novelty"] = novelty_record
                before_ids = [evidence_doc_id(item) for item in prompt_evidence]
                prompt_evidence = prefer_novel_prompt_evidence(
                    prompt_evidence,
                    seen_doc_ids=seen_prompt_doc_ids,
                    min_new=args.min_second_round_new_evidence,
                )
                after_ids = [evidence_doc_id(item) for item in prompt_evidence]
                retrieval_trace["prompt_evidence_ids_before_novelty"] = before_ids
                retrieval_trace["prompt_evidence_ids_after_novelty"] = after_ids
            user_prompt, evidence_brief = build_grounded_action_prompt(
                question=question,
                hypothesis=current_hypothesis,
                round_index=round_index,
                max_rounds=pipeline.max_retrieval_rounds,
                prompt_evidence=prompt_evidence,
                max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
                max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
                truncate_evidence_pack=args.truncate_evidence_pack,
                max_quote_chars=args.max_quote_chars,
                max_quotes_per_fact=args.max_quotes_per_fact,
                max_fact_quote_total_chars=args.max_fact_quote_total_chars,
                max_supported_facts=args.max_supported_facts,
                max_answer_quote_total_chars=args.max_answer_quote_total_chars,
            )
            seen_prompt_doc_ids.update(evidence_doc_id(item) for item in prompt_evidence if evidence_doc_id(item))
            api_payload = call_json_api(teacher, user_prompt, max_tokens=args.action_max_tokens, temperature=args.teacher_temperature)
            if isinstance(api_payload, dict) and str(api_payload.get("next_action") or "") == "answer_directly":
                locally_repaired_payload = discard_unfixable_quotes(
                    api_payload,
                    allowed_evidence=evidence_brief,
                    max_quote_chars=args.max_quote_chars,
                    max_quotes_per_fact=args.max_quotes_per_fact,
                    max_supported_facts=args.max_supported_facts,
                    max_answer_quote_total_chars=args.max_answer_quote_total_chars,
                )
                if locally_repaired_payload is not None:
                    api_payload = locally_repaired_payload
                    stats["repair:discard_unfixable_quote_precheck"] += 1
            action_payload, errors = normalize_action_payload(
                api_payload,
                question=question,
                round_index=round_index,
                max_rounds=pipeline.max_retrieval_rounds,
                allowed_evidence=evidence_brief,
                max_quote_chars=args.max_quote_chars,
                max_quotes_per_fact=args.max_quotes_per_fact,
                max_fact_quote_total_chars=args.max_fact_quote_total_chars,
                max_supported_facts=args.max_supported_facts,
                max_answer_quote_total_chars=args.max_answer_quote_total_chars,
            )
            if action_payload is None and api_payload is not None and str(api_payload.get("next_action") or "") == "answer_directly":
                repaired_payload = repair_action_payload(
                    teacher,
                    original_prompt=user_prompt,
                    invalid_payload=api_payload,
                    errors=errors,
                    max_tokens=args.action_max_tokens,
                    temperature=args.teacher_temperature,
                    max_quote_chars=args.max_quote_chars,
                    max_quotes_per_fact=args.max_quotes_per_fact,
                    max_fact_quote_total_chars=args.max_fact_quote_total_chars,
                    max_supported_facts=args.max_supported_facts,
                    max_answer_quote_total_chars=args.max_answer_quote_total_chars,
                )
                if isinstance(repaired_payload, dict) and str(repaired_payload.get("next_action") or "") == "answer_directly":
                    locally_repaired_payload = discard_unfixable_quotes(
                        repaired_payload,
                        allowed_evidence=evidence_brief,
                        max_quote_chars=args.max_quote_chars,
                        max_quotes_per_fact=args.max_quotes_per_fact,
                        max_supported_facts=args.max_supported_facts,
                        max_answer_quote_total_chars=args.max_answer_quote_total_chars,
                    )
                    if locally_repaired_payload is not None:
                        repaired_payload = locally_repaired_payload
                        stats["repair:discard_unfixable_quote_after_api_repair"] += 1
                repaired_action_payload, repaired_errors = normalize_action_payload(
                    repaired_payload,
                    question=question,
                    round_index=round_index,
                    max_rounds=pipeline.max_retrieval_rounds,
                    allowed_evidence=evidence_brief,
                    max_quote_chars=args.max_quote_chars,
                    max_quotes_per_fact=args.max_quotes_per_fact,
                    max_fact_quote_total_chars=args.max_fact_quote_total_chars,
                    max_supported_facts=args.max_supported_facts,
                    max_answer_quote_total_chars=args.max_answer_quote_total_chars,
                )
                if repaired_action_payload is not None:
                    api_payload = repaired_payload
                    action_payload = repaired_action_payload
                    errors = []
                    stats["repair:answer_quote_success"] += 1
                else:
                    stats["repair:answer_quote_failed"] += 1
                    if repaired_errors:
                        errors = repaired_errors
        except Exception as exc:
            stats[f"fail:round_{round_index}"] += 1
            append_jsonl(
                failures_path,
                {
                    "question_key": question_key,
                    "question": question,
                    "round": round_index,
                    "stage": "round",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            break

        trace_record = {
            "question_key": question_key,
            "question": question,
            "round": round_index,
            "hypothesis": asdict(current_hypothesis),
            "retrieval_trace": retrieval_trace,
            "api_payload": api_payload,
            "normalized_payload": action_payload,
            "normalize_errors": errors,
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
        trace_records.append(trace_record)
        append_jsonl(audit_path, trace_record)

        if action_payload is None:
            stats[f"drop:invalid_action_round_{round_index}"] += 1
            for error in errors:
                stats[f"invalid_reason:{error[:80]}"] += 1
            break

        record_id = stable_key(question_key, str(round_index), compact_json(action_payload))
        records.append(
            sft_record(
                record_id=f"{record_id}__teacher_state_grounded_action_sft",
                user_prompt=user_prompt,
                assistant_payload=action_payload,
                meta={
                    "question_key": question_key,
                    "round": round_index,
                    "source": "teacher_state_sft_api",
                    "schema": "grounded_action_v1",
                },
            )
        )
        stats[f"output_action:{action_payload['next_action']}"] += 1

        if action_payload["next_action"] in {"answer_directly", "abstain"}:
            break

        follow_up_payload = action_payload.get("follow_up_hypothesis")
        try:
            follow_up_hypothesis = normalize_hypothesis_payload(
                follow_up_payload,
                question=question,
                dialogue_context=current_hypothesis.dialogue_context,
                current_intent=current_hypothesis.intent,
            )
        except Exception as exc:
            stats["fail:follow_up_normalize"] += 1
            append_jsonl(
                failures_path,
                {
                    "question_key": question_key,
                    "question": question,
                    "round": round_index,
                    "stage": "follow_up_normalize",
                    "payload": follow_up_payload,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            break
        current_hypothesis = merge_hypotheses(current_hypothesis, follow_up_hypothesis)
        pending_queries = [build_retrieval_query(current_hypothesis)]
        pending_queries.extend(build_follow_up_hypothesis_queries(question, current_hypothesis))

    stats["questions_completed"] += 1
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline teacher-state grounded_action_v1 SFT data with API teacher and local RAG.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="teacher_state_grounded_action_sft_v1")
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--teacher-runtime-config", type=Path, default=DEFAULT_TEACHER_RUNTIME_CONFIG)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--dialogue-context", default="")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-api-request-logs", action="store_true")
    parser.add_argument("--truncate-evidence-pack", action="store_true", help="Use short truncated evidence blocks instead of full selected evidence.")
    parser.add_argument("--max-quote-chars", type=int, default=DEFAULT_MAX_QUOTE_CHARS)
    parser.add_argument("--max-quotes-per-fact", type=int, default=DEFAULT_MAX_QUOTES_PER_FACT)
    parser.add_argument("--max-fact-quote-total-chars", type=int, default=DEFAULT_MAX_FACT_QUOTE_TOTAL_CHARS)
    parser.add_argument("--max-supported-facts", type=int, default=DEFAULT_MAX_SUPPORTED_FACTS)
    parser.add_argument("--max-answer-quote-total-chars", type=int, default=DEFAULT_MAX_ANSWER_QUOTE_TOTAL_CHARS)
    parser.add_argument("--min-second-round-new-evidence", type=int, default=3)

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
    parser.add_argument("--minirag-mode-weights", type=parse_mode_weights, default=None)
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
    parser.add_argument("--enable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_true", default=None)
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
    parser.add_argument("--enable-mmr", dest="enable_mmr", action="store_true", default=None)
    parser.add_argument("--disable-mmr", dest="enable_mmr", action="store_false")
    parser.add_argument("--mmr-lambda", type=float, default=None)
    parser.add_argument("--enable-pyramid-order", dest="enable_pyramid_order", action="store_true", default=None)
    parser.add_argument("--disable-pyramid-order", dest="enable_pyramid_order", action="store_false")
    parser.add_argument("--enable-evidence-pinning", dest="enable_evidence_pinning", action="store_true", default=None)
    parser.add_argument("--disable-evidence-pinning", dest="enable_evidence_pinning", action="store_false")
    parser.add_argument("--enable-crag-refinement", dest="enable_crag_refinement", action="store_true", default=None)
    parser.add_argument("--disable-crag-refinement", dest="enable_crag_refinement", action="store_false")
    parser.add_argument("--crag-refine-top-sentences", type=int, default=None)
    parser.add_argument("--crag-refine-max-sentences", type=int, default=None)
    parser.add_argument("--answer-grounding-mode", choices=("off", "weak", "strict"), default=None)

    parser.add_argument("--teacher-backend", choices=("chat_completions", "openai_compatible_api", "responses_api", "responses"), default=None)
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--api-timeout", type=float, default=None)
    parser.add_argument("--teacher-max-tokens", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=0.1)
    parser.add_argument("--teacher-top-p", type=float, default=None)
    parser.add_argument("--action-max-tokens", type=int, default=4096)
    parser.add_argument("--no-json-response-format", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_local_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    if output_dir.exists() and args.overwrite:
        for name in ["train.json", "val.json", "dataset_info.json", "summary.json", "audit_records.jsonl", "failed.jsonl", "aligned_runtime.json"]:
            path = output_dir / name
            if path.exists():
                path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit_records.jsonl"
    failures_path = output_dir / "failed.jsonl"

    questions = load_questions(args)
    if not questions:
        raise SystemExit("No questions loaded. Pass --questions-file or --question.")

    pipeline, runtime_meta = build_retrieval_pipeline(args, output_dir)
    teacher = build_teacher_generator(args, output_dir)
    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    started = time.perf_counter()

    for item in tqdm(questions, desc="teacher-state sft", unit="question"):
        records.extend(
            process_question(
                item=item,
                teacher=teacher,
                pipeline=pipeline,
                args=args,
                audit_path=audit_path,
                failures_path=failures_path,
                stats=stats,
            )
        )

    train_records, val_records = split_records(records, seed=args.seed, val_ratio=args.val_ratio)
    write_json(output_dir / "train.json", train_records)
    write_json(output_dir / "val.json", val_records)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    summary = {
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "questions": len(questions),
        "records_total": len(records),
        "records_train": len(train_records),
        "records_val": len(val_records),
        "schema": "grounded_action_v1",
        "actions": dict(Counter(json.loads(record["conversations"][-1]["value"])["next_action"] for record in records)),
        "stats": dict(stats),
        "runtime": runtime_meta,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
