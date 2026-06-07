#!/usr/bin/env python3
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
    CONCLUSION_TASK_TYPE,
    FOLLOW_UP_HYPOTHESIS_TASK_TYPE,
    INITIAL_HYPOTHESIS_TASK_TYPE,
    CPUInferencePipeline,
    extract_json_object,
    repair_json_like_output,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.evaluate_multiround_retrieval_recall import (  # noqa: E402
    DEFAULT_RUNTIME_CONFIG_PATH,
    build_generator,
    build_query_config,
    config_value,
    load_runtime_config,
    parse_mode_weights,
    resolve_path,
)
from scripts.evaluate_retrieval_recall import extract_gold_text, load_listwise  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1"
DEFAULT_TEACHER_RUNTIME_CONFIG = PROJECT_ROOT / "api-mode/runtime_deepseek_api.json"
CHATML_MESSAGE_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}
DEFAULT_SYSTEM = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
INITIAL_FIELDS = (
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
GROUNDED_CONCLUSION_ACTIONS = {"answer_directly", "retrieve_more", "abstain"}


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def stable_key(*parts: str) -> str:
    return hashlib.sha1("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def resolve_local_path(path: Path | None, default: Path | None = None) -> Path | None:
    selected = path if path not in (None, "") else default
    if selected in (None, ""):
        return None
    selected = Path(selected)
    return selected if selected.is_absolute() else PROJECT_ROOT / selected


def strip_known_api_key_prefix(api_key: str) -> str:
    key = str(api_key or "").strip()
    if key.startswith("ds:sk-"):
        return key.split(":", 1)[1]
    return key


def load_api_mode_module() -> Any:
    module_path = PROJECT_ROOT / "api-mode/run_api_inference.py"
    spec = importlib.util.spec_from_file_location("goldenglow_api_mode_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API mode runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_task_type(prompt: str) -> str:
    if f"task: {INITIAL_HYPOTHESIS_TASK_TYPE}" in prompt or "hypothesis_builder" in prompt:
        if FOLLOW_UP_HYPOTHESIS_TASK_TYPE not in prompt and "follow_up_hypothesis_builder" not in prompt:
            return INITIAL_HYPOTHESIS_TASK_TYPE
    if f"task: {FOLLOW_UP_HYPOTHESIS_TASK_TYPE}" in prompt or "follow_up_hypothesis_builder" in prompt:
        return FOLLOW_UP_HYPOTHESIS_TASK_TYPE
    if f"task: {CONCLUSION_TASK_TYPE}" in prompt or "conclusion_generator" in prompt:
        return CONCLUSION_TASK_TYPE
    if "请直接回答用户问题" in prompt:
        return "direct_answer_generation"
    return "unknown"


def extract_question_from_prompt(prompt: str, fallback: str) -> str:
    patterns = (
        r"(?m)^question:\s*(.+?)\s*$",
        r"用户原问题:\s*(.+?)\s*(?:\n|$)",
        r"用户问题:\s*(.+?)\s*(?:\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt)
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return fallback


def split_chatml_prompt(prompt: str, task_type: str) -> tuple[str, str]:
    messages = [(role, content.strip()) for role, content in CHATML_MESSAGE_RE.findall(prompt) if content.strip()]
    if not messages:
        return DEFAULT_SYSTEM, prompt.strip()
    system_parts = [content for role, content in messages if role == "system"]
    user_parts = [content for role, content in messages if role == "user"]
    system = "\n\n".join(system_parts).strip() or DEFAULT_SYSTEM
    user = "\n\n".join(user_parts).strip()
    if not user:
        non_assistant = [content for role, content in messages if role != "assistant"]
        user = "\n\n".join(non_assistant).strip() or prompt.strip()
    if task_type == "direct_answer_generation" and "只输出 JSON" in system:
        system = "你是《明日方舟》剧情问答系统的最终回答模块。只依据证据输出自然语言答案。"
    return system, user


def clean_string_list(value: Any, *, limit: int) -> list[str]:
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


def normalize_json_payload(payload: dict[str, Any], *, task_type: str, question: str) -> dict[str, Any] | None:
    if task_type == INITIAL_HYPOTHESIS_TASK_TYPE:
        output = {field: payload.get(field) for field in INITIAL_FIELDS}
        output["question"] = str(output.get("question") or question).strip()
        output["intent"] = str(output.get("intent") or "plot_reasoning").strip()
        output["query_type"] = str(output.get("query_type") or "reasoning").strip()
        output["entities"] = clean_string_list(output.get("entities"), limit=12)
        output["keywords"] = clean_string_list(output.get("keywords"), limit=24)
        output["expected_answer_type"] = str(output.get("expected_answer_type") or "剧情问答").strip()
        output["dialogue_context"] = str(output.get("dialogue_context") or "").strip()
        if not output["question"] or not output["entities"] or not output["keywords"]:
            return None
        return output
    if task_type == FOLLOW_UP_HYPOTHESIS_TASK_TYPE:
        output = {field: payload.get(field) for field in FOLLOW_UP_FIELDS}
        output["question"] = str(output.get("question") or question).strip()
        output["query_type"] = str(output.get("query_type") or "reasoning").strip()
        output["entities"] = clean_string_list(output.get("entities"), limit=12)
        output["keywords"] = clean_string_list(output.get("keywords"), limit=24)
        output["expected_answer_type"] = str(output.get("expected_answer_type") or "剧情问答").strip()
        output["dialogue_context"] = str(output.get("dialogue_context") or "").strip()
        if not output["question"] or not output["entities"] or not output["keywords"]:
            return None
        return output
    if task_type == CONCLUSION_TASK_TYPE:
        output = {field: payload.get(field) for field in CONCLUSION_FIELDS}
        if "next_action" not in payload:
            decision = str(payload.get("decision") or "").strip().lower()
            mapped = {
                "answer": "answer_directly",
                "direct_answer": "answer_directly",
                "retrieve": "retrieve_more",
                "retrieve_more": "retrieve_more",
                "clarify": "clarify_user",
                "abstain": "abstain",
            }.get(decision)
            if mapped:
                output["next_action"] = mapped
        output["question"] = str(output.get("question") or question).strip()
        output["next_action"] = str(output.get("next_action") or "").strip()
        if output["next_action"] == "clarify_user":
            output["next_action"] = "abstain"
        if output["next_action"] in GROUNDED_CONCLUSION_ACTIONS:
            if output["next_action"] == "answer_directly":
                final_answer = str(payload.get("final_answer") or payload.get("answer") or "").strip()
                supported_facts = payload.get("supported_facts") if isinstance(payload.get("supported_facts"), list) else []
                inferred_facts = payload.get("inferred_facts") if isinstance(payload.get("inferred_facts"), list) else []
                if not final_answer:
                    return None
                return {
                    "next_action": "answer_directly",
                    "supported_facts": supported_facts,
                    "inferred_facts": inferred_facts,
                    "final_answer": final_answer,
                }
            if output["next_action"] == "retrieve_more":
                follow_up = payload.get("follow_up_hypothesis")
                if not isinstance(follow_up, dict):
                    return None
                return {
                    "next_action": "retrieve_more",
                    "follow_up_hypothesis": follow_up,
                }
            final_answer = str(payload.get("final_answer") or payload.get("answer") or "").strip()
            if not final_answer:
                final_answer = "现有证据不足以确认。"
            return {
                "next_action": "abstain",
                "final_answer": final_answer,
            }
        output["answer"] = str(output.get("answer") or "").strip()
        output["missing_slots"] = clean_string_list(output.get("missing_slots"), limit=8)
        output["clarification_question"] = str(output.get("clarification_question") or "").strip()
        if output["next_action"] != "retrieve_more":
            output["follow_up_hypothesis"] = None
        elif not isinstance(output.get("follow_up_hypothesis"), dict):
            output["follow_up_hypothesis"] = None
        if output["next_action"] not in {"answer_directly", "retrieve_more", "clarify_user", "abstain"}:
            return None
        if output["next_action"] in {"answer_directly", "abstain"} and not output["answer"]:
            return None
        return output
    return None


def canonical_response(raw: str, *, task_type: str, question: str, require_valid_json: bool) -> tuple[str, bool]:
    text = repair_json_like_output(str(raw or "").strip())
    if task_type in {INITIAL_HYPOTHESIS_TASK_TYPE, FOLLOW_UP_HYPOTHESIS_TASK_TYPE, CONCLUSION_TASK_TYPE}:
        payload = extract_json_object(text)
        if not isinstance(payload, dict):
            return (text[:4096], False)
        normalized = normalize_json_payload(payload, task_type=task_type, question=question)
        if normalized is None:
            return (compact_json(payload), False)
        return (compact_json(normalized), True)
    if require_valid_json:
        return ("", False)
    return (text, bool(text))


class RecordingGenerator:
    backend_name = "recording-generator"

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    @property
    def max_tokens(self) -> int:
        return int(getattr(self.inner, "max_tokens", 512))

    def describe_runtime(self) -> dict[str, Any]:
        payload = dict(self.inner.describe_runtime())
        payload["recording_wrapper"] = True
        return payload

    def clear(self) -> None:
        self.calls.clear()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        call_index = len(self.calls) + 1
        started = time.perf_counter()
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        try:
            output = self.inner.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
            )
        except Exception as exc:
            self.calls.append(
                {
                    "call_index": call_index,
                    "task_type": infer_task_type(prompt),
                    "prompt": prompt,
                    "kwargs": kwargs,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                }
            )
            raise
        self.calls.append(
            {
                "call_index": call_index,
                "task_type": infer_task_type(prompt),
                "prompt": prompt,
                "kwargs": kwargs,
                "output": output,
                "error": "",
                "elapsed_sec": round(time.perf_counter() - started, 3),
            }
        )
        return output


def make_pipeline_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        backend=args.student_backend,
        base_model=args.base_model,
        lora_path=args.lora_path,
        no_lora=args.no_lora,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        llama_cli=args.llama_cli,
        gguf_model=args.gguf_model,
        lora_gguf=args.lora_gguf,
        threads=args.threads,
        llama_device=args.llama_device,
        llama_gpu_layers=args.llama_gpu_layers,
        llama_batch_size=args.llama_batch_size,
        llama_ubatch_size=args.llama_ubatch_size,
        llama_flash_attn=args.llama_flash_attn,
        ctx_size=args.ctx_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        api_base_url=None,
        api_key_env=None,
        api_key=None,
        api_model=None,
        api_timeout=None,
        no_json_response_format=False,
        api_request_log_dir=None,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        fusion_top_k=args.fusion_top_k,
        reranker_candidate_top_k=args.reranker_candidate_top_k,
        rerank_batch_size=args.rerank_batch_size,
        minirag_top_k=args.minirag_top_k,
        minirag_weight=args.minirag_weight,
        minirag_mode_weights=args.minirag_mode_weights,
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
        enable_mmr=args.enable_mmr,
        mmr_lambda=args.mmr_lambda,
        enable_pyramid_order=args.enable_pyramid_order,
        enable_evidence_pinning=args.enable_evidence_pinning,
        enable_crag_refinement=args.enable_crag_refinement,
        crag_refine_top_sentences=args.crag_refine_top_sentences,
        crag_refine_max_sentences=args.crag_refine_max_sentences,
        self_consistency_samples=args.self_consistency_samples,
        self_consistency_temperature=args.self_consistency_temperature,
        conclusion_prompt_mode=args.conclusion_prompt_mode,
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
        response_format_json=bool(generator_cfg.get("response_format_json", True)) and not args.no_json_response_format,
        extra_body=generator_cfg.get("extra_body") if isinstance(generator_cfg.get("extra_body"), dict) else None,
        request_log_dir=request_log_dir,
    )


def build_retriever_and_pipeline(args: argparse.Namespace, output_dir: Path) -> tuple[CPUInferencePipeline, RecordingGenerator, dict[str, Any]]:
    runtime_config_path = resolve_local_path(args.runtime_config, DEFAULT_RUNTIME_CONFIG_PATH) or DEFAULT_RUNTIME_CONFIG_PATH
    runtime_config = load_runtime_config(runtime_config_path)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}
    pipeline_args = make_pipeline_args(args)

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
    query_config = build_query_config(pipeline_args, retrieval_cfg, rerank_top_k=rerank_top_k)
    student_generator = build_generator(pipeline_args, generator_cfg)
    recorder = RecordingGenerator(student_generator)
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=recorder,
        query_config=query_config,
        max_retrieval_rounds=int(config_value(args.max_rounds, inference_cfg, "max_retrieval_rounds", 2)),
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 12)),
        prompt_evidence_max_chars_per_doc=int(
            config_value(args.prompt_evidence_max_chars_per_doc, inference_cfg, "prompt_evidence_max_chars_per_doc", 900)
        ),
        prompt_conclusion_evidence_max_total_chars=int(
            config_value(
                args.prompt_conclusion_evidence_max_total_chars,
                inference_cfg,
                "prompt_conclusion_evidence_max_total_chars",
                9000,
            )
        ),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_evidence_pinning=bool(config_value(args.enable_evidence_pinning, inference_cfg, "enable_evidence_pinning", False)),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)),
        crag_refine_max_sentences=int(config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)),
        self_consistency_samples=int(config_value(args.self_consistency_samples, inference_cfg, "self_consistency_samples", 1)),
        self_consistency_temperature=float(
            config_value(args.self_consistency_temperature, inference_cfg, "self_consistency_temperature", 0.7)
        ),
        answer_grounding_mode=str(config_value(args.answer_grounding_mode, inference_cfg, "answer_grounding_mode", "weak")),
        conclusion_prompt_mode=str(config_value(args.conclusion_prompt_mode, inference_cfg, "conclusion_prompt_mode", "minimal")),
        use_model_hypothesis=bool(inference_cfg.get("use_model_hypothesis", True)),
        use_model_conclusion_generation=bool(inference_cfg.get("use_model_conclusion_generation", True)),
        web_context_config=inference_cfg.get("web_context") if isinstance(inference_cfg.get("web_context"), dict) else None,
    )
    metadata = {
        "runtime_config": str(runtime_config_path),
        "query_config": asdict(query_config),
        "student_runtime": recorder.describe_runtime(),
        "inference_config": {
            "max_retrieval_rounds": pipeline.max_retrieval_rounds,
            "prompt_evidence_top_k": pipeline.prompt_evidence_top_k,
            "conclusion_prompt_mode": pipeline.conclusion_prompt_mode,
            "answer_grounding_mode": pipeline.answer_grounding_mode,
            "web_context_enabled": pipeline.web_context_config.enabled,
        },
    }
    write_json(output_dir / "aligned_runtime.json", metadata)
    return pipeline, recorder, metadata


def load_question_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if args.questions_file:
        path = resolve_local_path(args.questions_file)
        assert path is not None
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, list):
                for index, item in enumerate(payload):
                    if isinstance(item, str):
                        items.append({"question": item, "question_key": stable_key("json", item)})
                    elif isinstance(item, dict) and item.get("question"):
                        items.append({**item, "question_key": str(item.get("question_key") or stable_key("json", item["question"]))})
        else:
            for index, line in enumerate(text.splitlines()):
                if not line.strip():
                    continue
                if path.suffix.lower() == ".jsonl":
                    payload = json.loads(line)
                    question = str(payload.get("question") or payload.get("query") or "").strip()
                    if question:
                        items.append({**payload, "question": question, "question_key": str(payload.get("question_key") or stable_key("jsonl", question))})
                else:
                    question = line.strip()
                    items.append({"question": question, "question_key": stable_key("txt", question)})
    for question in args.question or []:
        items.append({"question": question, "question_key": stable_key("inline", question)})
    if not items:
        listwise_path = resolve_local_path(args.listwise)
        assert listwise_path is not None
        records = [
            record
            for record in load_listwise(listwise_path)
            if str(record.get("query") or "").strip() and extract_gold_text(record)
        ]
        rng = random.Random(args.seed)
        rng.shuffle(records)
        if args.sample_offset > 0:
            records = records[args.sample_offset :]
        if args.sample is not None:
            records = records[: max(0, args.sample)]
        for index, record in enumerate(records):
            question = str(record.get("query") or "").strip()
            items.append(
                {
                    "question": question,
                    "question_key": stable_key("listwise", str(args.seed), str(args.sample_offset), str(index), question),
                    "source_record": {
                        "query_type": record.get("query_type"),
                        "answer_focus": record.get("answer_focus"),
                        "gold": extract_gold_text(record),
                    },
                }
            )
    if args.limit is not None:
        items = items[: max(0, args.limit)]
    return items


def make_kto_record(
    *,
    record_id: str,
    task_type: str,
    system: str,
    user: str,
    response: str,
    kto_tag: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": task_type,
        "bucket": "soda_blackbox",
        "system": system,
        "tools": "[]",
        "kto_tag": kto_tag,
        "conversations": [
            {"from": "human", "value": user},
            {"from": "gpt", "value": response},
        ],
        "meta": meta,
    }


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        prompt_key = str(record.get("meta", {}).get("prompt_key") or record.get("id") or index)
        by_prompt.setdefault(prompt_key, []).append(record)
    keys = list(by_prompt)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = max(1, int(round(len(records) * val_ratio))) if len(records) > 10 and val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for key in keys:
        target = val if val_count < target_val else train
        target.extend(by_prompt[key])
        if target is val:
            val_count += len(by_prompt[key])
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SODA semi-online black-box KTO data from local 4B prompts and API teacher outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--teacher-runtime-config", type=Path, default=DEFAULT_TEACHER_RUNTIME_CONFIG)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--listwise", type=Path, default=Path("data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"))
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--dialogue-context", default="")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-direct-answer", action="store_true")
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-api-request-logs", action="store_true")

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
    parser.add_argument("--self-consistency-samples", type=int, default=None)
    parser.add_argument("--self-consistency-temperature", type=float, default=None)
    parser.add_argument("--answer-grounding-mode", choices=("off", "weak", "strict"), default=None)

    parser.add_argument("--student-backend", choices=("vllm", "llama.cpp"), default=None)
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--llama-cli", type=Path, default=None)
    parser.add_argument("--gguf-model", type=Path, default=None)
    parser.add_argument("--lora-gguf", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--llama-device", default=None)
    parser.add_argument("--llama-gpu-layers", default=None)
    parser.add_argument("--llama-batch-size", type=int, default=None)
    parser.add_argument("--llama-ubatch-size", type=int, default=None)
    parser.add_argument("--llama-flash-attn", choices=("on", "off", "auto"), default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)

    parser.add_argument("--teacher-backend", choices=("chat_completions", "openai_compatible_api", "responses_api", "responses"), default=None)
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--api-timeout", type=float, default=None)
    parser.add_argument("--teacher-max-tokens", type=int, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--teacher-top-p", type=float, default=None)
    parser.add_argument("--no-json-response-format", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_local_path(args.output_dir, DEFAULT_OUTPUT_DIR) or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "audit_records.jsonl"
    raw_pairs_path = output_dir / "raw_pairs.jsonl"
    failures_path = output_dir / "failed.jsonl"

    existing_prompt_keys: set[str] = set()
    records: list[dict[str, Any]] = []
    if args.skip_existing and records_path.exists():
        for record in read_jsonl(records_path):
            records.append(record)
            prompt_key = str(record.get("meta", {}).get("prompt_key") or "")
            if prompt_key:
                existing_prompt_keys.add(prompt_key)

    items = load_question_items(args)
    if not items:
        raise SystemExit("No questions loaded.")

    pipeline, recorder, runtime_meta = build_retriever_and_pipeline(args, output_dir)
    teacher = None if args.dry_run else build_teacher_generator(args, output_dir)
    stats: Counter[str] = Counter()
    started_all = time.perf_counter()
    progress = tqdm(items, desc="soda blackbox", unit="question")
    for item in progress:
        question = str(item.get("question") or "").strip()
        question_key = str(item.get("question_key") or stable_key("question", question))
        if not question:
            continue
        recorder.clear()
        started = time.perf_counter()
        try:
            result = pipeline.run(question, args.dialogue_context)
            result_payload = asdict(result)
        except Exception as exc:
            stats["student_failed"] += 1
            append_jsonl(
                failures_path,
                {
                    "question_key": question_key,
                    "question": question,
                    "stage": "student_pipeline",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                },
            )
            continue

        for call in recorder.calls:
            task_type = str(call.get("task_type") or "unknown")
            if task_type == "unknown":
                stats["skipped_unknown_task"] += 1
                continue
            if task_type == "direct_answer_generation" and not args.include_direct_answer:
                stats["skipped_direct_answer"] += 1
                continue
            prompt = str(call.get("prompt") or "")
            prompt_key = stable_key(question_key, str(call.get("call_index")), task_type, prompt)
            if prompt_key in existing_prompt_keys:
                stats["skipped_existing"] += 1
                continue
            system, user_prompt = split_chatml_prompt(prompt, task_type)
            prompt_question = extract_question_from_prompt(prompt, question)
            student_output, student_valid = canonical_response(
                str(call.get("output") or ""),
                task_type=task_type,
                question=prompt_question,
                require_valid_json=False,
            )
            if args.dry_run:
                append_jsonl(
                    raw_pairs_path,
                    {
                        "question_key": question_key,
                        "prompt_key": prompt_key,
                        "question": question,
                        "task_type": task_type,
                        "system": system,
                        "user_prompt": user_prompt,
                        "student_output": student_output,
                        "student_valid": student_valid,
                        "teacher_output": "",
                        "dry_run": True,
                    },
                )
                stats[f"dry_task:{task_type}"] += 1
                continue
            assert teacher is not None
            try:
                teacher_raw = teacher.generate(
                    prompt,
                    max_tokens=(call.get("kwargs") or {}).get("max_tokens"),
                    temperature=0.1,
                    top_p=0.8,
                    repeat_penalty=1.0,
                )
            except Exception as exc:
                stats["teacher_failed"] += 1
                append_jsonl(
                    failures_path,
                    {
                        "question_key": question_key,
                        "prompt_key": prompt_key,
                        "question": question,
                        "task_type": task_type,
                        "stage": "teacher_replay",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            teacher_output, teacher_valid = canonical_response(
                teacher_raw,
                task_type=task_type,
                question=prompt_question,
                require_valid_json=task_type != "direct_answer_generation",
            )
            append_jsonl(
                raw_pairs_path,
                {
                    "question_key": question_key,
                    "prompt_key": prompt_key,
                    "question": question,
                    "task_type": task_type,
                    "system": system,
                    "user_prompt": user_prompt,
                    "student_output_raw": call.get("output") or "",
                    "student_output": student_output,
                    "student_valid": student_valid,
                    "teacher_output_raw": teacher_raw,
                    "teacher_output": teacher_output,
                    "teacher_valid": teacher_valid,
                    "student_elapsed_sec": call.get("elapsed_sec"),
                },
            )
            if not teacher_valid:
                stats[f"teacher_invalid:{task_type}"] += 1
                continue
            meta = {
                "soda_mode": "semi_online_blackbox_replay",
                "question_key": question_key,
                "prompt_key": prompt_key,
                "call_index": call.get("call_index"),
                "task_type": task_type,
                "student_valid": student_valid,
                "student_elapsed_sec": call.get("elapsed_sec"),
                "student_final_answer": str(result_payload.get("answer") or "")[:600],
                "source": item.get("source_record") or {},
            }
            pos = make_kto_record(
                record_id=f"{prompt_key}-teacher-pos",
                task_type=task_type,
                system=system,
                user=user_prompt,
                response=teacher_output,
                kto_tag=True,
                meta={**meta, "preference_role": "teacher_positive"},
            )
            records.append(pos)
            append_jsonl(records_path, pos)
            stats[f"positive:{task_type}"] += 1
            existing_prompt_keys.add(prompt_key)
            if not args.no_negative and student_output and student_output != teacher_output:
                neg = make_kto_record(
                    record_id=f"{prompt_key}-student-neg",
                    task_type=task_type,
                    system=system,
                    user=user_prompt,
                    response=student_output,
                    kto_tag=False,
                    meta={**meta, "preference_role": "student_negative"},
                )
                records.append(neg)
                append_jsonl(records_path, neg)
                stats[f"negative:{task_type}"] += 1
            else:
                stats[f"negative_skipped:{task_type}"] += 1
        stats["questions_completed"] += 1
        progress.set_postfix({"records": len(records), "failed": stats["student_failed"] + stats["teacher_failed"]})

    train_records, val_records = split_records(records, seed=args.seed, val_ratio=max(0.0, min(0.5, args.val_ratio)))
    write_json(output_dir / "train.json", train_records)
    write_json(output_dir / "val.json", val_records)
    write_json(output_dir / "dataset_info.json", dataset_info(output_dir.name))
    summary = {
        "output_dir": str(output_dir),
        "records_total": len(records),
        "records_train": len(train_records),
        "records_val": len(val_records),
        "dry_run": args.dry_run,
        "questions": len(items),
        "stats": dict(stats),
        "runtime": runtime_meta,
        "teacher_runtime_config": str(resolve_local_path(args.teacher_runtime_config, DEFAULT_TEACHER_RUNTIME_CONFIG)),
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "notes": [
            "SODA positive records are API teacher outputs replayed on exact prompts produced by the current student pipeline.",
            "SODA negative records are current student outputs on the same prompts.",
            "Retrieval, evidence selection, prompts, and fallback behavior are inherited from the same CPUInferencePipeline runtime config.",
        ],
    }
    write_json(output_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
