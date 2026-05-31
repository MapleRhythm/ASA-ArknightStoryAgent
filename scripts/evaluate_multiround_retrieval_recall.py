#!/usr/bin/env python3
"""Evaluate true multi-round retrieval recall with model-generated hypotheses.

Round 1 uses the 4B model to generate the initial hypothesis document. Later
rounds use the 4B model to generate follow-up hypothesis documents from the
accumulated evidence, then retrieve again. Recall is measured on the accumulated
per-round evidence pool, not on a single static query.
"""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def _should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_env == "train":
        return True
    executable = Path(sys.executable).as_posix().lower()
    return "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if _should_use_train_overrides():
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
from goldenglow.inference.cpu_pipeline import (  # noqa: E402
    CPUInferencePipeline,
    ConclusionResult,
    HypothesisDocument,
    LlamaCppRunner,
    VllmRunner,
    _resolve_referential_question,
    build_retrieval_query,
    merge_hypotheses,
    render_dialogue_context_for_prompt,
)
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402
from scripts.evaluate_retrieval_recall import (  # noqa: E402
    candidate_hit_source,
    extract_gold_text,
    load_listwise,
    normalize_text,
    parse_mode_weights,
    parse_top_ks,
)

DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_inference_gpu.json"
DEFAULT_LLAMA_CLI_PATH = PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-completion"
DEFAULT_GGUF_MODEL_PATH = (
    PROJECT_ROOT / "model" / "gguf" / "teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf"
)
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "model" / "qwen3.5-4b"
DEFAULT_VLLM_LORA_PATH = (
    PROJECT_ROOT
    / "model"
    / "lora"
    / "teacher_online_chain_short_prompt_v2_ds_flash_500_plus_smoke20_sample50_quality_fix3_qwen35_4b_lr3e5_epoch1"
)


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


def detect_visible_gpu_count() -> int:
    raw_devices = [
        item.strip()
        for item in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if item.strip()
    ]
    return len(raw_devices) if raw_devices else 1


def is_fatal_record_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    fatal_markers = (
        "vLLM engine initialization previously failed",
        "VllmConfig",
        "is not divisible by",
        "CUDA out of memory",
        "Base model path not found",
        "LoRA path not found",
    )
    return any(marker in message for marker in fatal_markers)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listwise",
        type=Path,
        default=Path(
            "data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_listwise.jsonl"
        ),
        help="Gold reranker_listwise.jsonl file.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0, help="Take --sample records after skipping this many records.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Shuffle records with this seed before applying --sample.")
    parser.add_argument("--top-ks", type=parse_top_ks, default="1,5,10,20,50")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument(
        "--planner-mode",
        choices=("direct_follow_up", "conclusion"),
        default="direct_follow_up",
        help=(
            "direct_follow_up: every later round calls follow_up_hypothesis_builder. "
            "conclusion: use the runtime conclusion planner and its follow_up_hypothesis when it asks to retrieve more."
        ),
    )
    parser.add_argument(
        "--dialogue-context",
        default="",
        help="Optional fixed dialogue context added to every question.",
    )
    parser.add_argument("--device", default=None, help="Retrieval device, e.g. cuda or cpu.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--reranker-max-length", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--reranker-candidate-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--minirag-top-k", type=int, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
    parser.add_argument(
        "--minirag-mode-weights",
        type=parse_mode_weights,
        default=None,
        help="Optional query_type multipliers, e.g. relation=1.0,fact=0.6.",
    )
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument(
        "--minirag-fusion-mode",
        choices=("score", "append"),
        default=None,
    )
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
    parser.add_argument("--enable-neighbor-expansion", action="store_true", default=None)
    parser.add_argument("--disable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_false")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=None)
    parser.add_argument("--neighbor-story-window", type=int, default=None)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=None)
    parser.add_argument("--prompt-evidence-top-k", type=int, default=None)
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
    parser.add_argument(
        "--backend",
        choices=("vllm", "llama.cpp", "api", "openai_compatible_api", "chat_completions", "responses_api", "responses"),
        default=None,
    )
    parser.add_argument("--base-model", type=Path, default=None)
    parser.add_argument("--lora-path", type=Path, default=None)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA even if runtime config contains a default vLLM LoRA path.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
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
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-model", type=str, default=None)
    parser.add_argument("--api-timeout", type=float, default=None)
    parser.add_argument("--no-json-response-format", action="store_true")
    parser.add_argument("--api-request-log-dir", type=Path, default=None)
    parser.add_argument("--jaccard-threshold", type=float, default=0.25)
    parser.add_argument("--overlap-threshold", type=float, default=0.32)
    parser.add_argument("--min-overlap-grams", type=int, default=60)
    parser.add_argument("--min-candidate-grams", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def validate_vllm_lora_path(path: Path | None) -> None:
    if path is None:
        return
    if (path / "adapter_config.json").exists():
        return
    child_adapters = (
        sorted(
            child
            for child in path.glob("*")
            if child.is_dir() and (child / "adapter_config.json").exists()
        )
        if path.is_dir()
        else []
    )
    hint = ""
    if child_adapters:
        hint = "\nCandidate adapter dirs:\n" + "\n".join(f"  {child}" for child in child_adapters[:8])
    raise SystemExit(
        f"Invalid vLLM LoRA path: {path}. Pass a directory containing adapter_config.json."
        + hint
    )


def load_api_mode_module() -> Any:
    module_path = PROJECT_ROOT / "api-mode" / "run_api_inference.py"
    spec = importlib.util.spec_from_file_location("goldenglow_api_mode_runner", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load API mode runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_generator(args: argparse.Namespace, generator_cfg: dict[str, Any]) -> Any:
    llama_cfg = generator_cfg.get("llama_cpp", {}) if isinstance(generator_cfg.get("llama_cpp"), dict) else {}
    vllm_cfg = generator_cfg.get("vllm", {}) if isinstance(generator_cfg.get("vllm"), dict) else {}
    backend = str(config_value(args.backend, generator_cfg, "backend", "vllm"))
    ctx_size = int(config_value(args.ctx_size, generator_cfg, "ctx_size", 12000))
    max_tokens = int(config_value(args.max_tokens, generator_cfg, "max_tokens", 512))
    temperature = float(config_value(args.temperature, generator_cfg, "temperature", 0.2))
    top_p = float(config_value(args.top_p, generator_cfg, "top_p", 0.9))
    repeat_penalty = float(config_value(args.repeat_penalty, generator_cfg, "repeat_penalty", 1.05))

    if backend == "vllm":
        base_model = resolve_path(
            args.base_model if args.base_model is not None else vllm_cfg.get("base_model_path"),
            default=DEFAULT_BASE_MODEL_PATH,
        )
        if args.no_lora:
            lora_path = None
        else:
            lora_path = resolve_path(
                args.lora_path if args.lora_path is not None else vllm_cfg.get("lora_path"),
                default=DEFAULT_VLLM_LORA_PATH if DEFAULT_VLLM_LORA_PATH.exists() else None,
            )
        validate_vllm_lora_path(lora_path)
        return VllmRunner(
            base_model_path=base_model or DEFAULT_BASE_MODEL_PATH,
            lora_path=lora_path,
            tensor_parallel_size=int(
                config_value(args.tensor_parallel_size, vllm_cfg, "tensor_parallel_size", detect_visible_gpu_count())
            ),
            gpu_memory_utilization=float(
                config_value(args.gpu_memory_utilization, vllm_cfg, "gpu_memory_utilization", 0.9)
            ),
            max_model_len=int(config_value(args.ctx_size, vllm_cfg, "max_model_len", ctx_size)),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            dtype=str(config_value(args.dtype, vllm_cfg, "dtype", "auto")),
            max_num_batched_tokens=(
                int(config_value(args.max_num_batched_tokens, vllm_cfg, "max_num_batched_tokens", 0)) or None
            ),
        )

    if backend in {"api", "openai_compatible_api", "chat_completions", "responses_api", "responses"}:
        api_mode = load_api_mode_module()
        configured_backend = str(generator_cfg.get("backend", "chat_completions"))
        api_backend = (
            configured_backend
            if backend == "api" and configured_backend in {"openai_compatible_api", "chat_completions", "responses_api", "responses"}
            else ("chat_completions" if backend == "api" else backend)
        )
        if api_backend in {"openai_compatible_api", "chat_completions"}:
            generator_cls = api_mode.OpenAICompatibleAPIRunner
        elif api_backend in {"responses_api", "responses"}:
            generator_cls = api_mode.ResponsesAPIRunner
        else:
            raise SystemExit(f"Unsupported API generator backend: {api_backend}")
        api_key_env = str(config_value(args.api_key_env, generator_cfg, "api_key_env", "DEEPSEEK_API_KEY"))
        api_key = args.api_key if args.api_key is not None else os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"Missing API key. Set environment variable {api_key_env} or pass --api-key.")
        api_mode.validate_api_key(api_key, api_key_env)
        request_log_dir = None
        if args.api_request_log_dir is not None:
            request_log_dir = args.api_request_log_dir
            if not request_log_dir.is_absolute():
                request_log_dir = PROJECT_ROOT / request_log_dir
        return generator_cls(
            api_base_url=str(config_value(args.api_base_url, generator_cfg, "api_base_url", "https://api.deepseek.com")),
            api_key=api_key,
            api_key_env=api_key_env,
            model=str(config_value(args.api_model, generator_cfg, "model", "deepseek-v4-flash")),
            timeout=float(config_value(args.api_timeout, generator_cfg, "timeout", 120)),
            max_tokens=max(int(config_value(args.max_tokens, generator_cfg, "max_tokens", 4096)), 4096),
            temperature=temperature,
            top_p=top_p,
            response_format_json=bool(generator_cfg.get("response_format_json", True)) and not args.no_json_response_format,
            extra_body=generator_cfg.get("extra_body") if isinstance(generator_cfg.get("extra_body"), dict) else None,
            request_log_dir=request_log_dir,
        )

    llama_cli = resolve_path(
        args.llama_cli if args.llama_cli is not None else llama_cfg.get("llama_cli_path"),
        default=DEFAULT_LLAMA_CLI_PATH,
    )
    gguf_model = resolve_path(
        args.gguf_model if args.gguf_model is not None else llama_cfg.get("gguf_model_path"),
        default=DEFAULT_GGUF_MODEL_PATH,
    )
    lora_path = resolve_path(
        args.lora_gguf if args.lora_gguf is not None else llama_cfg.get("lora_path"),
        default=None,
    )
    return LlamaCppRunner(
        llama_cli_path=llama_cli or DEFAULT_LLAMA_CLI_PATH,
        gguf_model_path=gguf_model or DEFAULT_GGUF_MODEL_PATH,
        lora_path=lora_path,
        threads=args.threads,
        ctx_size=int(config_value(args.ctx_size, llama_cfg, "ctx_size", ctx_size)),
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        device=config_value(args.llama_device, llama_cfg, "device", None),
        gpu_layers=config_value(args.llama_gpu_layers, llama_cfg, "gpu_layers", None),
        batch_size=(
            int(config_value(args.llama_batch_size, llama_cfg, "batch_size", 0))
            if config_value(args.llama_batch_size, llama_cfg, "batch_size", None) is not None
            else None
        ),
        ubatch_size=(
            int(config_value(args.llama_ubatch_size, llama_cfg, "ubatch_size", 0))
            if config_value(args.llama_ubatch_size, llama_cfg, "ubatch_size", None) is not None
            else None
        ),
        flash_attn=config_value(args.llama_flash_attn, llama_cfg, "flash_attn", None),
    )


def build_query_config(
    args: argparse.Namespace,
    retrieval_cfg: dict[str, Any],
    *,
    rerank_top_k: int,
) -> QueryConfig:
    minirag_mode_weights = args.minirag_mode_weights
    if minirag_mode_weights is None:
        configured = retrieval_cfg.get("minirag_mode_weights") or {}
        minirag_mode_weights = (
            {str(key): float(value) for key, value in configured.items()}
            if isinstance(configured, dict)
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
        minirag_chapter_isolation=bool(
            config_value(args.minirag_chapter_isolation, retrieval_cfg, "minirag_chapter_isolation", True)
        ),
        minirag_auto_second_retrieval=bool(
            config_value(
                args.minirag_auto_second_retrieval,
                retrieval_cfg,
                "minirag_auto_second_retrieval",
                True,
            )
        ),
        minirag_scope_seed_top_k=int(
            config_value(args.minirag_scope_seed_top_k, retrieval_cfg, "minirag_scope_seed_top_k", 40)
        ),
        minirag_expansion_query_top_k=int(
            config_value(
                args.minirag_expansion_query_top_k,
                retrieval_cfg,
                "minirag_expansion_query_top_k",
                8,
            )
        ),
        minirag_graph_scope_min_ratio=float(
            config_value(args.minirag_graph_scope_min_ratio, retrieval_cfg, "minirag_graph_scope_min_ratio", 1.0)
        ),
        minirag_second_pass_scope_min_ratio=float(
            config_value(
                args.minirag_second_pass_scope_min_ratio,
                retrieval_cfg,
                "minirag_second_pass_scope_min_ratio",
                2.5,
            )
        ),
        enable_storyline_sparse_scope=bool(
            config_value(args.enable_storyline_sparse_scope, retrieval_cfg, "enable_storyline_sparse_scope", True)
        ),
        storyline_scope_seed_top_k=int(
            config_value(args.storyline_scope_seed_top_k, retrieval_cfg, "storyline_scope_seed_top_k", 40)
        ),
        storyline_sparse_scope_min_ratio=float(
            config_value(args.storyline_sparse_scope_min_ratio, retrieval_cfg, "storyline_sparse_scope_min_ratio", 1.5)
        ),
        reranker_candidate_top_k=int(
            config_value(args.reranker_candidate_top_k, retrieval_cfg, "reranker_candidate_top_k", 120)
        ),
        enable_neighbor_expansion=bool(
            config_value(args.enable_neighbor_expansion, retrieval_cfg, "enable_neighbor_expansion", False)
        ),
        neighbor_max_seed_docs=int(
            config_value(args.neighbor_max_seed_docs, retrieval_cfg, "neighbor_max_seed_docs", 24)
        ),
        neighbor_story_window=int(config_value(args.neighbor_story_window, retrieval_cfg, "neighbor_story_window", 2)),
        neighbor_activity_story_sort_window=int(
            config_value(
                args.neighbor_activity_story_sort_window,
                retrieval_cfg,
                "neighbor_activity_story_sort_window",
                1,
            )
        ),
        rerank_batch_size=int(config_value(args.rerank_batch_size, retrieval_cfg, "rerank_batch_size", 8)),
    )


def evidence_key(item: dict[str, Any]) -> str:
    doc = item.get("document") or {}
    chain_text = str(item.get("evidence_chain_text") or "").strip()
    if chain_text:
        return "chain:" + chain_text
    return "doc:" + str(doc.get("id") or item.get("doc_index") or "")


def merge_evidence_pool(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {evidence_key(item) for item in existing}
    merged = list(existing)
    for item in new_items:
        key = evidence_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def infer_missing_slots(
    *,
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    query_type: str,
) -> list[str]:
    text = "\n".join(
        str(item.get("evidence_chain_text") or item.get("document", {}).get("clean_text") or "")
        for item in evidence[:8]
    )
    slots: list[str] = []
    if query_type in {"causality", "reasoning", "answerability"} or any(token in question for token in ("为什么", "为何", "原因", "动机")):
        slots.extend(["直接说明原因或动机的剧情原文", "事件前因后果和人物选择的桥接证据"])
    if query_type in {"relation", "reveal", "mystery"} or any(token in question for token in ("身份", "关系", "真相", "是谁", "阴谋")):
        slots.extend(["直接绑定人物身份或关系的证据", "揭示真相、主使、来历或称谓的上下文"])
    if query_type == "fact" or any(token in question for token in ("什么", "哪里", "何时", "发生了什么")):
        slots.append("能直接回答事实细节的原文片段")
    missing_entities = [
        entity
        for entity in hypothesis.entities[:6]
        if entity and entity not in text and entity not in {"第", "章"}
    ]
    if missing_entities:
        slots.append("当前证据缺少关键实体: " + "、".join(missing_entities[:4]))
    if not slots:
        slots.append("需要补充能直接回答原问题的剧情证据")
    return list(dict.fromkeys(slots))[:6]


def first_hit(
    items: list[dict[str, Any]],
    gold_text: str,
    *,
    max_k: int,
    args: argparse.Namespace,
) -> tuple[int, str, float] | None:
    for rank, item in enumerate(items[:max_k], start=1):
        hit = candidate_hit_source(
            item,
            gold_text,
            jaccard_threshold=args.jaccard_threshold,
            overlap_threshold=args.overlap_threshold,
            min_overlap_grams=args.min_overlap_grams,
            min_candidate_grams=args.min_candidate_grams,
        )
        if hit is not None:
            source, score = hit
            return rank, source, score
    return None


def build_generation_trace_record(
    *,
    round_index: int,
    hypothesis: HypothesisDocument,
    queries: list[str],
    hits: list[dict[str, Any]],
    planner_action: str,
    missing_slots: list[str] | None = None,
) -> dict[str, Any]:
    evidence_summary = []
    for item in hits[:3]:
        doc = item.get("document") or {}
        evidence_summary.append(
            {
                "id": str(doc.get("id") or ""),
                "story_name": str(doc.get("story_name") or ""),
                "stage_code": str(doc.get("stage_code") or ""),
                "snippet": re.sub(r"\s+", " ", str(doc.get("clean_text") or ""))[:80],
            }
        )
    return {
        "round": round_index,
        "queries": queries,
        "planner_action": planner_action,
        "hypothesis_task_type": "user_question_hypothesis_generation"
        if round_index == 1
        else "follow_up_hypothesis_generation",
        "hypothesis": asdict(hypothesis),
        "evidence_summary": evidence_summary,
        "missing_slots": missing_slots or [],
    }


def evaluate_record(
    record: dict[str, Any],
    *,
    pipeline: CPUInferencePipeline,
    query_config: QueryConfig,
    top_ks: list[int],
    max_rounds: int,
    planner_mode: str,
    dialogue_context: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    gold_text = extract_gold_text(record)
    question = str(record.get("query") or "").strip()
    if not gold_text or not question:
        return None

    query_type = str(record.get("query_type") or "unknown")
    current_hypothesis = pipeline.build_hypothesis(question, dialogue_context)
    evidence_pool: list[dict[str, Any]] = []
    retrieval_trace: list[dict[str, Any]] = []
    round_hits: list[dict[str, Any]] = []
    max_k = max(top_ks)
    stopped_reason = ""
    generation_errors: list[str] = []

    for round_index in range(1, max_rounds + 1):
        if round_index == 1:
            queries = [
                _resolve_referential_question(question, current_hypothesis.entities),
                build_retrieval_query(current_hypothesis),
            ]
        else:
            queries = [build_retrieval_query(current_hypothesis)]
        minirag_expansion_record = None
        if (
            round_index == 1
            and pipeline.query_config.minirag_chapter_isolation
            and pipeline.query_config.minirag_auto_second_retrieval
        ):
            _, _, hits, minirag_expansion_record = pipeline._retrieve_first_round_with_scoped_minirag_expansion(
                question,
                current_hypothesis,
                queries,
            )
        else:
            _, _, hits = pipeline._retrieve_round(question, current_hypothesis, queries)
        evidence_pool = merge_evidence_pool(evidence_pool, hits)
        missing_slots = infer_missing_slots(
            question=question,
            hypothesis=current_hypothesis,
            evidence=evidence_pool,
            query_type=current_hypothesis.query_type or query_type,
        )
        step_record = build_generation_trace_record(
            round_index=round_index,
            hypothesis=current_hypothesis,
            queries=queries,
            hits=hits,
            planner_action="retrieval_completed",
            missing_slots=missing_slots,
        )
        if minirag_expansion_record is not None:
            step_record["minirag_chapter_expansion"] = minirag_expansion_record
        retrieval_trace.append(step_record)

        round_hit = first_hit(hits, gold_text, max_k=max_k, args=args)
        cumulative_hits_for_k = {
            k: any(
                first_hit(round_info["hits"], gold_text, max_k=k, args=args) is not None
                for round_info in round_hits + [{"hits": hits}]
            )
            for k in top_ks
        }
        round_hits.append(
            {
                "round": round_index,
                "queries": queries,
                "hypothesis": asdict(current_hypothesis),
                "hit": {
                    "rank": round_hit[0],
                    "source": round_hit[1],
                    "score": round_hit[2],
                }
                if round_hit is not None
                else None,
                "cumulative_hits": cumulative_hits_for_k,
                "hits": hits,
            }
        )

        if round_index >= max_rounds:
            break

        try:
            if planner_mode == "conclusion":
                conclusion = pipeline.generate_conclusion(
                    question,
                    current_hypothesis,
                    evidence_pool,
                    retrieval_trace,
                    round_index,
                )
                step_record["conclusion"] = asdict(conclusion)
                step_record["planner_action"] = conclusion.next_action
                step_record["missing_slots"] = conclusion.missing_slots
                if conclusion.next_action != "retrieve_more":
                    stopped_reason = conclusion.next_action
                    break
                if conclusion.follow_up_hypothesis is not None:
                    follow_up = conclusion.follow_up_hypothesis
                else:
                    follow_up = pipeline.build_follow_up_hypothesis(
                        question,
                        current_hypothesis,
                        evidence_pool,
                        retrieval_trace,
                        conclusion,
                        round_index + 1,
                    )
            else:
                conclusion = ConclusionResult(
                    next_action="retrieve_more",
                    answer="",
                    missing_slots=missing_slots,
                    clarification_question="",
                    follow_up_hypothesis=None,
                )
                follow_up = pipeline.build_follow_up_hypothesis(
                    question,
                    current_hypothesis,
                    evidence_pool,
                    retrieval_trace,
                    conclusion,
                    round_index + 1,
                )
            current_hypothesis = merge_hypotheses(current_hypothesis, follow_up)
            step_record["follow_up_hypothesis"] = asdict(current_hypothesis)
        except Exception as exc:  # Keep the eval running; this is a generation failure metric.
            generation_errors.append(f"round={round_index + 1}: {type(exc).__name__}: {exc}")
            stopped_reason = "generation_error"
            break

    cumulative_hit = {
        k: any(bool(round_info["cumulative_hits"][k]) for round_info in round_hits)
        for k in top_ks
    }
    first_hit_round = None
    first_hit_rank = None
    first_hit_source = ""
    first_hit_score = 0.0
    for round_info in round_hits:
        hit = round_info.get("hit")
        if hit is not None:
            first_hit_round = int(round_info["round"])
            first_hit_rank = int(hit["rank"])
            first_hit_source = str(hit["source"])
            first_hit_score = float(hit["score"])
            break

    return {
        "query": question,
        "query_type": query_type,
        "gold_excerpt": normalize_text(gold_text)[:220],
        "rounds_run": len(round_hits),
        "stopped_reason": stopped_reason,
        "first_hit_round": first_hit_round,
        "first_hit_rank": first_hit_rank,
        "first_hit_source": first_hit_source,
        "first_hit_score": first_hit_score,
        "cumulative_hit": cumulative_hit,
        "generation_errors": generation_errors,
        "rounds": [
            {
                "round": round_info["round"],
                "queries": round_info["queries"],
                "hypothesis": round_info["hypothesis"],
                "hit": round_info["hit"],
                "cumulative_hits": round_info["cumulative_hits"],
                "top_doc_ids": [
                    str(item.get("document", {}).get("id") or "") for item in round_info["hits"][:5]
                ],
            }
            for round_info in round_hits
        ],
    }


def summarize_results(records: list[dict[str, Any]], *, top_ks: list[int], max_rounds: int) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"overall": {"count": 0}, "by_query_type": {}, "sample_misses": []}

    overall_hits = {k: sum(1 for record in records if record["cumulative_hit"][k]) for k in top_ks}
    rr_sum = 0.0
    first_round_sum = 0
    first_rank_sum = 0
    hit_count = 0
    hit_sources: dict[str, int] = {}
    score_sum = 0.0
    for record in records:
        if record["first_hit_round"] is None or record["first_hit_rank"] is None:
            continue
        hit_count += 1
        global_rank = (int(record["first_hit_round"]) - 1) * max(top_ks) + int(record["first_hit_rank"])
        rr_sum += 1.0 / max(global_rank, 1)
        first_round_sum += int(record["first_hit_round"])
        first_rank_sum += int(record["first_hit_rank"])
        source = str(record.get("first_hit_source") or "")
        if source:
            hit_sources[source] = hit_sources.get(source, 0) + 1
        score_sum += float(record.get("first_hit_score") or 0.0)

    round_recall: dict[str, dict[str, float]] = {}
    cumulative_by_round: dict[str, dict[str, float]] = {}
    for round_index in range(1, max_rounds + 1):
        round_bucket: dict[str, float] = {}
        cumulative_bucket: dict[str, float] = {}
        for k in top_ks:
            round_hits = 0
            cumulative_hits = 0
            for record in records:
                round_info = next(
                    (item for item in record["rounds"] if int(item["round"]) == round_index),
                    None,
                )
                if round_info and round_info["hit"] is not None and int(round_info["hit"]["rank"]) <= k:
                    round_hits += 1
                if any(
                    item["cumulative_hits"].get(k, False)
                    for item in record["rounds"]
                    if int(item["round"]) <= round_index
                ):
                    cumulative_hits += 1
            round_bucket[f"@{k}"] = round(round_hits / count, 4)
            cumulative_bucket[f"@{k}"] = round(cumulative_hits / count, 4)
        round_recall[f"round_{round_index}"] = round_bucket
        cumulative_by_round[f"round_{round_index}"] = cumulative_bucket

    per_type_raw: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        per_type_raw.setdefault(str(record.get("query_type") or "unknown"), []).append(record)

    def summarize_bucket(items: list[dict[str, Any]]) -> dict[str, Any]:
        bucket_count = len(items)
        bucket_hits = {k: sum(1 for item in items if item["cumulative_hit"][k]) for k in top_ks}
        return {
            "count": bucket_count,
            "missed": bucket_count - bucket_hits[max(top_ks)],
            "recall": {f"@{k}": round(bucket_hits[k] / bucket_count, 4) for k in top_ks},
        }

    sample_misses = [
        {
            "query": record["query"],
            "query_type": record["query_type"],
            "rounds_run": record["rounds_run"],
            "gold_excerpt": record["gold_excerpt"],
            "last_queries": record["rounds"][-1]["queries"] if record["rounds"] else [],
            "top_doc_ids": record["rounds"][-1]["top_doc_ids"] if record["rounds"] else [],
        }
        for record in records
        if not record["cumulative_hit"][max(top_ks)]
    ][:20]

    return {
        "overall": {
            "count": count,
            "missed": count - overall_hits[max(top_ks)],
            "mrr_global_round_rank": round(rr_sum / count, 4),
            "mean_first_hit_round": round(first_round_sum / max(hit_count, 1), 3),
            "mean_first_hit_rank_in_round": round(first_rank_sum / max(hit_count, 1), 3),
            "hit_sources": hit_sources,
            "mean_hit_score": round(score_sum / max(hit_count, 1), 4),
            "recall": {f"@{k}": round(overall_hits[k] / count, 4) for k in top_ks},
        },
        "round_recall": round_recall,
        "cumulative_recall_by_round": cumulative_by_round,
        "by_query_type": {
            query_type: summarize_bucket(items) for query_type, items in sorted(per_type_raw.items())
        },
        "sample_misses": sample_misses,
        "generation_error_count": sum(len(record["generation_errors"]) for record in records),
    }


def main() -> int:
    args = parse_args()
    runtime_config_path = args.runtime_config if args.runtime_config.is_absolute() else PROJECT_ROOT / args.runtime_config
    runtime_config = load_runtime_config(runtime_config_path)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}

    listwise_path = args.listwise if args.listwise.is_absolute() else PROJECT_ROOT / args.listwise
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    index_dir = args.index_dir if args.index_dir.is_absolute() else PROJECT_ROOT / args.index_dir
    documents_path = index_dir / "documents.jsonl"
    faiss_index_path = index_dir / "faiss.index"
    bm25_tokens_path = index_dir / "bm25_tokens.pkl"
    device = str(config_value(args.device, retrieval_cfg, "device", "cuda"))
    max_rounds = min(2, max(1, int(config_value(args.max_rounds, inference_cfg, "max_retrieval_rounds", 2))))
    max_k = max(args.top_ks)

    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True)) and not args.no_reranker
    configured_reranker = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
    reranker_model = (
        resolve_path(args.reranker_model if args.reranker_model is not None else configured_reranker, default=RERANKER_MODEL_DIR)
        if enable_reranker
        else None
    )
    reranker_max_length = int(config_value(args.reranker_max_length, retrieval_cfg, "reranker_max_length", 1024))
    minirag_index = resolve_path(
        args.minirag_index if args.minirag_index is not None else retrieval_cfg.get("minirag_index_path"),
        default=MINIRAG_GRAPH_PATH if bool(retrieval_cfg.get("enable_minirag", True)) else None,
    )

    all_records = load_listwise(listwise_path)
    records = list(all_records)
    if args.sample_seed is not None:
        rng = random.Random(args.sample_seed)
        rng.shuffle(records)
    if args.sample_offset > 0:
        records = records[args.sample_offset :]
    if args.sample is not None:
        records = records[: args.sample]

    sys.stderr.write(
        f"Loading retriever from {index_dir} on device={device} records={len(records)}\n"
    )
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=reranker_max_length,
        documents_path=documents_path if documents_path.exists() else DOCUMENTS_PATH,
        faiss_index_path=faiss_index_path if faiss_index_path.exists() else FAISS_INDEX_PATH,
        bm25_tokens_path=bm25_tokens_path if bm25_tokens_path.exists() else BM25_TOKENS_PATH,
        minirag_index_path=minirag_index,
        device=device,
    )
    query_config = build_query_config(args, retrieval_cfg, rerank_top_k=max_k)
    generator = build_generator(args, generator_cfg)
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=query_config,
        max_retrieval_rounds=max_rounds,
        prompt_evidence_top_k=int(config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 8)),
        enable_mmr=bool(config_value(args.enable_mmr, inference_cfg, "enable_mmr", False)),
        mmr_lambda=float(config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72)),
        enable_pyramid_order=bool(config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)),
        enable_evidence_pinning=bool(
            config_value(args.enable_evidence_pinning, inference_cfg, "enable_evidence_pinning", False)
        ),
        enable_crag_refinement=bool(config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)),
        crag_refine_top_sentences=int(
            config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)
        ),
        crag_refine_max_sentences=int(
            config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)
        ),
        self_consistency_samples=int(
            config_value(args.self_consistency_samples, inference_cfg, "self_consistency_samples", 1)
        ),
        self_consistency_temperature=float(
            config_value(args.self_consistency_temperature, inference_cfg, "self_consistency_temperature", 0.7)
        ),
        answer_grounding_mode=str(inference_cfg.get("answer_grounding_mode", "weak")),
        conclusion_prompt_mode=str(
            config_value(args.conclusion_prompt_mode, inference_cfg, "conclusion_prompt_mode", "full")
        ),
    )

    started = time.time()
    evaluated: list[dict[str, Any]] = []
    skipped = 0
    for index, record in enumerate(records, start=1):
        try:
            result = evaluate_record(
                record,
                pipeline=pipeline,
                query_config=query_config,
                top_ks=args.top_ks,
                max_rounds=max_rounds,
                planner_mode=args.planner_mode,
                dialogue_context=args.dialogue_context,
                args=args,
            )
        except Exception as exc:
            if is_fatal_record_error(exc):
                raise
            query = str(record.get("query") or "").strip()
            result = {
                "query": query,
                "query_type": str(record.get("query_type") or "unknown"),
                "gold_excerpt": normalize_text(extract_gold_text(record) or "")[:220],
                "rounds_run": 0,
                "stopped_reason": "record_error",
                "first_hit_round": None,
                "first_hit_rank": None,
                "first_hit_source": "",
                "first_hit_score": 0.0,
                "cumulative_hit": {k: False for k in args.top_ks},
                "generation_errors": [f"{type(exc).__name__}: {exc}"],
                "rounds": [],
            }
        if result is None:
            skipped += 1
            continue
        evaluated.append(result)
        if args.progress_every > 0 and index % args.progress_every == 0:
            elapsed = time.time() - started
            current_summary = summarize_results(evaluated, top_ks=args.top_ks, max_rounds=max_rounds)["overall"]
            sys.stderr.write(
                f"[{index}/{len(records)}] elapsed={elapsed:.1f}s "
                f"R@{max_k}={current_summary['recall'][f'@{max_k}']:.3f} "
                f"missed={current_summary['missed']}\n"
            )
            sys.stderr.flush()

    summary = summarize_results(evaluated, top_ks=args.top_ks, max_rounds=max_rounds)
    payload = {
        "tag": args.tag,
        "runtime_config": str(runtime_config_path),
        "listwise_path": str(listwise_path),
        "index_dir": str(index_dir),
        "device": device,
        "planner_mode": args.planner_mode,
        "max_rounds": max_rounds,
        "top_ks": args.top_ks,
        "sample": {
            "source_count": len(all_records),
            "count": len(records),
            "offset": args.sample_offset,
            "seed": args.sample_seed,
        },
        "skipped": skipped,
        "query_config": asdict(query_config),
        "generator_runtime": generator.describe_runtime(),
        "prompt_dialogue_context": render_dialogue_context_for_prompt(args.dialogue_context),
        "jaccard_threshold": args.jaccard_threshold,
        "overlap_threshold": args.overlap_threshold,
        "min_overlap_grams": args.min_overlap_grams,
        "min_candidate_grams": args.min_candidate_grams,
        "wall_seconds": round(time.time() - started, 2),
        **summary,
        "records": evaluated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    print("round_recall:")
    print(json.dumps(payload["round_recall"], ensure_ascii=False, indent=2))
    print("cumulative_recall_by_round:")
    print(json.dumps(payload["cumulative_recall_by_round"], ensure_ascii=False, indent=2))
    print("by_query_type:")
    print(json.dumps(payload["by_query_type"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
