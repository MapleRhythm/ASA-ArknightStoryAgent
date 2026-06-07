#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    if conda_env == "train":
        return True
    executable = Path(sys.executable).as_posix().lower()
    return "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


if should_use_train_overrides():
    if TRAIN_PYTHON_OVERLAY_DIR.exists():
        sys.path.insert(0, str(TRAIN_PYTHON_OVERLAY_DIR))
    if TRAIN_OVERRIDE_DIR.exists():
        sys.path.insert(0, str(TRAIN_OVERRIDE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    MINIRAG_GRAPH_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)

DEFAULT_LLAMA_CLI_PATH = PROJECT_ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-completion"
DEFAULT_GGUF_MODEL_PATH = (
    PROJECT_ROOT / "model" / "gguf" / "teacher_v2_plus_prompt_supplement_v2_qwen35_4b-merged-q4_k_m.gguf"
)
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "model" / "qwen3.5-4b"
DEFAULT_VLLM_LORA_PATH = (
    PROJECT_ROOT
    / "model"
    / "lora"
    / "teacher_scored_kto_mix_v1_from_soda_lora_qwen35_4b_lr8e7_beta001_epoch2"
)
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_inference.json"


def load_runtime_config(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_config_value(cli_value, config_section: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def resolve_path_value(cli_value, config_section: dict, key: str, default: Path | None) -> Path | None:
    value = cli_value if cli_value is not None else config_section.get(key, default)
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def validate_vllm_lora_path(lora_path: Path | None) -> None:
    if lora_path is None:
        return
    if (lora_path / "adapter_config.json").exists():
        return
    child_adapters = sorted(
        child
        for child in lora_path.glob("*")
        if child.is_dir() and (child / "adapter_config.json").exists()
    ) if lora_path.is_dir() else []
    hint = ""
    if child_adapters:
        hint = "\nCandidate adapter dirs:\n" + "\n".join(f"  {child}" for child in child_adapters[:8])
    raise SystemExit(
        "Invalid vLLM LoRA path: "
        f"{lora_path}. Pass a specific adapter directory containing adapter_config.json, "
        "not the parent model/lora directory."
        + hint
    )


def detect_visible_gpu_count() -> int:
    raw_devices = [item.strip() for item in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",") if item.strip()]
    return len(raw_devices) if raw_devices else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full inference pipeline.")
    parser.add_argument("question", type=str, nargs="?", default=None, help="Optional initial user question in Chinese.")
    parser.add_argument("--dialogue-context", type=str, default="", help="Optional multi-turn context.")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG_PATH,
        help="Path to runtime inference config JSON.",
    )
    parser.add_argument("--device", type=str, default=None, help="Retrieval device. Overrides runtime config.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("llama.cpp", "vllm"),
        default=None,
        help="Generation backend. Defaults to runtime config.",
    )
    parser.add_argument(
        "--llama-cli",
        type=Path,
        default=DEFAULT_LLAMA_CLI_PATH,
        help="Path to the llama.cpp CLI binary.",
    )
    parser.add_argument(
        "--gguf-model",
        type=Path,
        default=DEFAULT_GGUF_MODEL_PATH,
        help="Path to the GGUF model used for inference.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        default=None,
        help="Path to the Hugging Face base model for vLLM inference. Defaults to runtime config or model/qwen3.5-4b.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help=(
            "Optional adapter path. For llama.cpp, pass a GGUF LoRA file; for vLLM, pass a Hugging Face LoRA directory. "
            "Leave unset when using a merged GGUF model."
        ),
    )
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Do not load a LoRA adapter. Use this with --base-model pointing to a merged Hugging Face model.",
    )
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=None,
        help="Path to the reranker model. Overrides runtime config when provided.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable reranker for this run, overriding runtime config.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--llama-device", type=str, default=None, help="llama.cpp offload device, e.g. cuda or cpu.")
    parser.add_argument("--llama-gpu-layers", type=str, default=None, help="llama.cpp GPU layers, e.g. all, auto, 40.")
    parser.add_argument("--llama-batch-size", type=int, default=None)
    parser.add_argument("--llama-ubatch-size", type=int, default=None)
    parser.add_argument("--llama-flash-attn", type=str, choices=("on", "off", "auto"), default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--max-retrieval-rounds", type=int, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--enable-minirag", dest="enable_minirag", action="store_true", default=None)
    parser.add_argument("--disable-minirag", dest="enable_minirag", action="store_false")
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--minirag-top-k", type=int, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
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
    parser.add_argument("--reranker-candidate-top-k", type=int, default=None)
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
    parser.add_argument("--prompt-evidence-max-chars-per-doc", type=int, default=None)
    parser.add_argument("--prompt-conclusion-evidence-max-total-chars", type=int, default=None)
    parser.add_argument("--conclusion-prompt-mode", choices=("full", "minimal"), default=None)
    parser.add_argument(
        "--answer-grounding-mode",
        choices=("off", "weak", "strict", "quote", "grounded"),
        default=None,
        help="Runtime grounding guard. quote requires answer_directly quotes to match visible evidence text.",
    )
    parser.add_argument("--self-consistency-samples", type=int, default=None)
    parser.add_argument("--self-consistency-temperature", type=float, default=None)
    parser.add_argument("--enable-web-context", dest="enable_web_context", action="store_true", default=None)
    parser.add_argument("--disable-web-context", dest="enable_web_context", action="store_false")
    parser.add_argument("--web-context-max-pages", type=int, default=None)
    parser.add_argument("--web-context-timeout-seconds", type=float, default=None)
    parser.add_argument("--web-context-max-elapsed-seconds", type=float, default=None)
    parser.add_argument("--web-context-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--build-index-if-missing",
        action="store_true",
        help="Build the retrieval index automatically if missing.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Print only the final answer instead of the full JSON payload.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Optional UTF-8 text file with one question per line. Runs stateless batch mode.",
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        default=None,
        help="Optional JSONL output path for --questions-file results.",
    )
    return parser.parse_args()


def append_dialogue_turn(history: list[str], role: str, content: str) -> None:
    text = content.strip()
    if not text:
        return
    history.append(f"{role}: {text}")


def render_dialogue_context(history: list[str]) -> str:
    return "\n".join(history).strip()


ABSTAIN_TEXT_MARKERS = (
    "现有检索证据不足",
    "现有证据不足",
    "不足以确认",
    "无法确认",
    "无法判断",
    "不能确认",
    "没有足够",
    "未能找到",
    "无法回答",
    "当前证据只能确认",
)


def final_planner_action(payload: dict) -> str:
    trace = payload.get("retrieval_trace")
    if not isinstance(trace, list) or not trace:
        return ""
    last_step = trace[-1]
    if not isinstance(last_step, dict):
        return ""
    return str(last_step.get("planner_action") or "").strip()


def is_abstain_like_payload(payload: dict) -> bool:
    if final_planner_action(payload) == "abstain":
        return True
    answer = str(payload.get("answer") or "")
    return any(marker in answer for marker in ABSTAIN_TEXT_MARKERS)


def compact_abstain_evidence(payload: dict, *, limit: int = 3, max_chars: int = 900) -> list[dict]:
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return []
    compact: list[dict] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("clean_text") or "").split())
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        compact.append(
            {
                "rank": len(compact) + 1,
                "id": item.get("id"),
                "activity_name": item.get("activity_name"),
                "story_name": item.get("story_name"),
                "stage_code": item.get("stage_code"),
                "avg_tag": item.get("avg_tag"),
                "source_path": item.get("source_path"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "clean_text": text,
            }
        )
        if len(compact) >= limit:
            break
    return compact


def attach_abstain_evidence(payload: dict) -> dict:
    if not is_abstain_like_payload(payload):
        return payload
    payload["abstain_evidence_top3"] = compact_abstain_evidence(payload, limit=3)
    payload["abstain_evidence_trigger"] = {
        "final_action": final_planner_action(payload),
        "answer_markers": [
            marker for marker in ABSTAIN_TEXT_MARKERS if marker in str(payload.get("answer") or "")
        ],
    }
    return payload


def ensure_index(args: argparse.Namespace) -> None:
    if DOCUMENTS_PATH.exists() and FAISS_INDEX_PATH.exists() and BM25_TOKENS_PATH.exists():
        return
    if not args.build_index_if_missing:
        raise FileNotFoundError(
            "Retrieval index is missing. Run `python scripts/build_retrieval_index.py --device cpu` "
            "or add `--build-index-if-missing`."
        )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_retrieval_index.py"), "--device", "cpu"],
        check=True,
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    args = parse_args()
    ensure_index(args)
    runtime_config = load_runtime_config(args.runtime_config)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}
    llama_cpp_cfg = generator_cfg.get("llama_cpp", {}) if isinstance(generator_cfg.get("llama_cpp"), dict) else {}
    vllm_cfg = generator_cfg.get("vllm", {}) if isinstance(generator_cfg.get("vllm"), dict) else {}

    device = resolve_config_value(args.device, retrieval_cfg, "device", "cpu")
    backend = resolve_config_value(args.backend, generator_cfg, "backend", "llama.cpp")
    dense_top_k = int(resolve_config_value(args.dense_top_k, retrieval_cfg, "dense_top_k", 60))
    sparse_top_k = int(resolve_config_value(args.sparse_top_k, retrieval_cfg, "sparse_top_k", 60))
    fusion_top_k = int(resolve_config_value(args.fusion_top_k, retrieval_cfg, "fusion_top_k", 40))
    rerank_top_k = int(resolve_config_value(args.rerank_top_k, retrieval_cfg, "rerank_top_k", 15))
    rerank_batch_size = int(
        resolve_config_value(args.rerank_batch_size, retrieval_cfg, "rerank_batch_size", 8)
    )
    enable_neighbor_expansion = bool(retrieval_cfg.get("enable_neighbor_expansion", False))
    neighbor_max_seed_docs = int(retrieval_cfg.get("neighbor_max_seed_docs", 24))
    neighbor_story_window = int(retrieval_cfg.get("neighbor_story_window", 2))
    neighbor_activity_story_sort_window = int(
        retrieval_cfg.get("neighbor_activity_story_sort_window", 1)
    )
    enable_scoped_chapter_search = bool(retrieval_cfg.get("enable_scoped_chapter_search", True))
    scoped_chapter_dense_top_k = int(retrieval_cfg.get("scoped_chapter_dense_top_k", 160))
    scoped_chapter_sparse_top_k = int(retrieval_cfg.get("scoped_chapter_sparse_top_k", 160))
    enable_same_story_sweep = bool(retrieval_cfg.get("enable_same_story_sweep", True))
    same_story_sweep_max_seed_docs = int(retrieval_cfg.get("same_story_sweep_max_seed_docs", 8))
    same_story_sweep_max_docs_per_story = int(retrieval_cfg.get("same_story_sweep_max_docs_per_story", 24))
    same_story_sweep_extra_candidates = int(retrieval_cfg.get("same_story_sweep_extra_candidates", 80))
    reranker_max_length = int(retrieval_cfg.get("reranker_max_length", 1024))
    max_retrieval_rounds = resolve_config_value(
        args.max_retrieval_rounds,
        inference_cfg,
        "max_retrieval_rounds",
        None,
    )
    if max_retrieval_rounds is None:
        max_retrieval_rounds = 2
    max_retrieval_rounds = min(2, max(1, int(max_retrieval_rounds)))
    prompt_evidence_top_k = int(inference_cfg.get("prompt_evidence_top_k", 8))
    use_model_hypothesis = bool(inference_cfg.get("use_model_hypothesis", True))
    use_model_conclusion_generation = bool(
        inference_cfg.get(
            "use_model_conclusion_generation",
            inference_cfg.get("use_model_retrieval_planner", True),
        )
    )
    enable_mmr = bool(resolve_config_value(args.enable_mmr, inference_cfg, "enable_mmr", False))
    mmr_lambda = float(resolve_config_value(args.mmr_lambda, inference_cfg, "mmr_lambda", 0.72))
    enable_pyramid_order = bool(
        resolve_config_value(args.enable_pyramid_order, inference_cfg, "enable_pyramid_order", False)
    )
    enable_evidence_pinning = bool(
        resolve_config_value(args.enable_evidence_pinning, inference_cfg, "enable_evidence_pinning", False)
    )
    enable_crag_refinement = bool(
        resolve_config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)
    )
    crag_refine_top_sentences = int(
        resolve_config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)
    )
    crag_refine_max_sentences = int(
        resolve_config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)
    )
    prompt_evidence_max_chars_per_doc = int(
        resolve_config_value(
            args.prompt_evidence_max_chars_per_doc,
            inference_cfg,
            "prompt_evidence_max_chars_per_doc",
            520,
        )
    )
    prompt_conclusion_evidence_max_total_chars = int(
        resolve_config_value(
            args.prompt_conclusion_evidence_max_total_chars,
            inference_cfg,
            "prompt_conclusion_evidence_max_total_chars",
            5000,
        )
    )
    self_consistency_samples = int(
        resolve_config_value(args.self_consistency_samples, inference_cfg, "self_consistency_samples", 1)
    )
    self_consistency_temperature = float(
        resolve_config_value(args.self_consistency_temperature, inference_cfg, "self_consistency_temperature", 0.7)
    )
    answer_grounding_mode = str(
        resolve_config_value(args.answer_grounding_mode, inference_cfg, "answer_grounding_mode", "weak")
    )
    conclusion_prompt_mode = str(
        resolve_config_value(args.conclusion_prompt_mode, inference_cfg, "conclusion_prompt_mode", "full")
    )
    web_context_cfg = inference_cfg.get("web_context", {}) if isinstance(inference_cfg.get("web_context"), dict) else {}
    web_context_cfg = dict(web_context_cfg)
    web_context_cfg["enabled"] = bool(
        resolve_config_value(args.enable_web_context, web_context_cfg, "enabled", False)
    )
    if args.web_context_max_pages is not None:
        web_context_cfg["max_pages"] = args.web_context_max_pages
    if args.web_context_timeout_seconds is not None:
        web_context_cfg["timeout_seconds"] = args.web_context_timeout_seconds
    if args.web_context_max_elapsed_seconds is not None:
        web_context_cfg["max_elapsed_seconds"] = args.web_context_max_elapsed_seconds
    if args.web_context_cache_dir is not None:
        web_context_cfg["cache_dir"] = str(args.web_context_cache_dir)
    if web_context_cfg.get("cache_dir"):
        cache_dir = Path(str(web_context_cfg["cache_dir"]))
        if not cache_dir.is_absolute():
            cache_dir = PROJECT_ROOT / cache_dir
        web_context_cfg["cache_dir"] = str(cache_dir)
    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True))
    if args.no_reranker:
        enable_reranker = False
    ctx_size = int(resolve_config_value(args.ctx_size, generator_cfg, "ctx_size", 12000))
    max_tokens = int(resolve_config_value(args.max_tokens, generator_cfg, "max_tokens", 512))
    temperature = float(resolve_config_value(args.temperature, generator_cfg, "temperature", 0.2))
    top_p = float(resolve_config_value(args.top_p, generator_cfg, "top_p", 0.9))
    repeat_penalty = float(resolve_config_value(args.repeat_penalty, generator_cfg, "repeat_penalty", 1.05))

    reranker_model = None
    if enable_reranker:
        configured_reranker = retrieval_cfg.get("reranker_model_path") or retrieval_cfg.get("reranker_model")
        reranker_model = resolve_path_value(
            args.reranker_model,
            retrieval_cfg,
            "reranker_model_path" if retrieval_cfg.get("reranker_model_path") else "reranker_model",
            RERANKER_MODEL_DIR if configured_reranker is None else configured_reranker,
        )
        if reranker_model is None or not (reranker_model / "config.json").exists():
            raise SystemExit(
                "Invalid reranker model path: "
                f"{reranker_model or '<empty>'}. "
                "Pass a directory containing config.json, for example "
                "model/reranker/bge-reranker-v2-m3-evidence-chain-answerability."
            )
    if not enable_reranker:
        reranker_model = None
    enable_minirag = bool(resolve_config_value(args.enable_minirag, retrieval_cfg, "enable_minirag", False))
    minirag_top_k = int(resolve_config_value(args.minirag_top_k, retrieval_cfg, "minirag_top_k", 120))
    minirag_weight = float(resolve_config_value(args.minirag_weight, retrieval_cfg, "minirag_weight", 0.35))
    minirag_fusion_mode = str(
        resolve_config_value(args.minirag_fusion_mode, retrieval_cfg, "minirag_fusion_mode", "score")
    )
    minirag_chapter_isolation = bool(
        resolve_config_value(args.minirag_chapter_isolation, retrieval_cfg, "minirag_chapter_isolation", True)
    )
    minirag_auto_second_retrieval = bool(
        resolve_config_value(
            args.minirag_auto_second_retrieval,
            retrieval_cfg,
            "minirag_auto_second_retrieval",
            True,
        )
    )
    minirag_scope_seed_top_k = int(
        resolve_config_value(args.minirag_scope_seed_top_k, retrieval_cfg, "minirag_scope_seed_top_k", 40)
    )
    minirag_expansion_query_top_k = int(
        resolve_config_value(
            args.minirag_expansion_query_top_k,
            retrieval_cfg,
            "minirag_expansion_query_top_k",
            8,
        )
    )
    minirag_graph_scope_min_ratio = float(
        resolve_config_value(args.minirag_graph_scope_min_ratio, retrieval_cfg, "minirag_graph_scope_min_ratio", 1.0)
    )
    minirag_second_pass_scope_min_ratio = float(
        resolve_config_value(
            args.minirag_second_pass_scope_min_ratio,
            retrieval_cfg,
            "minirag_second_pass_scope_min_ratio",
            1.0,
        )
    )
    enable_storyline_sparse_scope = bool(
        resolve_config_value(args.enable_storyline_sparse_scope, retrieval_cfg, "enable_storyline_sparse_scope", True)
    )
    storyline_scope_seed_top_k = int(
        resolve_config_value(args.storyline_scope_seed_top_k, retrieval_cfg, "storyline_scope_seed_top_k", 40)
    )
    storyline_sparse_scope_min_ratio = float(
        resolve_config_value(
            args.storyline_sparse_scope_min_ratio,
            retrieval_cfg,
            "storyline_sparse_scope_min_ratio",
            1.5,
        )
    )
    reranker_candidate_top_k = int(
        resolve_config_value(args.reranker_candidate_top_k, retrieval_cfg, "reranker_candidate_top_k", 120)
    )
    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}
    minirag_index_path = None
    if enable_minirag:
        minirag_index_path = resolve_path_value(
            args.minirag_index,
            retrieval_cfg,
            "minirag_index_path",
            MINIRAG_GRAPH_PATH,
        )
        if minirag_index_path is None or not minirag_index_path.exists():
            raise SystemExit(
                "MiniRAG index is enabled but missing. Build it with "
                "`python scripts/build_minirag_index.py` or disable retrieval.enable_minirag."
            )

    from goldenglow.inference import CPUInferencePipeline  # noqa: E402
    from goldenglow.inference.cpu_pipeline import LlamaCppRunner, VllmRunner  # noqa: E402
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=reranker_max_length,
        minirag_index_path=minirag_index_path,
        device=device,
    )
    if backend == "vllm":
        base_model = resolve_path_value(args.base_model, vllm_cfg, "base_model_path", DEFAULT_BASE_MODEL_PATH)
        lora_path = None
        if not args.disable_lora:
            lora_path = resolve_path_value(
                args.lora_path,
                vllm_cfg,
                "lora_path",
                DEFAULT_VLLM_LORA_PATH if DEFAULT_VLLM_LORA_PATH.exists() else None,
            )
        validate_vllm_lora_path(lora_path)
        tensor_parallel_size = int(
            resolve_config_value(args.tensor_parallel_size, vllm_cfg, "tensor_parallel_size", detect_visible_gpu_count())
        )
        gpu_memory_utilization = float(
            resolve_config_value(args.gpu_memory_utilization, vllm_cfg, "gpu_memory_utilization", 0.9)
        )
        dtype = str(resolve_config_value(args.dtype, vllm_cfg, "dtype", "auto"))
        max_num_batched_tokens = resolve_config_value(
            args.max_num_batched_tokens,
            vllm_cfg,
            "max_num_batched_tokens",
            None,
        )
        enforce_eager = bool(resolve_config_value(args.enforce_eager, vllm_cfg, "enforce_eager", False))
        generator = VllmRunner(
            base_model_path=base_model or DEFAULT_BASE_MODEL_PATH,
            lora_path=lora_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=int(resolve_config_value(args.ctx_size, vllm_cfg, "max_model_len", ctx_size)),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            dtype=dtype,
            max_num_batched_tokens=int(max_num_batched_tokens) if max_num_batched_tokens is not None else None,
            enforce_eager=enforce_eager,
        )
    else:
        llama_cli = resolve_path_value(args.llama_cli, llama_cpp_cfg, "llama_cli_path", DEFAULT_LLAMA_CLI_PATH)
        gguf_model = resolve_path_value(args.gguf_model, llama_cpp_cfg, "gguf_model_path", DEFAULT_GGUF_MODEL_PATH)
        lora_path = resolve_path_value(args.lora_path, llama_cpp_cfg, "lora_path", None)
        llama_device = resolve_config_value(args.llama_device, llama_cpp_cfg, "device", None)
        llama_gpu_layers = resolve_config_value(args.llama_gpu_layers, llama_cpp_cfg, "gpu_layers", None)
        llama_batch_size = resolve_config_value(args.llama_batch_size, llama_cpp_cfg, "batch_size", None)
        llama_ubatch_size = resolve_config_value(args.llama_ubatch_size, llama_cpp_cfg, "ubatch_size", None)
        llama_flash_attn = resolve_config_value(args.llama_flash_attn, llama_cpp_cfg, "flash_attn", None)
        generator = LlamaCppRunner(
            llama_cli_path=llama_cli or DEFAULT_LLAMA_CLI_PATH,
            gguf_model_path=gguf_model or DEFAULT_GGUF_MODEL_PATH,
            lora_path=lora_path,
            threads=args.threads,
            ctx_size=int(resolve_config_value(args.ctx_size, llama_cpp_cfg, "ctx_size", ctx_size)),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            device=llama_device,
            gpu_layers=llama_gpu_layers,
            batch_size=int(llama_batch_size) if llama_batch_size is not None else None,
            ubatch_size=int(llama_ubatch_size) if llama_ubatch_size is not None else None,
            flash_attn=llama_flash_attn,
        )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=QueryConfig(
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            minirag_top_k=minirag_top_k,
            fusion_top_k=fusion_top_k,
            rerank_top_k=rerank_top_k,
            minirag_weight=minirag_weight,
            minirag_mode_weights={str(key): float(value) for key, value in minirag_mode_weights.items()},
            minirag_fusion_mode=minirag_fusion_mode,
            minirag_chapter_isolation=minirag_chapter_isolation,
            minirag_auto_second_retrieval=minirag_auto_second_retrieval,
            minirag_scope_seed_top_k=minirag_scope_seed_top_k,
            minirag_expansion_query_top_k=minirag_expansion_query_top_k,
            minirag_graph_scope_min_ratio=minirag_graph_scope_min_ratio,
            minirag_second_pass_scope_min_ratio=minirag_second_pass_scope_min_ratio,
            enable_storyline_sparse_scope=enable_storyline_sparse_scope,
            storyline_scope_seed_top_k=storyline_scope_seed_top_k,
            storyline_sparse_scope_min_ratio=storyline_sparse_scope_min_ratio,
            enable_scoped_chapter_search=enable_scoped_chapter_search,
            scoped_chapter_dense_top_k=scoped_chapter_dense_top_k,
            scoped_chapter_sparse_top_k=scoped_chapter_sparse_top_k,
            reranker_candidate_top_k=reranker_candidate_top_k,
            enable_neighbor_expansion=enable_neighbor_expansion,
            neighbor_max_seed_docs=neighbor_max_seed_docs,
            neighbor_story_window=neighbor_story_window,
            neighbor_activity_story_sort_window=neighbor_activity_story_sort_window,
            enable_same_story_sweep=enable_same_story_sweep,
            same_story_sweep_max_seed_docs=same_story_sweep_max_seed_docs,
            same_story_sweep_max_docs_per_story=same_story_sweep_max_docs_per_story,
            same_story_sweep_extra_candidates=same_story_sweep_extra_candidates,
            rerank_batch_size=rerank_batch_size,
        ),
        max_retrieval_rounds=max_retrieval_rounds,
        prompt_evidence_top_k=prompt_evidence_top_k,
        prompt_evidence_max_chars_per_doc=prompt_evidence_max_chars_per_doc,
        prompt_conclusion_evidence_max_total_chars=prompt_conclusion_evidence_max_total_chars,
        enable_mmr=enable_mmr,
        mmr_lambda=mmr_lambda,
        enable_pyramid_order=enable_pyramid_order,
        enable_evidence_pinning=enable_evidence_pinning,
        enable_crag_refinement=enable_crag_refinement,
        crag_refine_top_sentences=crag_refine_top_sentences,
        crag_refine_max_sentences=crag_refine_max_sentences,
        self_consistency_samples=self_consistency_samples,
        self_consistency_temperature=self_consistency_temperature,
        answer_grounding_mode=answer_grounding_mode,
        conclusion_prompt_mode=conclusion_prompt_mode,
        use_model_hypothesis=use_model_hypothesis,
        use_model_conclusion_generation=use_model_conclusion_generation,
        web_context_config=web_context_cfg,
    )
    dialogue_history: list[str] = []
    if args.dialogue_context.strip():
        dialogue_history.extend(
            line.strip()
            for line in args.dialogue_context.splitlines()
            if line.strip()
        )

    if args.questions_file is not None:
        questions = [
            line.strip()
            for line in args.questions_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        output_handle = None
        if args.batch_output is not None:
            args.batch_output.parent.mkdir(parents=True, exist_ok=True)
            output_handle = args.batch_output.open("w", encoding="utf-8")
        try:
            for index, question in enumerate(questions, start=1):
                started = time.perf_counter()
                print(f"[batch {index}/{len(questions)}] {question}", file=sys.stderr, flush=True)
                try:
                    result = pipeline.run(
                        question,
                        args.dialogue_context.strip(),
                        progress_callback=lambda stage: print(f"[stage] {stage}", file=sys.stderr, flush=True),
                    )
                    payload = asdict(result)
                    payload["elapsed_sec"] = round(time.perf_counter() - started, 3)
                    payload["error"] = ""
                    payload = attach_abstain_evidence(payload)
                except Exception as exc:
                    payload = {
                        "question": question,
                        "answer": "",
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                line = json.dumps(payload, ensure_ascii=False)
                if output_handle is not None:
                    output_handle.write(line + "\n")
                    output_handle.flush()
                elif args.answer_only:
                    print(payload.get("answer", ""), flush=True)
                else:
                    print(line, flush=True)
        finally:
            if output_handle is not None:
                output_handle.close()
        return

    pending_question = args.question
    print("Interactive inference ready. Type /exit to quit, /clear to reset dialogue context.", flush=True)

    while True:
        if pending_question is not None:
            question = pending_question.strip()
            pending_question = None
        else:
            try:
                question = input("\nUser> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.", flush=True)
                break

        if not question:
            continue
        if question in {"/exit", "/quit"}:
            print("Exiting.", flush=True)
            break
        if question == "/clear":
            dialogue_history.clear()
            print("Dialogue context cleared.", flush=True)
            continue

        dialogue_context = render_dialogue_context(dialogue_history)
        started = time.perf_counter()
        print("[running] pipeline start...", file=sys.stderr, flush=True)
        try:
            result = pipeline.run(
                question,
                dialogue_context,
                progress_callback=lambda stage: print(f"[stage] {stage}", file=sys.stderr, flush=True),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            print(f"[failed] {elapsed:.2f}s", file=sys.stderr, flush=True)
            continue
        elapsed = time.perf_counter() - started
        print(f"[done] {elapsed:.2f}s", file=sys.stderr, flush=True)

        if args.answer_only:
            print(result.answer, flush=True)
        else:
            print(json.dumps(attach_abstain_evidence(asdict(result)), ensure_ascii=False, indent=2), flush=True)

        append_dialogue_turn(dialogue_history, "user", question)
        append_dialogue_turn(dialogue_history, "assistant", result.answer)


if __name__ == "__main__":
    main()
