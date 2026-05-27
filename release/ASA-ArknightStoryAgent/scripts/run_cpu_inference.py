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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.config import (  # noqa: E402
    BM25_TOKENS_PATH,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_DIR,
    FAISS_INDEX_PATH,
    MINIRAG_GRAPH_PATH,
    QueryConfig,
    RERANKER_MODEL_DIR,
)

DEFAULT_LLAMA_CLI_PATH = PROJECT_ROOT / "third_party" / "llama.cpp" / "build-cpu" / "bin" / "llama-completion"
DEFAULT_GGUF_MODEL_PATH = PROJECT_ROOT / "model" / "gguf" / "qwen3.5-4b-q4_k_m.gguf"
DEFAULT_BASE_MODEL_PATH = PROJECT_ROOT / "model" / "qwen3.5-4b"
DEFAULT_VLLM_LORA_PATH = PROJECT_ROOT / "model" / "lora" / "asa-arknightstoryagent-4b-lora"
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_cpu_qwen35_4b_no_reranker.json"


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
        default=None,
        help="Path to the llama.cpp CLI binary.",
    )
    parser.add_argument(
        "--gguf-model",
        type=Path,
        default=None,
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
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repeat-penalty", type=float, default=None)
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--enable-minirag", dest="enable_minirag", action="store_true", default=None)
    parser.add_argument("--disable-minirag", dest="enable_minirag", action="store_false")
    parser.add_argument("--minirag-index", type=Path, default=None)
    parser.add_argument("--minirag-weight", type=float, default=None)
    parser.add_argument("--enable-mmr", dest="enable_mmr", action="store_true", default=None)
    parser.add_argument("--disable-mmr", dest="enable_mmr", action="store_false")
    parser.add_argument("--mmr-lambda", type=float, default=None)
    parser.add_argument("--enable-pyramid-order", dest="enable_pyramid_order", action="store_true", default=None)
    parser.add_argument("--disable-pyramid-order", dest="enable_pyramid_order", action="store_false")
    parser.add_argument("--enable-crag-refinement", dest="enable_crag_refinement", action="store_true", default=None)
    parser.add_argument("--disable-crag-refinement", dest="enable_crag_refinement", action="store_false")
    parser.add_argument("--crag-refine-top-sentences", type=int, default=None)
    parser.add_argument("--crag-refine-max-sentences", type=int, default=None)
    parser.add_argument("--max-retrieval-rounds", type=int, default=None)
    parser.add_argument("--self-consistency-samples", type=int, default=None)
    parser.add_argument("--self-consistency-temperature", type=float, default=None)
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
        "--stdio-jsonl",
        action="store_true",
        help="Run as a persistent JSONL stdio service for the Web UI.",
    )
    return parser.parse_args()


def append_dialogue_turn(history: list[str], role: str, content: str) -> None:
    text = content.strip()
    if not text:
        return
    history.append(f"{role}: {text}")


def render_dialogue_context(history: list[str]) -> str:
    return "\n".join(history).strip()


def render_history_payload(history: object) -> str:
    if not isinstance(history, list):
        return ""
    lines: list[str] = []
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role and content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


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
    reranker_max_length = int(retrieval_cfg.get("reranker_max_length", 1024))
    max_retrieval_rounds = args.max_retrieval_rounds
    if max_retrieval_rounds is None:
        max_retrieval_rounds = inference_cfg.get("max_retrieval_rounds")
    if max_retrieval_rounds is None:
        max_retrieval_rounds = 3
    max_retrieval_rounds = int(max_retrieval_rounds)
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
    enable_crag_refinement = bool(
        resolve_config_value(args.enable_crag_refinement, inference_cfg, "enable_crag_refinement", False)
    )
    crag_refine_top_sentences = int(
        resolve_config_value(args.crag_refine_top_sentences, inference_cfg, "crag_refine_top_sentences", 4)
    )
    crag_refine_max_sentences = int(
        resolve_config_value(args.crag_refine_max_sentences, inference_cfg, "crag_refine_max_sentences", 24)
    )
    self_consistency_samples = int(
        resolve_config_value(args.self_consistency_samples, inference_cfg, "self_consistency_samples", 1)
    )
    self_consistency_temperature = float(
        resolve_config_value(args.self_consistency_temperature, inference_cfg, "self_consistency_temperature", 0.7)
    )
    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True))
    if args.no_reranker:
        enable_reranker = False
    ctx_size = int(resolve_config_value(args.ctx_size, generator_cfg, "ctx_size", 8192))
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
    minirag_weight = float(resolve_config_value(args.minirag_weight, retrieval_cfg, "minirag_weight", 0.35))
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

    from asa_arknight_story_agent.inference import CPUInferencePipeline  # noqa: E402
    from asa_arknight_story_agent.inference.cpu_pipeline import LlamaCppRunner, VllmRunner  # noqa: E402
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=reranker_max_length,
        minirag_index_path=minirag_index_path,
        device=device,
    )
    if backend == "vllm":
        base_model = resolve_path_value(args.base_model, vllm_cfg, "base_model_path", DEFAULT_BASE_MODEL_PATH)
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
        llama_threads = int(resolve_config_value(args.threads, llama_cpp_cfg, "threads", 16))
        generator = LlamaCppRunner(
            llama_cli_path=llama_cli or DEFAULT_LLAMA_CLI_PATH,
            gguf_model_path=gguf_model or DEFAULT_GGUF_MODEL_PATH,
            lora_path=lora_path,
            threads=llama_threads,
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
            fusion_top_k=fusion_top_k,
            rerank_top_k=rerank_top_k,
            minirag_weight=minirag_weight,
            minirag_mode_weights={str(key): float(value) for key, value in minirag_mode_weights.items()},
            enable_neighbor_expansion=enable_neighbor_expansion,
            neighbor_max_seed_docs=neighbor_max_seed_docs,
            neighbor_story_window=neighbor_story_window,
            neighbor_activity_story_sort_window=neighbor_activity_story_sort_window,
            rerank_batch_size=rerank_batch_size,
        ),
        max_retrieval_rounds=max_retrieval_rounds,
        prompt_evidence_top_k=prompt_evidence_top_k,
        enable_mmr=enable_mmr,
        mmr_lambda=mmr_lambda,
        enable_pyramid_order=enable_pyramid_order,
        enable_crag_refinement=enable_crag_refinement,
        crag_refine_top_sentences=crag_refine_top_sentences,
        crag_refine_max_sentences=crag_refine_max_sentences,
        self_consistency_samples=self_consistency_samples,
        self_consistency_temperature=self_consistency_temperature,
        use_model_hypothesis=use_model_hypothesis,
        use_model_conclusion_generation=use_model_conclusion_generation,
    )

    if args.stdio_jsonl:
        print(
            json.dumps(
                {
                    "event": "ready",
                    "backend": backend,
                    "device": device,
                    "runtime": pipeline.generator.describe_runtime(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                if request.get("command") in {"exit", "quit", "stop"}:
                    print(json.dumps({"event": "stopped"}, ensure_ascii=False), flush=True)
                    break
                question = str(request.get("message") or request.get("question") or "").strip()
                if not question:
                    raise ValueError("message cannot be empty")
                dialogue_context = str(request.get("dialogue_context") or "").strip()
                if not dialogue_context:
                    dialogue_context = render_history_payload(request.get("history"))

                old_rounds = pipeline.max_retrieval_rounds
                old_max_tokens = getattr(pipeline.generator, "max_tokens", None)
                if request.get("max_retrieval_rounds") not in (None, ""):
                    pipeline.max_retrieval_rounds = max(1, int(request["max_retrieval_rounds"]))
                if request.get("max_tokens") not in (None, "") and hasattr(pipeline.generator, "max_tokens"):
                    pipeline.generator.max_tokens = max(1, int(request["max_tokens"]))

                started = time.perf_counter()
                stages: list[str] = []
                try:
                    result = pipeline.run(question, dialogue_context, progress_callback=stages.append)
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "event": "result",
                                "ok": True,
                                "elapsed": elapsed,
                                "stages": stages,
                                "result": asdict(result),
                                "answer": result.answer,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                finally:
                    pipeline.max_retrieval_rounds = old_rounds
                    if old_max_tokens is not None and hasattr(pipeline.generator, "max_tokens"):
                        pipeline.generator.max_tokens = old_max_tokens
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "error",
                            "ok": False,
                            "error": type(exc).__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return

    dialogue_history: list[str] = []
    if args.dialogue_context.strip():
        dialogue_history.extend(
            line.strip()
            for line in args.dialogue_context.splitlines()
            if line.strip()
        )

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
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2), flush=True)

        append_dialogue_turn(dialogue_history, "user", question)
        append_dialogue_turn(dialogue_history, "assistant", result.answer)


if __name__ == "__main__":
    main()
