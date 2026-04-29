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
    / "teacher_v2_plus_prompt_supplement_v4_qwen35_4b"
    / "checkpoint-381"
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
    return parser.parse_args()


def append_dialogue_turn(history: list[str], role: str, content: str) -> None:
    text = content.strip()
    if not text:
        return
    history.append(f"{role}: {text}")


def render_dialogue_context(history: list[str]) -> str:
    return "\n".join(history).strip()


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
    max_retrieval_rounds = inference_cfg.get("max_retrieval_rounds")
    if max_retrieval_rounds is None:
        legacy_follow_up_rounds = inference_cfg.get("max_follow_up_rounds")
        if legacy_follow_up_rounds is None:
            max_retrieval_rounds = 3
        else:
            max_retrieval_rounds = int(legacy_follow_up_rounds) + 1
    max_retrieval_rounds = int(max_retrieval_rounds)
    prompt_evidence_top_k = int(inference_cfg.get("prompt_evidence_top_k", 8))
    use_model_hypothesis = bool(inference_cfg.get("use_model_hypothesis", True))
    use_model_conclusion_generation = bool(
        inference_cfg.get(
            "use_model_conclusion_generation",
            inference_cfg.get("use_model_retrieval_planner", True),
        )
    )
    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True))
    if args.no_reranker:
        enable_reranker = False
    ctx_size = int(resolve_config_value(args.ctx_size, generator_cfg, "ctx_size", 8192))
    max_tokens = int(resolve_config_value(args.max_tokens, generator_cfg, "max_tokens", 512))
    temperature = float(resolve_config_value(args.temperature, generator_cfg, "temperature", 0.2))
    top_p = float(resolve_config_value(args.top_p, generator_cfg, "top_p", 0.9))
    repeat_penalty = float(resolve_config_value(args.repeat_penalty, generator_cfg, "repeat_penalty", 1.05))

    reranker_model = args.reranker_model
    if reranker_model is None and enable_reranker:
        reranker_model = RERANKER_MODEL_DIR
    if not enable_reranker:
        reranker_model = None

    from goldenglow.inference import CPUInferencePipeline  # noqa: E402
    from goldenglow.inference.cpu_pipeline import LlamaCppRunner, VllmRunner  # noqa: E402
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
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
            fusion_top_k=fusion_top_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
        ),
        max_retrieval_rounds=max_retrieval_rounds,
        prompt_evidence_top_k=prompt_evidence_top_k,
        use_model_hypothesis=use_model_hypothesis,
        use_model_conclusion_generation=use_model_conclusion_generation,
    )
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
