#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_OVERRIDE_DIR = PROJECT_ROOT / ".vendor" / "train_override"
TRAIN_PYTHON_OVERLAY_DIR = PROJECT_ROOT / ".python_packages" / "train"


def should_use_train_overrides() -> bool:
    override_flag = os.environ.get("GOLDENGLOW_USE_TRAIN_OVERRIDE")
    if override_flag is not None:
        return override_flag.lower() in {"1", "true", "yes", "on"}
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip().lower()
    executable = Path(sys.executable).as_posix().lower()
    return conda_env == "train" or "/envs/train/" in executable or executable.endswith("/envs/train/bin/python")


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

CHATML_MESSAGE_RE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "api-mode" / "runtime_api.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "api_mode_runs"

API_MODE_SYSTEM_APPENDIX = """你正在替代本地微调 4B 模型，为一个《明日方舟》RAG 检索管线生成机器可解析输出。
请先理解自己当前扮演的组件：
- hypothesis_builder / follow_up_hypothesis_builder：只生成用于下一步检索的 JSON，不回答剧情问题。
- conclusion_generator：基于当前证据判断 answer_directly / retrieve_more / clarify_user / abstain，并严格输出 JSON。
- 普通问答：只依据给定证据直接回答，不要使用训练记忆补证据。
硬性要求：
- 严格遵守用户消息里给出的 schema、字段名和 action 枚举。
- 需要 JSON 时只输出一个 JSON 对象，不要 markdown，不要解释，不要输出思维过程。
- 不要输出或展开 reasoning_content / chain-of-thought；最终内容必须写在 assistant message 的 content 字段中。
- 不要把缺证据时的推测写成事实；证据不足时按 schema 选择 retrieve_more 或 abstain。
- 多轮上下文只能用于消歧，不得把上一轮 assistant 的错误结论当作事实。"""


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_config_value(cli_value: Any, config_section: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def resolve_path_value(cli_value: Any, config_section: dict[str, Any], key: str, default: Path | str | None) -> Path | None:
    value = cli_value if cli_value is not None else config_section.get(key, default)
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_index(build_index_if_missing: bool) -> None:
    if DOCUMENTS_PATH.exists() and FAISS_INDEX_PATH.exists() and BM25_TOKENS_PATH.exists():
        return
    if not build_index_if_missing:
        raise FileNotFoundError(
            "Retrieval index is missing. Run `python scripts/build_retrieval_index.py --device cpu` "
            "or add `--build-index-if-missing`."
        )
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_retrieval_index.py"), "--device", "cpu"],
        check=True,
        cwd=PROJECT_ROOT,
    )


def render_dialogue_context(history: list[str]) -> str:
    return "\n".join(history).strip()


def append_dialogue_turn(history: list[str], role: str, content: str) -> None:
    text = content.strip()
    if text:
        history.append(f"{role}: {text}")


def chatml_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for role, content in CHATML_MESSAGE_RE.findall(prompt):
        cleaned = content.strip()
        if not cleaned:
            continue
        # Local-model prompts end with an assistant prefill like "{" or
        # "<think>...</think>". Chat APIs generally do not support prefill, and
        # the user prompt already states the output format.
        if role == "assistant":
            continue
        if role == "system":
            cleaned = cleaned + "\n\n" + API_MODE_SYSTEM_APPENDIX
        messages.append({"role": role, "content": cleaned})
    if messages:
        return messages
    return [
        {"role": "system", "content": API_MODE_SYSTEM_APPENDIX},
        {"role": "user", "content": prompt},
    ]


def prompt_wants_json(prompt: str) -> bool:
    return "JSON" in prompt or "json" in prompt or "输出必须是单个" in prompt or "schema" in prompt


def validate_api_key(api_key: str, api_key_env: str) -> None:
    if "你的" in api_key or "API key" in api_key or "api key" in api_key:
        raise SystemExit(f"{api_key_env} still contains a placeholder. Replace it with the real API key.")
    try:
        api_key.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise SystemExit(f"{api_key_env} contains non-ASCII characters. API keys must be copied exactly as plain text.") from exc
    if any(char.isspace() for char in api_key):
        raise SystemExit(f"{api_key_env} contains whitespace. Check whether the API key was copied with extra spaces.")


def normalize_chat_completions_url(api_base_url: str) -> str:
    url = api_base_url.strip().rstrip("/")
    if not url:
        raise ValueError("api_base_url cannot be empty.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid api_base_url: {api_base_url!r}")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def normalize_responses_url(api_base_url: str) -> str:
    url = api_base_url.strip().rstrip("/")
    if not url:
        raise ValueError("api_base_url cannot be empty.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid api_base_url: {api_base_url!r}")
    if parsed.path.rstrip("/").endswith("/responses"):
        return url
    return f"{url}/responses"


def extract_responses_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = response.get("output")
    if not isinstance(output, list):
        raise RuntimeError(f"Unexpected responses payload: {json.dumps(response, ensure_ascii=False)[:1000]}")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text") or part.get("output_text") or part.get("content")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    if texts:
        return "\n".join(texts).strip()
    raise RuntimeError(f"API returned no text: {json.dumps(response, ensure_ascii=False)[:1000]}")


class OpenAICompatibleAPIRunner:
    backend_name = "openai-compatible-api"

    def __init__(
        self,
        *,
        api_base_url: str,
        api_key: str | None,
        api_key_env: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 768,
        temperature: float = 0.1,
        top_p: float = 0.9,
        response_format_json: bool = True,
        extra_body: dict[str, Any] | None = None,
        request_log_dir: Path | None = None,
    ) -> None:
        self.api_base_url = normalize_chat_completions_url(api_base_url)
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = 1.0
        self.response_format_json = response_format_json
        self.extra_body = extra_body or {}
        self.request_log_dir = request_log_dir
        self._request_index = 0

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "generator_backend": self.backend_name,
            "api_base_url": self.api_base_url,
            "api_key_env": self.api_key_env,
            "model": self.model,
            "runtime_mode": "remote_base_model_api_no_lora",
            "local_generation_model": None,
            "lora_path": None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": self.extra_body,
        }

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.api_base_url, data=data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc
        parsed = json.loads(body)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"API returned non-object JSON: {body[:500]}")
        return parsed

    def _write_request_log(
        self,
        *,
        payload: dict[str, Any],
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if self.request_log_dir is None:
            return
        self.request_log_dir.mkdir(parents=True, exist_ok=True)
        self._request_index += 1
        safe_payload = dict(payload)
        safe_payload["messages"] = [
            {
                "role": message.get("role"),
                "content": message.get("content"),
            }
            for message in payload.get("messages", [])
            if isinstance(message, dict)
        ]
        record = {
            "request_index": self._request_index,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": safe_payload,
            "response": response,
            "error": error,
        }
        path = self.request_log_dir / f"api_request_{self._request_index:03d}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        del repeat_penalty
        messages = chatml_prompt_to_messages(prompt)
        wants_json = prompt_wants_json(prompt)
        requested_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if wants_json:
            # The shared pipeline uses small local-4B budgets for JSON tasks.
            # Remote models may emit longer valid JSON, so keep a safer floor.
            requested_max_tokens = max(requested_max_tokens, self.max_tokens, 4096)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "max_tokens": requested_max_tokens,
        }
        payload.update(self.extra_body)
        if self.response_format_json and wants_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._post_chat_completion(payload)
            self._write_request_log(payload=payload, response=response)
        except RuntimeError as exc:
            self._write_request_log(payload=payload, error=str(exc))
            # Some OpenAI-compatible providers do not implement response_format.
            if "response_format" not in payload:
                raise
            payload.pop("response_format", None)
            response = self._post_chat_completion(payload)
            self._write_request_log(payload=payload, response=response)

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            try:
                message = response["choices"][0]["message"]
                finish_reason = response["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError):
                message = {}
                finish_reason = None
            reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
            if reasoning:
                raise RuntimeError(
                    "API returned reasoning_content without assistant content. "
                    f"finish_reason={finish_reason}. Increase --max-tokens or use a non-reasoning/no-thinking model mode."
                ) from exc
            raise RuntimeError(f"Unexpected chat completion response: {json.dumps(response, ensure_ascii=False)[:1000]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"API returned empty content: {json.dumps(response, ensure_ascii=False)[:1000]}")
        return content.strip()


class ResponsesAPIRunner(OpenAICompatibleAPIRunner):
    backend_name = "responses-api"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_base_url = normalize_responses_url(kwargs["api_base_url"])

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        del repeat_penalty
        messages = chatml_prompt_to_messages(prompt)
        response_input = [
            {
                "role": message["role"],
                "content": [{"type": "input_text", "text": message["content"]}],
            }
            for message in messages
        ]
        wants_json = prompt_wants_json(prompt)
        requested_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if wants_json:
            requested_max_tokens = max(requested_max_tokens, self.max_tokens, 4096)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": response_input,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": top_p if top_p is not None else self.top_p,
            "max_output_tokens": requested_max_tokens,
        }
        payload.update(self.extra_body)
        try:
            response = self._post_chat_completion(payload)
            self._write_request_log(payload=payload, response=response)
        except RuntimeError as exc:
            self._write_request_log(payload=payload, error=str(exc))
            raise
        return extract_responses_text(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Goldenglow inference with local retrieval and a remote API model.")
    parser.add_argument("question", type=str, nargs="?", default=None, help="Optional initial user question in Chinese.")
    parser.add_argument("--dialogue-context", type=str, default="", help="Optional multi-turn context.")
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG_PATH)
    parser.add_argument("--device", type=str, default=None, help="Retrieval device. Overrides runtime config.")
    parser.add_argument("--embedding-model", type=Path, default=EMBEDDING_MODEL_DIR)
    parser.add_argument("--reranker-model", type=Path, default=None)
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--dense-top-k", type=int, default=None)
    parser.add_argument("--sparse-top-k", type=int, default=None)
    parser.add_argument("--fusion-top-k", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=None)
    parser.add_argument("--build-index-if-missing", action="store_true")
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--no-json-response-format", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no-save-run", action="store_true")
    parser.add_argument("--answer-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_index(args.build_index_if_missing)
    runtime_config = load_runtime_config(args.runtime_config)
    retrieval_cfg = runtime_config.get("retrieval", {}) if isinstance(runtime_config.get("retrieval"), dict) else {}
    inference_cfg = runtime_config.get("inference", {}) if isinstance(runtime_config.get("inference"), dict) else {}
    generator_cfg = runtime_config.get("generator", {}) if isinstance(runtime_config.get("generator"), dict) else {}

    device = str(resolve_config_value(args.device, retrieval_cfg, "device", "cpu"))
    dense_top_k = int(resolve_config_value(args.dense_top_k, retrieval_cfg, "dense_top_k", 60))
    sparse_top_k = int(resolve_config_value(args.sparse_top_k, retrieval_cfg, "sparse_top_k", 60))
    fusion_top_k = int(resolve_config_value(args.fusion_top_k, retrieval_cfg, "fusion_top_k", 40))
    rerank_top_k = int(resolve_config_value(args.rerank_top_k, retrieval_cfg, "rerank_top_k", 15))
    rerank_batch_size = int(resolve_config_value(args.rerank_batch_size, retrieval_cfg, "rerank_batch_size", 8))
    enable_neighbor_expansion = bool(retrieval_cfg.get("enable_neighbor_expansion", False))
    neighbor_max_seed_docs = int(retrieval_cfg.get("neighbor_max_seed_docs", 24))
    neighbor_story_window = int(retrieval_cfg.get("neighbor_story_window", 2))
    neighbor_activity_story_sort_window = int(
        retrieval_cfg.get("neighbor_activity_story_sort_window", 1)
    )
    reranker_max_length = int(retrieval_cfg.get("reranker_max_length", 1024))
    enable_minirag = bool(retrieval_cfg.get("enable_minirag", False))
    minirag_index_path = None
    if enable_minirag:
        minirag_index_path = resolve_path_value(None, retrieval_cfg, "minirag_index_path", MINIRAG_GRAPH_PATH)
        if minirag_index_path is None or not minirag_index_path.exists():
            raise SystemExit(
                "MiniRAG index is enabled but missing. Build it with "
                "`python scripts/build_minirag_index.py` or disable retrieval.enable_minirag."
            )
    max_retrieval_rounds = int(inference_cfg.get("max_retrieval_rounds", 3))
    prompt_evidence_top_k = int(inference_cfg.get("prompt_evidence_top_k", 12))
    enable_mmr = bool(inference_cfg.get("enable_mmr", False))
    mmr_lambda = float(inference_cfg.get("mmr_lambda", 0.72))
    enable_pyramid_order = bool(inference_cfg.get("enable_pyramid_order", False))
    enable_crag_refinement = bool(inference_cfg.get("enable_crag_refinement", False))
    crag_refine_top_sentences = int(inference_cfg.get("crag_refine_top_sentences", 4))
    crag_refine_max_sentences = int(inference_cfg.get("crag_refine_max_sentences", 24))
    self_consistency_samples = int(inference_cfg.get("self_consistency_samples", 1))
    self_consistency_temperature = float(inference_cfg.get("self_consistency_temperature", 0.7))

    enable_reranker = bool(retrieval_cfg.get("enable_reranker", True))
    if args.no_reranker:
        enable_reranker = False
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
                f"{reranker_model or '<empty>'}. Pass a directory containing config.json."
            )

    api_key_env = str(resolve_config_value(args.api_key_env, generator_cfg, "api_key_env", "OPENAI_API_KEY"))
    api_key = args.api_key if args.api_key is not None else os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key. Set environment variable {api_key_env} or pass --api-key.")
    validate_api_key(api_key, api_key_env)
    run_log_dir = None
    if not args.no_save_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_dir = (args.log_dir if args.log_dir.is_absolute() else PROJECT_ROOT / args.log_dir) / timestamp
        run_log_dir.mkdir(parents=True, exist_ok=True)
    generator_backend = str(generator_cfg.get("backend", "openai_compatible_api"))
    generator_cls: type[OpenAICompatibleAPIRunner]
    if generator_backend in {"openai_compatible_api", "chat_completions"}:
        generator_cls = OpenAICompatibleAPIRunner
    elif generator_backend in {"responses_api", "responses"}:
        generator_cls = ResponsesAPIRunner
    else:
        raise SystemExit(f"Unsupported generator backend: {generator_backend}")
    generator = generator_cls(
        api_base_url=str(resolve_config_value(args.api_base_url, generator_cfg, "api_base_url", "https://api.openai.com/v1/chat/completions")),
        api_key=api_key,
        api_key_env=api_key_env,
        model=str(resolve_config_value(args.model, generator_cfg, "model", "gpt-4.1")),
        timeout=float(resolve_config_value(args.timeout, generator_cfg, "timeout", 120)),
        max_tokens=int(resolve_config_value(args.max_tokens, generator_cfg, "max_tokens", 768)),
        temperature=float(resolve_config_value(args.temperature, generator_cfg, "temperature", 0.1)),
        top_p=float(resolve_config_value(args.top_p, generator_cfg, "top_p", 0.9)),
        response_format_json=bool(generator_cfg.get("response_format_json", True)) and not args.no_json_response_format,
        extra_body=generator_cfg.get("extra_body") if isinstance(generator_cfg.get("extra_body"), dict) else None,
        request_log_dir=run_log_dir,
    )

    from goldenglow.inference import CPUInferencePipeline  # noqa: E402
    from goldenglow.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402

    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=args.embedding_model,
        reranker_model_path=reranker_model,
        reranker_max_length=reranker_max_length,
        minirag_index_path=minirag_index_path,
        device=device,
    )
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=QueryConfig(
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            fusion_top_k=fusion_top_k,
            rerank_top_k=rerank_top_k,
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
        use_model_hypothesis=bool(inference_cfg.get("use_model_hypothesis", True)),
        use_model_conclusion_generation=bool(inference_cfg.get("use_model_conclusion_generation", True)),
    )

    dialogue_history: list[str] = []
    if args.dialogue_context.strip():
        dialogue_history.extend(line.strip() for line in args.dialogue_context.splitlines() if line.strip())

    pending_question = args.question
    print("API-mode inference ready. Type /exit to quit, /clear to reset dialogue context.", flush=True)
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

        started = time.perf_counter()
        print("[running] api-mode pipeline start...", file=sys.stderr, flush=True)
        try:
            result = pipeline.run(
                question,
                render_dialogue_context(dialogue_history),
                progress_callback=lambda stage: print(f"[stage] {stage}", file=sys.stderr, flush=True),
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            if run_log_dir is not None:
                failure = {
                    "question": question,
                    "elapsed_seconds": round(elapsed, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "api_request_count": generator._request_index,
                }
                (run_log_dir / "failure.json").write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[saved] {run_log_dir}", file=sys.stderr, flush=True)
            print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            print(f"[failed] {elapsed:.2f}s", file=sys.stderr, flush=True)
            continue
        elapsed = time.perf_counter() - started
        print(f"[done] {elapsed:.2f}s", file=sys.stderr, flush=True)
        if run_log_dir is not None:
            result_path = run_log_dir / "result.json"
            result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
            summary_path = run_log_dir / "summary.json"
            summary = {
                "question": question,
                "answer": result.answer,
                "elapsed_seconds": round(elapsed, 3),
                "result_path": str(result_path),
                "api_request_count": generator._request_index,
                "retrieval_query": result.retrieval_query,
            }
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[saved] {run_log_dir}", file=sys.stderr, flush=True)

        if args.answer_only:
            print(result.answer, flush=True)
        else:
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2), flush=True)

        append_dialogue_turn(dialogue_history, "user", question)
        append_dialogue_turn(dialogue_history, "assistant", result.answer)


if __name__ == "__main__":
    main()
