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

from asa_arknight_story_agent.config import (  # noqa: E402
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
- 如果证据能支持部分回答，应优先给出“可确认部分”，不要因为缺少完整背景直接 abstain；但不能把未被证据支持的内容写成确定事实。
- 多轮上下文只能用于消歧，不得把上一轮 assistant 的错误结论当作事实。"""

API_MODE_QA_SYSTEM_APPENDIX = """你正在替代本地微调 4B 模型，为一个《明日方舟》RAG 检索管线生成最终自然语言答案。
硬性要求：
- 只依据用户消息中的检索证据回答，不要使用训练记忆补证据。
- 不要输出 JSON、markdown 表格、schema 字段或链路分析。
- 如果证据不足，明确说明哪些部分不足以确认。
- 如果证据能支持部分回答，应优先给出“可确认部分”，不要因为缺少完整背景直接 abstain；但不能把未被证据支持的内容写成确定事实。
- 不要输出或展开 reasoning_content / chain-of-thought；最终内容必须写在 assistant message 的 content 字段中。"""


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


def build_plain_initial_prompt(question: str, dialogue_context: str = "", *, prompt_hint: str = "") -> str:
    context_block = f"\n\n对话上下文：\n{dialogue_context.strip()}" if dialogue_context.strip() else ""
    hint = prompt_hint.strip() or "直接回答用户问题。可以使用你已有的知识；不要编造不确定细节。"
    return f"{hint}{context_block}\n\n用户问题：{question.strip()}"


def evidence_text_from_hits(
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    max_chars_per_doc: int = 700,
    max_total_chars: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[str] = []
    simplified: list[dict[str, Any]] = []
    total_chars = 0
    for index, item in enumerate(hits[:top_k], start=1):
        doc = item.get("document") or {}
        text = str(item.get("evidence_chain_text") or doc.get("clean_text") or doc.get("text") or "").strip()
        if len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc].rstrip() + "..."
        doc_id = str(doc.get("id") or "")
        title = " / ".join(str(v) for v in [doc.get("activity_name"), doc.get("story_name"), doc.get("stage_code")] if v)
        blocks.append(f"[证据{index}] {title}\nID: {doc_id}\n{text}")
        simplified.append(
            {
                "id": doc_id,
                "activity_name": doc.get("activity_name"),
                "story_name": doc.get("story_name"),
                "stage_code": doc.get("stage_code"),
                "avg_tag": doc.get("avg_tag"),
                "source_path": doc.get("source_path"),
                "fusion_score": item.get("fusion_score"),
                "rerank_score": item.get("rerank_score"),
                "evidence_chain_score": item.get("evidence_chain_score"),
                "dense_score": item.get("dense_score"),
                "sparse_score": item.get("sparse_score"),
                "minirag_score": item.get("minirag_score"),
                "evidence_chain_text": item.get("evidence_chain_text"),
                "clean_text": doc.get("clean_text"),
            }
        )
        total_chars += len(blocks[-1])
        if max_total_chars is not None and total_chars >= max_total_chars:
            break
    return "\n\n".join(blocks), simplified


def build_revision_prompt(question: str, initial_answer: str, evidence_text: str, dialogue_context: str = "") -> str:
    context_block = f"\n\n对话上下文：\n{dialogue_context.strip()}" if dialogue_context.strip() else ""
    return f"""你是《明日方舟》剧情问答校正器。
下面先给出模型对原问题的初答，再给出本地 RAG 检索到的证据。
任务：用证据校正初答中的幻觉、错因、错名和过度推断，输出最终自然语言答案。

规则：
1. 证据能支持的内容可以保留并说明。
2. 初答和证据冲突时，以证据为准。
3. 证据不足时明确说“不足以确认”，不要强行补设定。
4. 对“为什么启动/开启/启用/动用某物”类问题，优先找同时包含动作、目标物、目的/危机/代价的直接证据；不要把背景动机当成直接原因。
5. 如果证据里有“为了解决某场危机”“以某人性命为代价”等直接表述，最终答案必须先写这些直接表述，再补充背景。
6. 不要输出 JSON，不要输出推理过程，不要写 markdown 表格。
7. 最好用 2-4 段中文回答；必要时指出“初答中哪些说法无法由证据支持”。

原问题：{question.strip()}{context_block}

初答：
{initial_answer.strip()}

检索证据：
{evidence_text.strip() if evidence_text.strip() else "（没有检索到有效证据）"}

最终校正答案："""


def chatml_prompt_to_messages(prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_appendix = API_MODE_SYSTEM_APPENDIX if prompt_wants_json(prompt) else API_MODE_QA_SYSTEM_APPENDIX
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
            cleaned = cleaned + "\n\n" + system_appendix
        messages.append({"role": role, "content": cleaned})
    if messages:
        return messages
    return [
        {"role": "system", "content": system_appendix},
        {"role": "user", "content": prompt},
    ]


def prompt_wants_json(prompt: str) -> bool:
    if "请直接回答用户问题" in prompt or "最终校正答案" in prompt:
        return False
    return any(
        marker in prompt
        for marker in (
            "output_schema:",
            "输出必须是单个 JSON",
            "只输出 JSON",
            "必须输出 JSON",
            "字段严格包含",
            "schema: hypothesis",
            "schema: conclusion",
            "task: user_question_hypothesis_generation",
            "task: follow_up_hypothesis_generation",
            "task: conclusion_generation",
        )
    )


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
            message = response["choices"][0]["message"]
            content = message["content"]
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
            reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
            if reasoning:
                retry_payload = dict(payload)
                retry_messages = list(payload.get("messages") or [])
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "上一条响应没有 assistant content。请只输出符合 schema 的单个 JSON 对象，不要解释。"
                            if wants_json
                            else "上一条响应没有 assistant content。请只输出最终答案正文，不要解释推理过程。"
                        ),
                    }
                )
                retry_payload["messages"] = retry_messages
                retry_response = self._post_chat_completion(retry_payload)
                self._write_request_log(payload=retry_payload, response=retry_response)
                try:
                    retry_content = retry_response["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise RuntimeError(
                        f"Unexpected chat completion retry response: {json.dumps(retry_response, ensure_ascii=False)[:1000]}"
                    ) from exc
                if isinstance(retry_content, str) and retry_content.strip():
                    return retry_content.strip()
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
    parser.add_argument("--enable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_true", default=None)
    parser.add_argument("--disable-neighbor-expansion", dest="enable_neighbor_expansion", action="store_false")
    parser.add_argument("--neighbor-max-seed-docs", type=int, default=None)
    parser.add_argument("--neighbor-story-window", type=int, default=None)
    parser.add_argument("--neighbor-activity-story-sort-window", type=int, default=None)
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
    parser.add_argument("--prompt-evidence-top-k", type=int, default=None)
    parser.add_argument("--max-retrieval-rounds", type=int, default=None)
    parser.add_argument("--conclusion-prompt-mode", choices=("full", "minimal"), default=None)
    parser.add_argument("--self-consistency-samples", type=int, default=None)
    parser.add_argument("--self-consistency-temperature", type=float, default=None)
    parser.add_argument("--answer-grounding-mode", choices=("off", "weak", "strict"), default=None)
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
    parser.add_argument(
        "--pipeline-mode",
        choices=("standard", "answer_then_retrieve_refine"),
        default=None,
        help="API mode pipeline. answer_then_retrieve_refine first asks the LLM directly, retrieves with that answer, then asks it to correct hallucinations.",
    )
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
    enable_neighbor_expansion = bool(
        resolve_config_value(args.enable_neighbor_expansion, retrieval_cfg, "enable_neighbor_expansion", False)
    )
    neighbor_max_seed_docs = int(
        resolve_config_value(args.neighbor_max_seed_docs, retrieval_cfg, "neighbor_max_seed_docs", 24)
    )
    neighbor_story_window = int(
        resolve_config_value(args.neighbor_story_window, retrieval_cfg, "neighbor_story_window", 2)
    )
    neighbor_activity_story_sort_window = int(
        resolve_config_value(
            args.neighbor_activity_story_sort_window,
            retrieval_cfg,
            "neighbor_activity_story_sort_window",
            1,
        )
    )
    reranker_max_length = int(retrieval_cfg.get("reranker_max_length", 1024))
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
        resolve_config_value(args.minirag_auto_second_retrieval, retrieval_cfg, "minirag_auto_second_retrieval", True)
    )
    minirag_scope_seed_top_k = int(
        resolve_config_value(args.minirag_scope_seed_top_k, retrieval_cfg, "minirag_scope_seed_top_k", 40)
    )
    minirag_expansion_query_top_k = int(
        resolve_config_value(args.minirag_expansion_query_top_k, retrieval_cfg, "minirag_expansion_query_top_k", 8)
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
        resolve_config_value(args.storyline_sparse_scope_min_ratio, retrieval_cfg, "storyline_sparse_scope_min_ratio", 1.5)
    )
    reranker_candidate_top_k = int(
        resolve_config_value(args.reranker_candidate_top_k, retrieval_cfg, "reranker_candidate_top_k", 120)
    )
    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}
    minirag_index_path = None
    if enable_minirag:
        minirag_index_path = resolve_path_value(args.minirag_index, retrieval_cfg, "minirag_index_path", MINIRAG_GRAPH_PATH)
        if minirag_index_path is None or not minirag_index_path.exists():
            raise SystemExit(
                "MiniRAG index is enabled but missing. Build it with "
                "`python scripts/build_minirag_index.py` or disable retrieval.enable_minirag."
            )
    max_retrieval_rounds = int(
        resolve_config_value(args.max_retrieval_rounds, inference_cfg, "max_retrieval_rounds", 2)
    )
    max_retrieval_rounds = min(2, max(1, max_retrieval_rounds))
    prompt_evidence_top_k = int(
        resolve_config_value(args.prompt_evidence_top_k, inference_cfg, "prompt_evidence_top_k", 12)
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
    pipeline_mode = str(resolve_config_value(args.pipeline_mode, inference_cfg, "pipeline_mode", "standard"))
    initial_prompt_hint = str(inference_cfg.get("initial_prompt_hint", "") or "")
    initial_answer_max_tokens = int(inference_cfg.get("initial_answer_max_tokens", 1024))
    refine_answer_max_tokens = int(inference_cfg.get("refine_answer_max_tokens", 1536))

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

    from asa_arknight_story_agent.inference import CPUInferencePipeline  # noqa: E402
    from asa_arknight_story_agent.inference.cpu_pipeline import InferenceResult  # noqa: E402
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever  # noqa: E402

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
            reranker_candidate_top_k=reranker_candidate_top_k,
            enable_neighbor_expansion=enable_neighbor_expansion,
            neighbor_max_seed_docs=neighbor_max_seed_docs,
            neighbor_story_window=neighbor_story_window,
            neighbor_activity_story_sort_window=neighbor_activity_story_sort_window,
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
        conclusion_prompt_mode=conclusion_prompt_mode,
        answer_grounding_mode=answer_grounding_mode,
        use_model_hypothesis=bool(inference_cfg.get("use_model_hypothesis", True)),
        use_model_conclusion_generation=bool(inference_cfg.get("use_model_conclusion_generation", True)),
        web_context_config=inference_cfg.get("web_context") if isinstance(inference_cfg.get("web_context"), dict) else None,
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
        print(f"[running] api-mode pipeline start mode={pipeline_mode}", file=sys.stderr, flush=True)
        try:
            dialogue_context = render_dialogue_context(dialogue_history)
            if pipeline_mode == "answer_then_retrieve_refine":
                print("[stage] initial_direct_answer", file=sys.stderr, flush=True)
                initial_prompt = build_plain_initial_prompt(
                    question,
                    dialogue_context,
                    prompt_hint=initial_prompt_hint,
                )
                initial_answer = generator.generate(
                    initial_prompt,
                    max_tokens=initial_answer_max_tokens,
                    temperature=max(float(resolve_config_value(args.temperature, generator_cfg, "temperature", 0.2)), 0.2),
                    top_p=float(resolve_config_value(args.top_p, generator_cfg, "top_p", 0.9)),
                )
                retrieval_query = f"{question}\n\n初步回答：{initial_answer}"
                print("[stage] retrieval_from_initial_answer", file=sys.stderr, flush=True)
                if enable_reranker:
                    hits = retriever.search(retrieval_query, config=pipeline.query_config)
                else:
                    hits = retriever.search_pre_rerank(retrieval_query, config=pipeline.query_config)
                evidence_text, simplified_evidence = evidence_text_from_hits(
                    hits,
                    top_k=prompt_evidence_top_k,
                    max_chars_per_doc=prompt_evidence_max_chars_per_doc,
                    max_total_chars=prompt_conclusion_evidence_max_total_chars,
                )
                print("[stage] evidence_grounded_revision", file=sys.stderr, flush=True)
                final_answer = generator.generate(
                    build_revision_prompt(question, initial_answer, evidence_text, dialogue_context),
                    max_tokens=refine_answer_max_tokens,
                    temperature=0.1,
                    top_p=float(resolve_config_value(args.top_p, generator_cfg, "top_p", 0.9)),
                )
                result = InferenceResult(
                    question=question,
                    intent="api_answer_then_retrieve_refine",
                    hypothesis={
                        "question": question,
                        "initial_answer": initial_answer,
                        "retrieval_query": retrieval_query,
                    },
                    model_runtime={
                        **generator.describe_runtime(),
                        "api_pipeline_mode": pipeline_mode,
                        "initial_answer_max_tokens": initial_answer_max_tokens,
                        "refine_answer_max_tokens": refine_answer_max_tokens,
                    },
                    retrieval_query=retrieval_query,
                    retrieval_trace=[
                        {
                            "stage": "initial_direct_answer",
                            "initial_answer": initial_answer,
                        },
                        {
                            "stage": "retrieval_from_initial_answer",
                            "evidence_count": len(simplified_evidence),
                            "top_doc_ids": [item.get("id") for item in simplified_evidence[:5]],
                        },
                        {
                            "stage": "evidence_grounded_revision",
                        },
                    ],
                    evidence=simplified_evidence,
                    answer=final_answer,
                )
            elif pipeline_mode == "standard":
                result = pipeline.run(
                    question,
                    dialogue_context,
                    progress_callback=lambda stage: print(f"[stage] {stage}", file=sys.stderr, flush=True),
                )
            else:
                raise ValueError(f"Unsupported pipeline_mode: {pipeline_mode}")
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
