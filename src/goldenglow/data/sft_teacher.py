from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenglow.config import DOCUMENTS_PATH, EXCEL_ROOT, STORY_ROOT
from goldenglow.data.story_parser import build_story_documents


SUPPORTED_TASK_TYPES = {
    "canon_qa",
    "worldbuilding_qa",
    "persona_grounded_qa",
    "multi_turn_dialogue",
    "intent_hypothesis_rag",
    "tool_calling_rag",
    "unknown_rag_negative",
}

TASK_BUCKET_MAP = {
    "canon_qa": "knowledge",
    "worldbuilding_qa": "knowledge",
    "persona_grounded_qa": "style",
    "multi_turn_dialogue": "tool",
    "intent_hypothesis_rag": "tool",
    "tool_calling_rag": "tool",
    "unknown_rag_negative": "tool",
}

INSUFFICIENT_EVIDENCE_MARKERS = (
    "证据不足",
    "无法确认",
    "不能确认",
    "不确定",
    "检索结果不足",
    "现有检索",
    "没有足够证据",
    "无法支持",
)

FORBIDDEN_FINAL_ANSWER_MARKERS = (
    "根据检索",
    "检索到的剧情证据",
    "根据剧情证据",
    "根据证据",
    "检索结果显示",
    "从检索结果来看",
)

INTENT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_intent",
        "description": "识别用户问题的剧情问答意图、是否需要澄清以及是否需要检索。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "dialogue_context": {"type": "string"},
            },
            "required": ["question"],
        },
    },
}

HYPOTHESIS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "build_hypothesis",
        "description": "基于用户问题、意图和多轮上下文构造服务于检索的结构化假设文档。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "intent": {"type": "string"},
                "dialogue_context": {"type": "string"},
                "resolved_references": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["question", "intent"],
        },
    },
}

RAG_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_story_context",
        "description": "根据用户问题生成假设文档并检索明日方舟剧情证据，用于剧情问答。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "hypothesis": {"type": "string"},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "top_k": {"type": "integer"},
            },
            "required": ["question", "hypothesis", "keywords", "top_k"],
        },
    },
}

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class TeacherApiConfig:
    api_type: str
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: int = 120
    temperature: float = 0.8
    max_output_tokens: int = 4000
    json_mode: bool = True
    extra_headers: dict[str, str] | None = None


def load_generation_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_story_documents(documents_path: Path | None = None) -> list[dict]:
    path = documents_path or DOCUMENTS_PATH
    if path.exists():
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return build_story_documents(STORY_ROOT, EXCEL_ROOT)


def normalize_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def make_sample_fingerprint(sample: dict) -> str:
    normalized_messages = []
    for message in sample.get("messages", []):
        normalized_messages.append(
            {
                "role": message.get("role"),
                "name": message.get("name"),
                "content": normalize_message_content(message.get("content")),
                "tool_calls": message.get("tool_calls"),
            }
        )
    payload = {
        "task_type": sample.get("task_type"),
        "messages": normalized_messages,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(weight for _, weight in items)
    cursor = rng.random() * total
    seen = 0.0
    for name, weight in items:
        seen += weight
        if cursor <= seen:
            return name
    return items[-1][0]


def sample_evidence_documents(
    documents: list[dict],
    rng: random.Random,
    max_docs: int,
) -> list[dict]:
    count = max(1, rng.randint(1, max_docs))
    return rng.sample(documents, min(count, len(documents)))


def sample_worldbuilding_topic(
    topics: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    if not topics:
        raise ValueError("Worldbuilding generation requires at least one topic.")
    return dict(rng.choice(topics))


def format_evidence_pack(evidence_docs: list[dict]) -> str:
    blocks = []
    for index, doc in enumerate(evidence_docs, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"activity_name: {doc.get('activity_name') or ''}",
                    f"story_name: {doc.get('story_name') or ''}",
                    f"stage_code: {doc.get('stage_code') or ''}",
                    f"avg_tag: {doc.get('avg_tag') or ''}",
                    f"source_path: {doc.get('source_path') or ''}",
                    "clean_text:",
                    doc.get("clean_text", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def format_worldbuilding_topic(topic: dict[str, Any]) -> str:
    lines = [
        f"主题: {topic.get('topic') or ''}",
        f"目标: {topic.get('goal') or ''}",
    ]

    keywords = topic.get("keywords") or []
    if keywords:
        lines.append("关键词: " + ", ".join(str(item) for item in keywords))

    focus_points = topic.get("focus_points") or []
    if focus_points:
        lines.append("重点展开:")
        lines.extend(f"- {item}" for item in focus_points)

    avoid = topic.get("avoid") or []
    if avoid:
        lines.append("禁止触及:")
        lines.extend(f"- {item}" for item in avoid)

    return "\n".join(lines)


def build_teacher_prompts(
    *,
    task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    samples_per_request: int,
) -> tuple[str, str]:
    if task_type == "worldbuilding_qa":
        system_prompt = (
            "你是一个严格的中文教师模型数据合成器。"
            "你的任务是基于给定的《明日方舟》世界观主题生成高质量 SFT 训练样本。"
            "这些样本用于通用世界观注入，而不是复述具体剧情桥段。"
            "允许使用稳定、通行、跨剧情复用的官方世界观常识，但不要引入冷门细节、争议推断、二创设定或未经确认的数字。"
            "必须返回严格 JSON，不要输出任何额外说明。"
        )
    else:
        system_prompt = (
            "你是一个严格的中文教师模型数据合成器。"
            "你的任务是基于给定的《明日方舟》剧情证据生成高质量 SFT 训练样本。"
            "只能使用证据中明确出现或可直接归纳的信息，不得编造设定。"
            "必须返回严格 JSON，不要输出任何额外说明。"
        )

    task_specific_rules = {
        "canon_qa": "生成剧情事实问答，答案必须能从证据直接支持。",
        "worldbuilding_qa": "生成通用世界观问答，重点是概念解释、制度背景、社会结构或通用设定，不要依赖具体剧情片段。",
        "persona_grounded_qa": "生成带有澄闪语气但事实严格受证据约束的问答，语气轻柔、克制、礼貌，不要卖萌过度。",
        "multi_turn_dialogue": "生成 2 到 4 轮多轮对话，包含追问、指代消解或澄清，至少出现 2 次 user 发言和 2 次 assistant 发言。",
        "intent_hypothesis_rag": "生成完整工具链样本：assistant 必须依次调用 detect_intent、build_hypothesis、retrieve_story_context，然后直接给出最终答案。",
        "tool_calling_rag": "生成必须先完成意图识别、假设文档生成和检索再回答的样本，所有 tool_calls.arguments 必须是合法 JSON 字符串。",
        "unknown_rag_negative": "生成用户问题超出证据、含混或带诱导性的负样本，assistant 必须依次调用意图识别、假设文档和检索工具；当 tool 结果为空、低相关或不足时，最终回答必须明确拒绝幻觉并说明证据不足。",
    }[task_type]

    if task_type in {"intent_hypothesis_rag", "tool_calling_rag", "unknown_rag_negative"}:
        message_example = [
            {"role": "system", "content": "optional string"},
            {"role": "user", "content": "string"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "detect_intent",
                            "arguments": "{\"question\":\"...\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "detect_intent",
                "content": "{\"intent\":\"plot_fact\",\"need_retrieval\":true}",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "build_hypothesis",
                            "arguments": "{\"question\":\"...\",\"intent\":\"plot_fact\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "build_hypothesis",
                "content": "{\"question\":\"...\",\"entities\":[],\"keywords\":[]}",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "retrieve_story_context",
                            "arguments": "{\"question\":\"...\",\"hypothesis\":\"...\",\"keywords\":[\"...\"],\"top_k\":8}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "retrieve_story_context",
                "content": "剧情证据 JSON 或 [] 或低相关检索结果",
            },
            {"role": "assistant", "content": "直接回答用户问题，不暴露检索过程。"},
        ]
    elif task_type == "worldbuilding_qa":
        message_example = [
            {"role": "system", "content": "optional string"},
            {"role": "user", "content": "请先用一句话解释这个设定。"},
            {"role": "assistant", "content": "不超过100字的简短回答"},
            {"role": "user", "content": "那它通常会怎样影响社会或地区？"},
            {"role": "assistant", "content": "不超过100字的简短追问回答"},
            {"role": "user", "content": "再补一句最容易混淆的点。"},
            {"role": "assistant", "content": "不超过100字的简短澄清回答"},
        ]
    else:
        message_example = [
            {"role": "system", "content": "optional string"},
            {"role": "user", "content": "string"},
            {"role": "assistant", "content": "string"},
        ]

    schema_text = {
        "samples": [
            {
                "id": "string",
                "task_type": task_type,
                "messages": message_example,
                "meta": {
                    "grounded": True,
                    "difficulty": "easy|medium|hard",
                    "source_story_ids": ["string"],
                    "source_stage_codes": ["string"],
                    "source_activity_names": ["string"],
                    "worldbuilding_topic": "string or null",
                    "notes": "string",
                },
            }
        ]
    }

    requirements = [
        f"1. {task_specific_rules}",
        "2. `messages` 里的 role 只允许是 `system`、`user`、`assistant`、`tool`。",
        "3. 所有文本使用中文。",
        "4. 输出必须是一个 JSON 对象，顶层只有 `samples` 字段。",
    ]

    if task_type == "worldbuilding_qa":
        requirements.extend(
            [
                "5. 不要引用具体剧情桥段、对白原句、关卡编号、活动标题或角色私密动机。",
                "6. 优先生成可跨剧情复用的定义、解释、比较、概念辨析、常识性制度描述等样本。",
                "7. 如果某个细节不够稳定，就使用保守表达，例如“通常”“一般而言”“常被视为”，不要编造精确设定。",
                "8. 可以生成“这个主题需要区分地区差异或组织差异”的问答，但不要虚构未被广泛接受的细节。",
                "9. 每条样本都必须是多轮对话，至少包含 3 次 `user` 发言和 3 次 `assistant` 发言，且后续轮次必须基于前一轮追问展开。",
                "10. 每次 `assistant` 回答都必须是短答，不超过 100 个中文字符。",
                "11. 每条样本只围绕当前主题包里的一个核心设定展开，不要把多个国家、多个生物体系混在同一条样本里。",
                "12. 不要使用 `tool_calls`，也不要输出 `tool` 角色。",
            ]
        )
        source_block = "世界观主题包：\n" + format_worldbuilding_topic(worldbuilding_topic or {})
    else:
        requirements.extend(
            [
                "5. 只允许使用证据中的信息；如果证据不足，就生成“需要澄清”或“证据不足”的样本。",
                "6. 如果任务类型不是 `intent_hypothesis_rag`、`tool_calling_rag` 或 `unknown_rag_negative`，不要使用 `tool_calls` 和 `tool` role。",
                "7. `multi_turn_dialogue` 必须体现上下文延续，不要把多个单轮问答拼在一起。",
                "8. 工具链样本必须包含：user -> assistant(detect_intent) -> tool -> assistant(build_hypothesis) -> tool -> assistant(retrieve_story_context) -> tool -> assistant(final answer)。",
                "9. 最终 assistant 必须直接回答用户，不要出现“根据检索到的剧情证据”“根据检索结果”“根据证据”等暴露检索过程的措辞。",
            ]
        )
        if task_type in {"intent_hypothesis_rag", "tool_calling_rag", "unknown_rag_negative"}:
            requirements.extend(
                [
                    "10. 必须使用下面这些工具定义，且顺序固定为 detect_intent -> build_hypothesis -> retrieve_story_context：",
                    json.dumps(
                        [INTENT_TOOL_SCHEMA, HYPOTHESIS_TOOL_SCHEMA, RAG_TOOL_SCHEMA],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "11. detect_intent 的 tool 结果必须包含 intent、need_retrieval，可包含 clarification_needed。",
                    "12. build_hypothesis 的 tool 结果必须是结构化 JSON，包含 question、intent、entities、keywords、constraints 或 expected_answer_type。",
                    "13. retrieve_story_context 的 tool 调用必须使用 build_hypothesis 产出的实体和关键词。",
                ]
            )
        if task_type == "unknown_rag_negative":
            requirements.extend(
                [
                    "14. 用户问题必须是证据包无法直接支持的负样本，例如含混指代、错误前提、诱导补完、二创设定、跨章节过度归因或问不存在的细节。",
                    "15. assistant 在完成三段工具链前不得直接回答或猜测。",
                    "16. retrieve_story_context 的 tool content 应表现为空结果、低相关结果或不足以支持结论的检索结果；不要伪造能支持问题前提的证据。",
                    "17. 最终 assistant 必须明确说“证据不足/无法确认/不确定/现有检索结果不足”，并拒绝把诱导问题说成事实。",
                    "18. 最终回答可以用澄闪式轻柔语气，但不能用语气掩盖不确定性。",
                ]
            )
        source_block = "证据包：\n" + format_evidence_pack(evidence_docs)

    user_prompt = (
        f"请基于下面给定的材料生成 {samples_per_request} 条 `{task_type}` 类型训练样本。\n\n"
        "要求：\n"
        + "\n".join(requirements)
        + "\n\n返回格式示例：\n"
        + json.dumps(schema_text, ensure_ascii=False, indent=2)
        + "\n\n"
        + source_block
    )

    return system_prompt, user_prompt


def call_teacher_api(
    api_config: TeacherApiConfig,
    *,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict]:
    api_key = os.environ.get(api_config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing teacher API key env var: {api_config.api_key_env}"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if api_config.extra_headers:
        headers.update(api_config.extra_headers)

    if api_config.api_type == "chat_completions":
        url = api_config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": api_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": api_config.temperature,
            "max_tokens": api_config.max_output_tokens,
        }
        if api_config.json_mode:
            payload["response_format"] = {"type": "json_object"}
    elif api_config.api_type == "responses":
        url = api_config.base_url.rstrip("/") + "/responses"
        payload = {
            "model": api_config.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": api_config.temperature,
            "max_output_tokens": api_config.max_output_tokens,
        }
    else:
        raise ValueError(f"Unsupported api_type: {api_config.api_type}")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=api_config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Teacher API HTTPError {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Teacher API URLError: {exc}") from exc

    decoded = json.loads(raw)
    return extract_response_text(api_config.api_type, decoded), decoded


def extract_response_text(api_type: str, payload: dict) -> str:
    if api_type == "chat_completions":
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("No choices in chat completion payload")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get("text") or item.get("content") or "")
            return "\n".join(texts).strip()
        raise ValueError("Unsupported message content shape")

    if api_type == "responses":
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"]
        output = payload.get("output") or []
        texts: list[str] = []
        for item in output:
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict):
                    text = (
                        part.get("text")
                        or part.get("output_text")
                        or part.get("content")
                        or ""
                    )
                    if text:
                        texts.append(text)
        if texts:
            return "\n".join(texts).strip()
        raise ValueError("No text in responses payload")

    raise ValueError(f"Unsupported api_type: {api_type}")


def parse_teacher_json(text: str) -> dict:
    candidate = text.strip()
    match = JSON_BLOCK_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    return json.loads(candidate)


def categorize_task_type(task_type: str) -> str:
    return TASK_BUCKET_MAP.get(task_type, "knowledge")


def _normalize_tool_calls(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    clean_calls: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        clean_calls.append(
            {
                "id": item.get("id") or "call_1",
                "type": item.get("type") or "function",
                "function": {
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "",
                },
            }
        )
    return clean_calls


def _has_non_empty_content(message: dict) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return content is not None


def _validate_multi_turn_messages(messages: list[dict]) -> bool:
    user_turns = sum(1 for message in messages if message["role"] == "user")
    assistant_turns = sum(1 for message in messages if message["role"] == "assistant")
    if user_turns < 2 or assistant_turns < 2:
        return False
    if any(message["role"] == "tool" for message in messages):
        return False
    return True


def _validate_worldbuilding_messages(messages: list[dict]) -> bool:
    user_turns = sum(1 for message in messages if message["role"] == "user")
    assistant_messages = [
        message
        for message in messages
        if message["role"] == "assistant" and _has_non_empty_content(message)
    ]
    if user_turns < 3 or len(assistant_messages) < 3:
        return False
    if any(message["role"] == "tool" for message in messages):
        return False
    for message in assistant_messages:
        content = normalize_message_content(message.get("content"))
        if len(content) > 100:
            return False
    return True


def _final_assistant_message(messages: list[dict]) -> dict | None:
    return next(
        (
            messages[index]
            for index in range(len(messages) - 1, -1, -1)
            if messages[index]["role"] == "assistant"
            and _has_non_empty_content(messages[index])
        ),
        None,
    )


def _final_answer_hides_retrieval(messages: list[dict]) -> bool:
    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False
    final_text = normalize_message_content(final_assistant.get("content"))
    return not any(marker in final_text for marker in FORBIDDEN_FINAL_ANSWER_MARKERS)


def _tool_call_function_name(message: dict) -> str | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    function = first_call.get("function")
    if not isinstance(function, dict):
        return None
    return function.get("name")


def _tool_call_arguments(message: dict) -> dict | None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None
    function = first_call.get("function")
    if not isinstance(function, dict):
        return None
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def _validate_tool_chain_messages(messages: list[dict]) -> bool:
    expected_tool_names = [
        "detect_intent",
        "build_hypothesis",
        "retrieve_story_context",
    ]
    cursor = -1

    first_user_index = next(
        (index for index, message in enumerate(messages) if message["role"] == "user"),
        None,
    )
    if first_user_index is None:
        return False
    cursor = first_user_index

    for tool_name in expected_tool_names:
        assistant_index = next(
            (
                index
                for index in range(cursor + 1, len(messages))
                if messages[index]["role"] == "assistant"
                and _tool_call_function_name(messages[index]) == tool_name
            ),
            None,
        )
        if assistant_index is None:
            return False

        arguments = _tool_call_arguments(messages[assistant_index])
        if arguments is None:
            return False

        tool_index = next(
            (
                index
                for index in range(assistant_index + 1, len(messages))
                if messages[index]["role"] == "tool"
                and messages[index].get("name") == tool_name
            ),
            None,
        )
        if tool_index is None:
            return False

        if tool_name == "detect_intent":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
        elif tool_name == "build_hypothesis":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
            if not isinstance(arguments.get("intent"), str) or not arguments["intent"].strip():
                return False
        elif tool_name == "retrieve_story_context":
            if not isinstance(arguments.get("question"), str) or not arguments["question"].strip():
                return False
            if not isinstance(arguments.get("hypothesis"), str) or not arguments["hypothesis"].strip():
                return False
            keywords = arguments.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                return False
            top_k = arguments.get("top_k")
            if not isinstance(top_k, int) or top_k <= 0:
                return False

        cursor = tool_index

    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False
    final_index = messages.index(final_assistant)
    if final_index <= cursor:
        return False
    return _final_answer_hides_retrieval(messages)


def _validate_unknown_negative_messages(messages: list[dict]) -> bool:
    if not _validate_tool_chain_messages(messages):
        return False

    final_assistant = _final_assistant_message(messages)
    if final_assistant is None:
        return False

    final_text = normalize_message_content(final_assistant.get("content"))
    if not any(marker in final_text for marker in INSUFFICIENT_EVIDENCE_MARKERS):
        return False

    tool_messages = [message for message in messages if message["role"] == "tool"]
    tool_text = "\n".join(normalize_message_content(message.get("content")) for message in tool_messages)
    weak_tool_markers = (
        "[]",
        "空",
        "低相关",
        "无相关",
        "没有检索到",
        "不足以支持",
        "无法支持",
        "证据不足",
        "未找到",
    )
    return any(marker in tool_text for marker in weak_tool_markers)


def validate_and_normalize_samples(
    payload: dict,
    *,
    expected_task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    request_id: str,
) -> list[dict]:
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Teacher payload must contain a list field named 'samples'")

    source_story_ids = [doc.get("story_id") for doc in evidence_docs if doc.get("story_id")]
    source_stage_codes = [doc.get("stage_code") for doc in evidence_docs if doc.get("stage_code")]
    source_activity_names = [doc.get("activity_name") for doc in evidence_docs if doc.get("activity_name")]

    normalized: list[dict] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue
        task_type = sample.get("task_type") or expected_task_type
        if task_type not in SUPPORTED_TASK_TYPES:
            continue
        messages = sample.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            continue

        clean_messages = []
        valid = True
        has_tool_call = False
        has_tool_message = False

        for message in messages:
            if not isinstance(message, dict):
                valid = False
                break
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                valid = False
                break
            clean_message = {
                "role": role,
                "content": message.get("content", ""),
            }
            if role == "tool":
                clean_message["name"] = message.get("name") or "retrieve_story_context"
                has_tool_message = True
            if role == "assistant" and message.get("tool_calls"):
                clean_tool_calls = _normalize_tool_calls(message.get("tool_calls"))
                if clean_tool_calls:
                    clean_message["tool_calls"] = clean_tool_calls
                    has_tool_call = True
            clean_messages.append(clean_message)

        if not valid:
            continue
        if task_type in {"intent_hypothesis_rag", "tool_calling_rag", "unknown_rag_negative"} and (
            not has_tool_call or not has_tool_message
        ):
            continue
        if task_type not in {"intent_hypothesis_rag", "tool_calling_rag", "unknown_rag_negative"}:
            if any(msg.get("tool_calls") for msg in clean_messages):
                continue
            if any(msg["role"] == "tool" for msg in clean_messages):
                continue
        if task_type == "worldbuilding_qa" and not _validate_worldbuilding_messages(clean_messages):
            continue
        if task_type == "multi_turn_dialogue" and not _validate_multi_turn_messages(clean_messages):
            continue
        if task_type in {"intent_hypothesis_rag", "tool_calling_rag"} and not _validate_tool_chain_messages(clean_messages):
            continue
        if task_type == "unknown_rag_negative" and not _validate_unknown_negative_messages(clean_messages):
            continue

        meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
        normalized_sample = {
            "id": sample.get("id") or f"{request_id}-{index:04d}",
            "task_type": task_type,
            "bucket": categorize_task_type(task_type),
            "messages": clean_messages,
            "tools": [INTENT_TOOL_SCHEMA, HYPOTHESIS_TOOL_SCHEMA, RAG_TOOL_SCHEMA]
            if task_type in {"intent_hypothesis_rag", "tool_calling_rag", "unknown_rag_negative"}
            else [],
            "meta": {
                "category": categorize_task_type(task_type),
                "grounded": True,
                "difficulty": meta.get("difficulty") or "medium",
                "notes": meta.get("notes") or "",
                "source_story_ids": meta.get("source_story_ids") or source_story_ids,
                "source_stage_codes": meta.get("source_stage_codes") or source_stage_codes,
                "source_activity_names": meta.get("source_activity_names") or source_activity_names,
                "worldbuilding_topic": meta.get("worldbuilding_topic")
                or (worldbuilding_topic.get("topic") if worldbuilding_topic else None),
                "generation_mode": (
                    "topic_driven" if task_type == "worldbuilding_qa" else "evidence_grounded"
                ),
                "request_id": request_id,
            },
        }
        normalized.append(normalized_sample)
    return normalized


def dedupe_samples(samples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for sample in samples:
        fingerprint = make_sample_fingerprint(sample)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        sample["fingerprint"] = fingerprint
        unique.append(sample)
    return unique


def split_samples(
    samples: list[dict],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    train_cut = int(len(shuffled) * train_ratio)
    val_cut = train_cut + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_cut],
        "val": shuffled[train_cut:val_cut],
        "test": shuffled[val_cut:],
    }


def build_request_record(
    *,
    request_id: str,
    task_type: str,
    evidence_docs: list[dict],
    worldbuilding_topic: dict[str, Any] | None,
    evidence_mode: str | None,
    retrieval_query: str | None,
    retrieval_seed_doc_id: str | None,
    system_prompt: str,
    user_prompt: str,
    raw_text: str | None,
    parsed_ok: bool,
    accepted_samples: int,
    latency_seconds: float,
    error: str | None = None,
) -> dict:
    return {
        "request_id": request_id,
        "task_type": task_type,
        "bucket": categorize_task_type(task_type),
        "evidence_doc_ids": [doc["id"] for doc in evidence_docs],
        "evidence_mode": evidence_mode,
        "retrieval_query": retrieval_query,
        "retrieval_seed_doc_id": retrieval_seed_doc_id,
        "worldbuilding_topic": worldbuilding_topic,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_text": raw_text,
        "parsed_ok": parsed_ok,
        "accepted_samples": accepted_samples,
        "latency_seconds": latency_seconds,
        "error": error,
        "created_at": int(time.time()),
    }
