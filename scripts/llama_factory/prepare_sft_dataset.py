#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "sft_data" / "teacher_v2_plus_prompt_supplement_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "llama_factory" / "teacher_v2_plus_prompt_supplement_v2"
CORE_JSON_TASK_TYPES = {
    "user_question_hypothesis_generation",
    "follow_up_hypothesis_generation",
    "conclusion_generation",
}

ROLE_TAGS = {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "observation_tag": "observation",
    "function_tag": "function_call",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def validate_record(record: dict[str, Any]) -> None:
    messages = record.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("empty messages")
    system_positions = [idx for idx, message in enumerate(messages) if message.get("role") == "system"]
    if len(system_positions) > 1:
        raise ValueError("multiple system messages")
    if system_positions and system_positions[0] != 0:
        raise ValueError("system message is not first")
    if messages[-1].get("role") != "assistant":
        raise ValueError("last message is not assistant")
    if record.get("task_type") in CORE_JSON_TASK_TYPES:
        if any(message.get("role") == "tool" or message.get("tool_calls") for message in messages):
            raise ValueError("legacy tool trace in core json task")
        final_content = messages[-1].get("content")
        if not isinstance(final_content, str):
            raise ValueError("assistant content is not string")
        try:
            payload = json.loads(final_content)
        except json.JSONDecodeError as exc:
            raise ValueError("assistant content is not valid json") from exc
        if not isinstance(payload, dict):
            raise ValueError("assistant content is not a json object")


def normalize_tool_call(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    normalized_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function_payload = tool_call.get("function") or {}
        arguments = function_payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = arguments
        normalized_calls.append(
            {
                "name": function_payload.get("name", ""),
                "arguments": arguments,
            }
        )
    payload: dict[str, Any] | list[dict[str, Any]]
    payload = normalized_calls[0] if len(normalized_calls) == 1 else normalized_calls
    return {
        "from": ROLE_TAGS["function_tag"],
        "value": json.dumps(payload, ensure_ascii=False),
    }


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    conversations: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = str(message.get("content") or "").strip()
            if content:
                system_parts.append(content)
            continue
        if role == "user":
            conversations.append(
                {
                    "from": ROLE_TAGS["user_tag"],
                    "value": str(message.get("content") or ""),
                }
            )
            continue
        if role == "tool":
            conversations.append(
                {
                    "from": ROLE_TAGS["observation_tag"],
                    "value": str(message.get("content") or ""),
                }
            )
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                conversations.append(normalize_tool_call(message))
            else:
                conversations.append(
                    {
                        "from": ROLE_TAGS["assistant_tag"],
                        "value": str(message.get("content") or ""),
                    }
                )
            continue
        raise ValueError(f"Unsupported role: {role!r}")

    if not conversations:
        raise ValueError("No trainable conversations found in sample.")

    return "\n\n".join(system_parts), conversations


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    system_prompt, conversations = convert_messages(record.get("messages") or [])
    output = {
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "bucket": record.get("bucket") or record.get("meta", {}).get("category"),
        "system": system_prompt,
        "tools": json.dumps(record.get("tools") or [], ensure_ascii=False),
        "conversations": conversations,
        "meta": record.get("meta") or {},
    }
    return output


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_dataset_info(entries: dict[str, str]) -> dict[str, Any]:
    dataset_info: dict[str, Any] = {}
    for dataset_name, file_name in sorted(entries.items()):
        dataset_info[dataset_name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
            },
            "tags": dict(ROLE_TAGS),
        }
    return dataset_info


def export_split(
    input_path: Path,
    output_path: Path,
    *,
    dataset_name: str,
    dataset_entries: dict[str, str],
) -> tuple[int, Counter]:
    records = load_jsonl(input_path)
    converted: list[dict[str, Any]] = []
    for record in records:
        validate_record(record)
        converted.append(convert_record(record))
    write_json(output_path, converted)
    dataset_entries[dataset_name] = output_path.name

    counter: Counter = Counter()
    for record in records:
        counter[record.get("task_type") or "unknown"] += 1
    return len(converted), counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert generated SFT jsonl files into LLaMA-Factory sharegpt datasets."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_prefix = source_dir.name

    dataset_entries: dict[str, str] = {}
    manifest: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        input_path = source_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.json"
        count, task_counter = export_split(
            input_path,
            output_path,
            dataset_name=f"{dataset_prefix}_{split}",
            dataset_entries=dataset_entries,
        )
        manifest["splits"][split] = {
            "samples": count,
            "task_type_distribution": dict(task_counter),
        }

    for bucket in ("style", "knowledge", "tool"):
        bucket_dir = source_dir / bucket
        if not bucket_dir.exists():
            continue
        for split in ("train", "val", "test"):
            input_path = bucket_dir / f"{split}.jsonl"
            if not input_path.exists():
                continue
            output_path = output_dir / f"{bucket}_{split}.json"
            count, task_counter = export_split(
                input_path,
                output_path,
                dataset_name=f"{dataset_prefix}_{bucket}_{split}",
                dataset_entries=dataset_entries,
            )
            manifest["splits"][f"{bucket}_{split}"] = {
                "samples": count,
                "task_type_distribution": dict(task_counter),
            }

    write_json(output_dir / "dataset_info.json", build_dataset_info(dataset_entries))
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
