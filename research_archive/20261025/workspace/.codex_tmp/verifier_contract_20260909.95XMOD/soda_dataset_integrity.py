"""CPU-only shared integrity rules for SODA verifier builds and shard merges."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
import re
from typing import Any


QUESTION_RE = re.compile(r"(?m)^(?:question|用户原问题|问题)[：:][ \t]*(.+?)[ \t]*$")


def strict_json_object(text: str) -> dict[str, Any] | None:
    """Do not repair an unparseable model response into a positive target."""
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    try:
        payload = json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def training_input_signature(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("Training record must contain input and response messages")
    value = {
        "system": record.get("system"), "tools": record.get("tools"),
        "conversations": conversations[:-1],
    }
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def question_split_key(record: dict[str, Any]) -> str:
    for message in record.get("conversations", []):
        if message.get("from") != "human":
            continue
        match = QUESTION_RE.search(str(message.get("value") or ""))
        if match:
            question = re.sub(r"\s+", "", match.group(1)).strip("？?。！!，,；;：:\"'")
            if question:
                return "question:" + question
    # Never separate a chosen/rejected pair just because its row IDs differ.
    return "input:" + training_input_signature(record)


def assert_no_label_conflicts(records: list[dict[str, Any]]) -> None:
    labels: dict[tuple[str, str], bool] = {}
    for record in records:
        tag = record.get("kto_tag")
        if type(tag) is not bool:
            raise ValueError("kto_tag must be a boolean")
        input_signature = training_input_signature(record)
        raw = record["conversations"][-1].get("value")
        if not isinstance(raw, str):
            raise ValueError("Training response must be text")
        parsed = strict_json_object(raw)
        response = json.dumps(parsed, ensure_ascii=False, sort_keys=True) if parsed is not None else raw
        key = (input_signature, response)
        if key in labels and labels[key] is not tag:
            raise ValueError(f"KTO label conflict for input {key[0]}")
        labels[key] = tag


def split_records(
    records: list[dict[str, Any]], *, seed: int, val_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split once after combining shards; never concatenate shard-local splits."""
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must satisfy 0 <= val_ratio < 1")
    assert_no_label_conflicts(records)
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_question[question_split_key(record)].append(record)
    keys = sorted(by_question)
    rng = random.Random(seed)
    rng.shuffle(keys)
    target_val = int(round(len(records) * val_ratio)) if val_ratio > 0 else 0
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    val_count = 0
    for index, key in enumerate(keys):
        bucket = by_question[key]
        # Keep at least one complete question group for training.
        if val_count < target_val and index < len(keys) - 1:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
