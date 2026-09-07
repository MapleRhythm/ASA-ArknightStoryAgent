#!/usr/bin/env python3
"""Build a natural-language control from a reviewed reader-evidence matrix.

The control preserves the question and visible evidence while removing the
Exx action, JSON, and citation protocol. Human answer facts are retained only
as hidden diagnostic metadata for post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(r"^question:\s*(.+)$", re.MULTILINE)
EVIDENCE_RE = re.compile(
    r"^evidence:\s*\n(.*?)\noutput_schema:\s*[^\n]+(?:\n|$)",
    re.MULTILINE | re.DOTALL,
)
DEFAULT_SYSTEM = (
    "你是《明日方舟》剧情问答阅读器。只依据当前提供的证据，用简洁自然语言"
    "完整回答问题的所有部分。不要使用外部知识，不要输出 JSON、E 编号或推理"
    "过程；证据不足时明确说明不足。"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    annotations: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError(f"invalid annotation at line {line_number}")
        if value["id"] in annotations:
            raise ValueError(f"duplicate annotation id: {value['id']}")
        annotations[value["id"]] = value
    return annotations


def extract_question_and_evidence(prompt: str) -> tuple[str, str]:
    question = QUESTION_RE.search(prompt)
    evidence = EVIDENCE_RE.search(prompt)
    if question is None or evidence is None:
        raise ValueError("cannot parse question/evidence from Exx prompt")
    return question.group(1).strip(), evidence.group(1).strip()


def answer_fact_texts(annotation: dict[str, Any]) -> list[str]:
    facts = annotation.get("answer_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError(f"annotation has no answer facts: {annotation.get('id')}")
    texts = [
        str(fact.get("fact") or "").strip()
        for fact in facts
        if isinstance(fact, dict)
    ]
    if len(texts) != len(facts) or not all(texts):
        raise ValueError(f"annotation has invalid answer facts: {annotation.get('id')}")
    return texts


def build_control_rows(
    matrix_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    *,
    source_condition: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in matrix_rows:
        diagnostic = row.get("diagnostic")
        if not isinstance(diagnostic, dict):
            continue
        if diagnostic.get("condition") != source_condition:
            continue
        source_id = str(diagnostic.get("source_id") or "")
        if not source_id or source_id in seen_sources:
            raise ValueError(f"missing or duplicate source id: {source_id!r}")
        if diagnostic.get("reference_sufficiency") != "sufficient":
            raise ValueError(f"control requires sufficient evidence: {row.get('id')}")
        annotation = annotations.get(source_id)
        if annotation is None or annotation.get("review_status") != "approved":
            raise ValueError(f"missing approved annotation: {source_id}")
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"row has no prompt: {row.get('id')}")
        prompt = str(conversations[0].get("value") or "")
        question, evidence = extract_question_and_evidence(prompt)
        control_condition = f"{source_condition}_natural_language"
        selected.append(
            {
                # The existing deterministic Transformers runner selects this
                # task type. The true control identity is recorded below and
                # the prompt contains no grounded-action protocol.
                "task_type": "grounded_action_generation",
                "id": f"{source_id}::{control_condition}",
                "system": DEFAULT_SYSTEM,
                "conversations": [
                    {
                        "from": "human",
                        "value": "\n".join(
                            [
                                f"问题：{question}",
                                "",
                                "当前证据：",
                                evidence,
                                "",
                                "请直接给出完整、简洁的自然语言答案。",
                            ]
                        ),
                    }
                ],
                "diagnostic": {
                    **diagnostic,
                    "condition": control_condition,
                    "source_condition": source_condition,
                    "control_type": "natural_language_short_answer",
                    "expected_output": "natural_language",
                    "reference_answer_facts": answer_fact_texts(annotation),
                    "reference_annotation_reviewer": annotation.get("reviewer"),
                },
            }
        )
        seen_sources.add(source_id)
    if not selected:
        raise ValueError(f"no rows selected for condition: {source_condition}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", default="minimal_sufficient")
    args = parser.parse_args()

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    for path in (args.output, metadata_path):
        if path.exists():
            parser.error(f"refusing to overwrite output: {path}")
    if not args.matrix.is_file() or not args.annotations.is_file():
        parser.error("matrix or annotations path does not exist")

    rows = build_control_rows(
        read_json_array(args.matrix),
        read_annotations(args.annotations),
        source_condition=args.condition,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "protocol": "asa_reader_natural_language_control_v1",
        "matrix": str(args.matrix),
        "matrix_sha256": sha256_file(args.matrix),
        "annotations": str(args.annotations),
        "annotations_sha256": sha256_file(args.annotations),
        "condition": args.condition,
        "rows": len(rows),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "answer_rubric_visible_to_model": False,
        "evidence_order_preserved": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
