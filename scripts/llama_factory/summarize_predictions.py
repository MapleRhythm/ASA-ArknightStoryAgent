#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "llama_factory" / "teacher_v2_plus_prompt_supplement_v2" / "test.json"
)
DEFAULT_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "llama_factory_eval"
    / "teacher_v2_plus_prompt_supplement_v2_qwen35_4b"
    / "generated_predictions.jsonl"
)
DEFAULT_OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "llama_factory_eval"
    / "teacher_v2_plus_prompt_supplement_v2_qwen35_4b"
    / "custom_metrics.json"
)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def char_f1(prediction: str, reference: str) -> float:
    pred_counter = Counter(normalize_text(prediction))
    ref_counter = Counter(normalize_text(reference))
    overlap = sum((pred_counter & ref_counter).values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())
    if pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def last_assistant_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    for message in reversed(conversations):
        if message.get("from") == "gpt":
            return str(message.get("value") or "")
    return ""


def build_summary(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(references) != len(predictions):
        raise ValueError(
            f"Prediction count mismatch: references={len(references)} predictions={len(predictions)}"
        )

    overall_exact = 0
    overall_char_f1 = 0.0
    per_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"samples": 0, "exact_match": 0, "char_f1_sum": 0.0}
    )
    hard_cases: list[dict[str, Any]] = []

    for reference, prediction in zip(references, predictions, strict=True):
        label = str(prediction.get("label") or last_assistant_text(reference))
        predict = str(prediction.get("predict") or "")
        task_type = str(reference.get("task_type") or "unknown")
        exact = int(normalize_text(predict) == normalize_text(label))
        score = char_f1(predict, label)

        overall_exact += exact
        overall_char_f1 += score

        per_task_entry = per_task[task_type]
        per_task_entry["samples"] += 1
        per_task_entry["exact_match"] += exact
        per_task_entry["char_f1_sum"] += score

        if score < 0.4:
            hard_cases.append(
                {
                    "id": reference.get("id"),
                    "task_type": task_type,
                    "label": label,
                    "predict": predict,
                    "char_f1": round(score, 4),
                }
            )

    summary = {
        "samples": len(references),
        "exact_match": round(overall_exact / len(references), 4) if references else 0.0,
        "char_f1": round(overall_char_f1 / len(references), 4) if references else 0.0,
        "per_task_type": {},
        "hard_cases": hard_cases[:20],
    }
    for task_type, stats in sorted(per_task.items()):
        samples = stats["samples"]
        summary["per_task_type"][task_type] = {
            "samples": samples,
            "exact_match": round(stats["exact_match"] / samples, 4) if samples else 0.0,
            "char_f1": round(stats["char_f1_sum"] / samples, 4) if samples else 0.0,
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize LLaMA-Factory generated_predictions.jsonl against the converted test set."
    )
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE_FILE)
    parser.add_argument("--predictions-file", type=Path, default=DEFAULT_PREDICTIONS_FILE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = load_json(args.reference_file.resolve())
    predictions = load_jsonl(args.predictions_file.resolve())
    summary = build_summary(references, predictions)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
