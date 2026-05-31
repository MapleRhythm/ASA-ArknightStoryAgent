#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/llama_factory/soda_blackbox_deepseek_v1_550_parallel_api_verifier_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def response_payload(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return {}
    raw = str(conversations[-1].get("value") or "").strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def action_of(record: dict[str, Any]) -> str:
    return str(response_payload(record).get("next_action") or "<none>")


def summarize_verifier(records: list[dict[str, Any]]) -> tuple[Counter, dict[str, list[dict[str, Any]]]]:
    stats: Counter = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stats["verifier_records"] += 1
        error = str(record.get("error") or "")
        if error:
            stats["verifier_error"] += 1
            if len(examples["verifier_error"]) < 5:
                examples["verifier_error"].append(record)
            continue
        verdict = record.get("verifier") if isinstance(record.get("verifier"), dict) else {}
        action = str(verdict.get("correct_action") or "<missing>")
        stats[f"correct_action:{action}"] += 1
        stats[f"evidence_sufficient:{bool(verdict.get('evidence_sufficient'))}"] += 1
        stats[f"use_for_training:{bool(verdict.get('use_for_training', True))}"] += 1
        if verdict.get("teacher_answer_uses_prior_knowledge"):
            stats["teacher_answer_uses_prior_knowledge"] += 1
        for field in ("student_action_error", "teacher_action_error", "label_reason"):
            value = str(verdict.get(field) or "<missing>")
            stats[f"{field}:{value}"] += 1
            if value not in {"", "none", "<missing>"} and len(examples[f"{field}:{value}"]) < 5:
                examples[f"{field}:{value}"].append(record)
    return stats, examples


def summarize_dataset(records: list[dict[str, Any]], prefix: str) -> Counter:
    stats: Counter = Counter()
    for record in records:
        task_type = str(record.get("task_type") or record.get("meta", {}).get("task_type") or "<missing>")
        kto_tag = bool(record.get("kto_tag"))
        stats[f"{prefix}:records"] += 1
        stats[f"{prefix}:kto_tag:{kto_tag}"] += 1
        stats[f"{prefix}:task_type:{task_type}"] += 1
        if task_type == "conclusion_generation":
            stats[f"{prefix}:conclusion_action:{kto_tag}:{action_of(record)}"] += 1
            reason = str(record.get("meta", {}).get("api_verifier_reason") or "<original>")
            stats[f"{prefix}:reason:{reason}"] += 1
    return stats


def format_counter(counter: Counter) -> list[str]:
    return [f"- {key}: {value}" for key, value in sorted(counter.items())]


def render_examples(examples: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for key, records in sorted(examples.items()):
        if not records:
            continue
        lines.extend(["", f"### {key}", ""])
        for record in records[:3]:
            verdict = record.get("verifier") if isinstance(record.get("verifier"), dict) else {}
            lines.append(
                "- "
                + json.dumps(
                    {
                        "prompt_key": record.get("prompt_key"),
                        "question": record.get("question"),
                        "round": record.get("round"),
                        "correct_action": verdict.get("correct_action"),
                        "student_action_error": verdict.get("student_action_error"),
                        "teacher_action_error": verdict.get("teacher_action_error"),
                        "label_reason": verdict.get("label_reason"),
                        "missing_slots": verdict.get("missing_slots"),
                    },
                    ensure_ascii=False,
                )
            )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit API-verifier relabeled SODA/KTO dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else PROJECT_ROOT / args.dataset_dir
    output_path = args.output or dataset_dir / "api_verifier_audit_report.md"
    verifier_records = read_jsonl(dataset_dir / "api_verifier_records.jsonl")
    train_records = read_json(dataset_dir / "train.json") or []
    val_records = read_json(dataset_dir / "val.json") or []
    build_summary = read_json(dataset_dir / "build_summary.json")

    verifier_stats, examples = summarize_verifier(verifier_records)
    dataset_stats = Counter()
    dataset_stats.update(summarize_dataset(train_records, "train"))
    dataset_stats.update(summarize_dataset(val_records, "val"))

    lines = [
        "# SODA API Verifier Audit",
        "",
        f"- dataset_dir: {dataset_dir}",
        f"- verifier_records: {len(verifier_records)}",
        f"- train_records: {len(train_records)}",
        f"- val_records: {len(val_records)}",
        "",
        "## Build Summary",
        "",
    ]
    if build_summary:
        lines.extend(f"- {key}: {value}" for key, value in build_summary.items() if key != "stats")
        if isinstance(build_summary.get("stats"), dict):
            lines.extend(["", "### Build Stats", ""])
            lines.extend(f"- {key}: {value}" for key, value in sorted(build_summary["stats"].items()))
    else:
        lines.append("- missing build_summary.json")

    lines.extend(["", "## Verifier Stats", ""])
    lines.extend(format_counter(verifier_stats) or ["- no verifier records"])
    lines.extend(["", "## Dataset Stats", ""])
    lines.extend(format_counter(dataset_stats) or ["- no train/val records"])
    lines.extend(["", "## Example Decisions", ""])
    lines.extend(render_examples(examples) or ["- no examples"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
