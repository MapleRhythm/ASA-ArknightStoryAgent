#!/usr/bin/env python3
"""Build a leakage-audited Exx evaluation split.

The primary split excludes every normalized question seen by the actual SFT
train file.  It can combine labelled Exx validation rows (for model-only
evaluation) with external question lists (for end-to-end evaluation).  Inputs
remain read-only and outputs are versioned by the caller.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(r"^(?:question|问题)[:：]\s*(.+)$", re.MULTILINE)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question(text: str) -> str:
    text = re.sub(r"\s+", "", str(text or ""))
    text = text.translate(str.maketrans("‘’“”", "''''"))
    return text.strip("？?。！!，,；;：:\"'")


def record_question(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return ""
    match = QUESTION_RE.search(str(conversations[0].get("value") or ""))
    return match.group(1).strip() if match else ""


def text_questions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--labelled-json", action="append", type=Path, default=[])
    parser.add_argument("--question-file", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.88,
        help="Reject evaluation questions too similar to any train question; 0 disables.",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"output directory must be empty or absent: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_json(args.train_json)
    train_questions = {
        normalize_question(question)
        for row in train_rows
        if (question := record_question(row))
    }
    train_question_list = sorted(train_questions)

    def overlaps_train(key: str) -> tuple[bool, str]:
        if key in train_questions:
            return True, "exact"
        if args.near_duplicate_threshold <= 0:
            return False, ""
        candidates = difflib.get_close_matches(
            key,
            train_question_list,
            n=1,
            cutoff=args.near_duplicate_threshold,
        )
        return (True, "near_duplicate") if candidates else (False, "")
    labelled_rows: list[dict[str, Any]] = []
    rejected = Counter()
    labelled_seen: set[str] = set()
    e2e_seen: set[str] = set()
    inputs: list[dict[str, Any]] = [
        {
            "kind": "train",
            "path": str(args.train_json.resolve()),
            "rows": len(train_rows),
            "sha256": sha256_file(args.train_json),
        }
    ]
    for path in args.labelled_json:
        rows = read_json(path)
        inputs.append(
            {"kind": "labelled", "path": str(path.resolve()), "rows": len(rows), "sha256": sha256_file(path)}
        )
        for row in rows:
            question = record_question(row)
            key = normalize_question(question)
            if not key:
                rejected["labelled:missing_question"] += 1
            elif (overlap := overlaps_train(key))[0]:
                rejected[f"labelled:train_{overlap[1]}"] += 1
            elif key in labelled_seen:
                rejected["labelled:duplicate"] += 1
            else:
                labelled_seen.add(key)
                labelled_rows.append(row)

    e2e_questions: list[str] = []
    for path in args.question_file:
        questions = text_questions(path)
        inputs.append(
            {"kind": "questions", "path": str(path.resolve()), "rows": len(questions), "sha256": sha256_file(path)}
        )
        for question in questions:
            key = normalize_question(question)
            if (overlap := overlaps_train(key))[0]:
                rejected[f"questions:train_{overlap[1]}"] += 1
            elif not key:
                rejected["questions:missing_question"] += 1
            elif key in e2e_seen:
                rejected["questions:duplicate"] += 1
            else:
                e2e_seen.add(key)
                e2e_questions.append(question)

    write_json(args.output_dir / "model_only_labelled.json", labelled_rows)
    (args.output_dir / "e2e_questions.txt").write_text(
        "\n".join(e2e_questions) + ("\n" if e2e_questions else ""), encoding="utf-8"
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "exact_normalized_question_exclusion": True,
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "primary_metrics_must_not_use_train_overlap": True,
            "legacy_eval_sets_are_regression_only": True,
        },
        "inputs": inputs,
        "train_unique_questions": len(train_questions),
        "output_counts": {
            "model_only_labelled": len(labelled_rows),
            "e2e_questions": len(e2e_questions),
        },
        "rejected": dict(sorted(rejected.items())),
    }
    write_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
