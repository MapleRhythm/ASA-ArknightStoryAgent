#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    source_bytes = args.matrix.read_bytes()
    payload = json.loads(source_bytes)
    rows = payload["rows"]
    question_to_sources: dict[str, list[str]] = {}
    for row in rows:
        question = str(row["question"]).strip()
        if not question:
            raise SystemExit(f"empty question in row {row.get('id')}")
        question_to_sources.setdefault(question, []).append(str(row["source_id"]))

    ordered = sorted(
        question_to_sources,
        key=lambda question: (hashlib.sha256(question.encode("utf-8")).hexdigest(), question),
    )
    question_bytes = ("\n".join(ordered) + "\n").encode("utf-8")
    args.questions.write_bytes(question_bytes)
    manifest = {
        "version": "fixed58_unique_questions_v1",
        "evaluation_only": True,
        "source_matrix": str(args.matrix),
        "source_matrix_sha256": sha256_bytes(source_bytes),
        "source_state_count": len(rows),
        "unique_question_count": len(ordered),
        "selection": "unique exact question text; sorted by sha256(question), independent of answer/evidence",
        "questions_sha256": sha256_bytes(question_bytes),
        "questions": [
            {
                "index": index,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "question": question,
                "historical_state_count": len(question_to_sources[question]),
                "historical_source_ids": sorted(question_to_sources[question]),
            }
            for index, question in enumerate(ordered, start=1)
        ],
        "limitations": [
            "These 58 questions are development questions derived from 79 historical evidence states.",
            "The fresh E2E run must not reuse historical evidence or historical planner state.",
            "Question-level results are not an unseen holdout or a certified human-gold accuracy estimate.",
        ],
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "unique_question_count": len(ordered),
        "questions_sha256": manifest["questions_sha256"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
