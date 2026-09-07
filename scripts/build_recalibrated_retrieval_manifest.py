#!/usr/bin/env python3
"""Build a strict retrieval manifest from Exx tasks and recalibrated labels.

The Exx task file preserves the mapping from prompt-local E labels to corpus
document IDs.  The preference evaluation identifies held-out positive tasks,
while the recalibrated dataset supplies the post-audit action and citations.
Only recalibrated ``answer_directly`` rows with fully resolvable citations are
emitted.

By default, every occurrence of a duplicated normalized question is excluded.
Different prompts can support the same question with alternative evidence
sets; arbitrarily selecting one would make all-gold retrieval coverage look
more precise than the labels justify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "asa_recalibrated_retrieval_manifest_v2"


class ManifestBuildError(ValueError):
    """Raised when source data cannot be converted without guessing."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            for key in ("rows", "records", "items"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
            else:
                values = [value]
        else:
            raise ManifestBuildError(f"expected JSON object/array: {path}")
    if not all(isinstance(row, dict) for row in values):
        raise ManifestBuildError(f"all rows must be JSON objects: {path}")
    return [row for row in values if isinstance(row, dict)]


def object_value(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ManifestBuildError(f"{field} is not valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ManifestBuildError(f"{field} must be a JSON object")


def preference_task_id(row: dict[str, Any]) -> str:
    meta = object_value(row.get("meta") or {}, field="preference meta")
    return str(meta.get("task_id") or "").strip()


def is_positive_preference(row: dict[str, Any]) -> bool:
    if row.get("kto_tag") is True:
        return True
    meta = object_value(row.get("meta") or {}, field="preference meta")
    return str(meta.get("preference_polarity") or "").strip() == "desirable"


def story_families(row: dict[str, Any]) -> list[str]:
    meta = object_value(row.get("meta") or {}, field="preference meta")
    value = meta.get("story_families")
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(str(item).strip() for item in value if str(item).strip())
    )


def recalibrated_task_id(row: dict[str, Any]) -> str:
    row_id = str(row.get("id") or "").strip()
    return row_id.removeprefix("relabel-") if row_id.startswith("relabel-") else ""


def assistant_payload(row: dict[str, Any]) -> dict[str, Any]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ManifestBuildError("missing conversations")
    for message in reversed(conversations):
        if not isinstance(message, dict):
            continue
        if str(message.get("from") or "").strip().lower() not in {"gpt", "assistant"}:
            continue
        return object_value(message.get("value"), field="assistant value")
    raise ManifestBuildError("missing assistant message")


def supported_fact_citations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    facts = payload.get("supported_facts")
    if not isinstance(facts, list) or not facts:
        return []
    result: list[dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ManifestBuildError("supported_facts items must be objects")
        fact_text = str(fact.get("fact") or "").strip()
        if not fact_text:
            raise ManifestBuildError("supported fact missing fact text")
        evidence_ids = fact.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ManifestBuildError("supported fact missing evidence_ids")
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for item in evidence_ids:
            evidence_id = str(item).strip()
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                normalized_ids.append(evidence_id)
        if not normalized_ids:
            raise ManifestBuildError("supported fact has no valid evidence_ids")
        result.append({"fact": fact_text, "evidence_ids": normalized_ids})
    return result


def cited_evidence_ids(facts: list[dict[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            evidence_id
            for fact in facts
            for evidence_id in fact["evidence_ids"]
        )
    )


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", question).strip("？?。！!，,；;：:\"'")


def task_hypothesis(task: dict[str, Any]) -> dict[str, Any]:
    hypothesis = object_value(task.get("hypothesis"), field="task hypothesis")
    question = str(task.get("question") or "").strip()
    if not question:
        raise ManifestBuildError("task missing question")
    hypothesis = dict(hypothesis)
    hypothesis["question"] = str(hypothesis.get("question") or question)
    return hypothesis


def evidence_mapping(task: dict[str, Any]) -> dict[str, str]:
    evidence = task.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ManifestBuildError("task missing evidence")
    result: dict[str, str] = {}
    for item in evidence:
        if not isinstance(item, dict):
            raise ManifestBuildError("task evidence items must be objects")
        label = str(item.get("label") or "").strip()
        doc_id = str(item.get("doc_id") or "").strip()
        if not label or not doc_id:
            raise ManifestBuildError("task evidence missing label/doc_id")
        if label in result and result[label] != doc_id:
            raise ManifestBuildError(f"conflicting document IDs for {label}")
        result[label] = doc_id
    return result


def build_manifest_rows(
    *,
    tasks: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    recalibrated: list[dict[str, Any]],
    drop_duplicate_questions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_by_id = {
        str(row.get("task_id") or "").strip(): row
        for row in tasks
        if str(row.get("task_id") or "").strip()
    }
    recalibrated_by_task: dict[str, dict[str, Any]] = {}
    duplicate_recalibrated_ids: set[str] = set()
    for row in recalibrated:
        task_id = recalibrated_task_id(row)
        if not task_id:
            continue
        if task_id in recalibrated_by_task:
            duplicate_recalibrated_ids.add(task_id)
        else:
            recalibrated_by_task[task_id] = row

    exclusions: Counter[str] = Counter()
    exclusion_examples: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    seen_preference_tasks: set[str] = set()

    def exclude(reason: str, task_id: str, detail: str = "") -> None:
        exclusions[reason] += 1
        if len(exclusion_examples) < 30:
            exclusion_examples.append(
                {"task_id": task_id, "reason": reason, "detail": detail}
            )

    for preference in preferences:
        if not is_positive_preference(preference):
            continue
        try:
            task_id = preference_task_id(preference)
        except ManifestBuildError as exc:
            exclude("invalid_preference_meta", "", str(exc))
            continue
        if not task_id:
            exclude("missing_task_id", "")
            continue
        if task_id in seen_preference_tasks:
            exclude("duplicate_positive_task", task_id)
            continue
        seen_preference_tasks.add(task_id)
        task = tasks_by_id.get(task_id)
        if task is None:
            exclude("missing_task", task_id)
            continue
        if task_id in duplicate_recalibrated_ids:
            exclude("duplicate_recalibrated_task", task_id)
            continue
        recalibrated_row = recalibrated_by_task.get(task_id)
        if recalibrated_row is None:
            exclude("missing_recalibrated_task", task_id)
            continue
        try:
            payload = assistant_payload(recalibrated_row)
            if payload.get("next_action") != "answer_directly":
                exclude(
                    "recalibrated_not_answer_directly",
                    task_id,
                    str(payload.get("next_action") or ""),
                )
                continue
            fact_citations = supported_fact_citations(payload)
            cited_ids = cited_evidence_ids(fact_citations)
            if not cited_ids:
                exclude("recalibrated_no_citations", task_id)
                continue
            mapping = evidence_mapping(task)
            missing_ids = [item for item in cited_ids if item not in mapping]
            if missing_ids:
                exclude("unresolved_evidence_id", task_id, ",".join(missing_ids))
                continue
            hypothesis = task_hypothesis(task)
        except ManifestBuildError as exc:
            exclude("invalid_source_row", task_id, str(exc))
            continue
        question = str(task.get("question") or "").strip()
        families = story_families(preference)
        eligible.append(
            {
                "manifest_version": MANIFEST_VERSION,
                "id": task_id,
                "question": question,
                "question_family": "|".join(sorted(families))
                or normalize_question(question),
                "query_type": str(hypothesis.get("query_type") or "unknown"),
                "gold_evidence_ids": [mapping[item] for item in cited_ids],
                "gold_supported_facts": [
                    {
                        "fact": fact["fact"],
                        "gold_evidence_ids": [
                            mapping[item] for item in fact["evidence_ids"]
                        ],
                        "source_evidence_ids": fact["evidence_ids"],
                    }
                    for fact in fact_citations
                ],
                "hypothesis": hypothesis,
                "source": {
                    "task_round": str(task.get("round") or ""),
                    "story_families": families,
                    "recalibrated_row_id": str(recalibrated_row.get("id") or ""),
                    "recalibrated_evidence_ids": cited_ids,
                },
            }
        )

    if drop_duplicate_questions:
        by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            by_question[normalize_question(row["question"])].append(row)
        deduplicated: list[dict[str, Any]] = []
        for rows in by_question.values():
            if len(rows) == 1:
                deduplicated.extend(rows)
                continue
            for row in rows:
                exclude("duplicate_question", row["id"], row["question"])
        eligible = deduplicated

    eligible.sort(key=lambda row: row["id"])
    audit = {
        "manifest_version": MANIFEST_VERSION,
        "tasks_total": len(tasks),
        "preferences_total": len(preferences),
        "positive_preference_tasks": len(seen_preference_tasks),
        "recalibrated_total": len(recalibrated),
        "rows_written": len(eligible),
        "exclusions": dict(sorted(exclusions.items())),
        "exclusion_examples": exclusion_examples,
        "question_family_count": len(
            {str(row["question_family"]) for row in eligible}
        ),
        "query_type_counts": dict(
            sorted(Counter(str(row["query_type"]) for row in eligible).items())
        ),
        "gold_chain_length_counts": dict(
            sorted(
                Counter(len(row["gold_evidence_ids"]) for row in eligible).items()
            )
        ),
        "supported_fact_count": sum(
            len(row["gold_supported_facts"]) for row in eligible
        ),
        "multi_citation_fact_count": sum(
            len(fact["gold_evidence_ids"]) > 1
            for row in eligible
            for fact in row["gold_supported_facts"]
        ),
        "drop_duplicate_questions": drop_duplicate_questions,
    }
    return eligible, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--preference-eval", type=Path, required=True)
    parser.add_argument("--recalibrated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-duplicate-questions", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace it: {args.output}")
    rows, audit = build_manifest_rows(
        tasks=read_json_rows(args.tasks),
        preferences=read_json_rows(args.preference_eval),
        recalibrated=read_json_rows(args.recalibrated),
        drop_duplicate_questions=not args.keep_duplicate_questions,
    )
    if not rows:
        raise SystemExit("no strict recalibrated retrieval rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit.update(
        {
            "inputs": {
                "tasks": str(args.tasks.resolve()),
                "tasks_sha256": sha256_file(args.tasks),
                "preference_eval": str(args.preference_eval.resolve()),
                "preference_eval_sha256": sha256_file(args.preference_eval),
                "recalibrated": str(args.recalibrated.resolve()),
                "recalibrated_sha256": sha256_file(args.recalibrated),
            },
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
        }
    )
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
