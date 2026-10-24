#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from asa_arknight_story_agent.inference.grounding.quote_checks import (
    validate_quote_grounding,
)
from asa_arknight_story_agent.inference.payload.normalization import (
    normalize_conclusion_payload,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_payload(diagnostics: dict) -> dict | None:
    for field in ("normalized_output", "raw_output"):
        text = str(diagnostics.get(field) or "").strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def hypothesis_from_row(row: dict) -> HypothesisDocument:
    value = row.get("hypothesis") or {}
    return HypothesisDocument(
        question=str(value.get("question") or row["question"]),
        intent=str(value.get("intent") or "plot_fact"),
        query_type=str(value.get("query_type") or "fact"),
        entities=list(value.get("entities") or []),
        keywords=list(value.get("keywords") or []),
        expected_answer_type=str(value.get("expected_answer_type") or "事实"),
        dialogue_context=str(value.get("dialogue_context") or ""),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--visible-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    result_rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    visible_rows = json.loads(
        args.visible_analysis.read_text(encoding="utf-8")
    )["results"]
    if len(result_rows) != len(visible_rows):
        raise ValueError("row count mismatch")

    replayed = []
    for index, (row, visible) in enumerate(
        zip(result_rows, visible_rows, strict=True),
        start=1,
    ):
        if row["question"] != visible["question"]:
            raise ValueError(f"question mismatch at row {index}")
        trace = row.get("retrieval_trace") or []
        conclusion = (trace[-1].get("conclusion") or {}) if trace else {}
        diagnostics = conclusion.get("generation_diagnostics") or {}
        payload = parse_payload(diagnostics)
        record = {
            "index": index,
            "id": visible["id"],
            "question": row["question"],
            "original_action": conclusion.get("next_action"),
            "original_grounding_status": diagnostics.get("grounding_status"),
            "status": "not_replayed",
        }
        if payload is None:
            replayed.append(record)
            continue
        try:
            normalized = normalize_conclusion_payload(
                payload,
                question=row["question"],
                dialogue_context=hypothesis_from_row(row).dialogue_context,
                current_intent=hypothesis_from_row(row).intent,
                current_hypothesis=hypothesis_from_row(row),
                max_round_reached=True,
            )
            checked = validate_quote_grounding(
                question=row["question"],
                hypothesis=hypothesis_from_row(row),
                evidence=[],
                conclusion=normalized,
                max_round_reached=True,
                evidence_prompt_text=visible["evidence_brief"],
            )
            final = checked or normalized
            record.update(
                {
                    "status": "ok",
                    "candidate_action": final.next_action,
                    "candidate_answer": final.answer,
                    "candidate_grounding_status": final.generation_diagnostics.get(
                        "grounding_status",
                        "accepted" if final.next_action == "answer_directly" else final.next_action,
                    ),
                    "candidate_grounding_warnings": final.grounding_warnings,
                    "candidate_supported_facts": final.supported_facts,
                    "grounding_repairs": final.generation_diagnostics.get(
                        "grounding_repairs",
                        [],
                    ),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        replayed.append(record)

    changed = [
        row
        for row in replayed
        if row.get("status") == "ok"
        and row.get("candidate_action") != row.get("original_action")
    ]
    report = {
        "version": "quote_unique_visible_rebinding_replay_v1",
        "evaluation_only": True,
        "training_ready": False,
        "source_results_sha256": sha256(args.results),
        "source_visible_analysis_sha256": sha256(args.visible_analysis),
        "rows": len(replayed),
        "statuses": dict(Counter(row["status"] for row in replayed)),
        "changed_actions": len(changed),
        "changed_ids": [row["id"] for row in changed],
        "boundaries": [
            "Replays existing raw final-round outputs; it does not regenerate or retrieve.",
            "Only quote grounding is replayed against exact final-round visible evidence.",
            "A newly accepted answer still requires content review before deployment.",
        ],
        "results": replayed,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": report["rows"],
                "statuses": report["statuses"],
                "changed_actions": report["changed_actions"],
                "changed_ids": report["changed_ids"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
