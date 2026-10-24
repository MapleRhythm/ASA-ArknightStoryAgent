#!/usr/bin/env python3
"""Content-bound mechanical quote audit; never substitutes for semantic review."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_rendered_evidence_visibility import (  # noqa: E402
    hypothesis_from_snapshot, read_rows, rerank_candidates, select_items,
    sha256_file, text_sha,
)
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt  # noqa: E402
from asa_arknight_story_agent.inference.generation.conclusion_generation import (  # noqa: E402
    normalize_minimal_conclusion_output, parse_conclusion_output,
)
from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes  # noqa: E402
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding  # noqa: E402
from asa_arknight_story_agent.inference.model_runtime.output_cleaning import sanitize_generation_output  # noqa: E402


def analyze_row(row: dict[str, Any], prediction: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    if prediction["input_row_sha256"] != text_sha(json.dumps(row, ensure_ascii=False, sort_keys=True)):
        raise ValueError("prediction input row hash mismatch")
    if prediction["actual_prompt_sha256"] != text_sha(prediction["actual_prompt"]):
        raise ValueError("actual prompt hash mismatch")
    if (row["system"] not in prediction["actual_prompt"]
            or row["user"] not in prediction["actual_prompt"]):
        raise ValueError("actual generation prompt does not contain frozen messages")
    if row["template"] == "runtime_raw" and prediction["actual_prompt"] != row["runtime_prompt"]:
        raise ValueError("runtime_raw prompt mismatch")
    raw = sanitize_generation_output(prediction["raw_output"], prediction["actual_prompt"])
    normalized = normalize_minimal_conclusion_output(raw, "minimal")
    result: dict[str, Any] = {
        "id": row["id"], "source_id": row["source_id"], "question": row["question"],
        "renderer": row["renderer"], "template": row["template"],
        "finish_reason": prediction["finish_reason"],
        "prompt_token_count": prediction["prompt_token_count"],
        "generated_token_count": prediction["generated_token_count"],
        "generation_seconds": prediction["generation_seconds"],
        "raw_output": prediction["raw_output"], "normalized_output": normalized,
        "semantic_review": "pending",
    }
    try:
        payload = json.loads(normalized)
        result["normalized_strict_object"] = isinstance(payload, dict)
        result["raw_action"] = payload.get("next_action") if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        result["normalized_strict_object"] = False
        result["raw_action"] = None
    hypothesis = hypothesis_from_snapshot(snapshot)
    evidence = select_items(snapshot, rerank_candidates(snapshot), mode="quote", strategy="production", top_k=10)
    try:
        conclusion = parse_conclusion_output(
            normalized, question=row["question"], current_hypothesis=hypothesis, max_round_reached=False,
        )
    except Exception as exc:
        result["parse_error"] = str(exc)
        return result
    result["parsed"] = asdict(conclusion)
    if conclusion.next_action == "answer_directly":
        issues, warnings = validate_grounded_quotes(
            conclusion=conclusion, evidence=evidence, question=row["question"],
            evidence_prompt_text=row["evidence_brief"],
        )
        result["quote_issues"], result["quote_warnings"] = issues, warnings
        exact_quotes = []
        mapping = evidence_id_text_map_from_prompt(row["evidence_brief"])
        for fact in conclusion.supported_facts:
            for ref in fact.get("evidence_refs") or []:
                quote, eid = str(ref.get("quote") or ""), str(ref.get("evidence_id") or "")
                exact_quotes.append({
                    "evidence_id": eid, "quote": quote,
                    "literal_visible": bool(quote and eid in mapping and quote in mapping[eid]),
                })
        result["literal_quotes"] = exact_quotes
    validated = validate_conclusion_grounding(
        question=row["question"], hypothesis=hypothesis, evidence=evidence,
        conclusion=copy.deepcopy(conclusion), max_round_reached=False, mode="quote",
        evidence_prompt_text=row["evidence_brief"],
    )
    result["validated"] = asdict(validated)
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f'{row["template"]}/{row["renderer"]}'].append(row)
    output = {}
    for name, values in sorted(groups.items()):
        output[name] = {
            "rows": len(values),
            "normalized_strict_objects": sum(value["normalized_strict_object"] for value in values),
            "parse_errors": sum("parse_error" in value for value in values),
            "length_rows": sum(value["finish_reason"] == "length" for value in values),
            "raw_actions": dict(Counter(value["raw_action"] for value in values)),
            "validated_actions": dict(Counter(value.get("validated", {}).get("next_action", "parse_error") for value in values)),
            "mean_prompt_tokens": sum(value["prompt_token_count"] for value in values) / len(values),
            "mean_generated_tokens": sum(value["generated_token_count"] for value in values) / len(values),
            "mean_generation_seconds": sum(value["generation_seconds"] for value in values) / len(values),
            "answer_rows_with_quote_issues": sum(bool(value.get("quote_issues")) for value in values),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite analysis")
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    predictions = read_rows(args.predictions)
    snapshots = {row["id"]: row for row in read_rows(args.snapshots)}
    if sha256_file(args.snapshots) != matrix["source_snapshot_sha256"]:
        parser.error("snapshot hash mismatch")
    by_id = {row["id"]: row for row in matrix["rows"]}
    prediction_ids = {row["id"] for row in predictions}
    if (len(by_id) != len(matrix["rows"]) or len(prediction_ids) != len(predictions)
            or prediction_ids != by_id.keys()):
        parser.error("need exact, unique prediction/matrix ID sets")
    rows = [analyze_row(by_id[item["id"]], item, snapshots[item["source_id"]]) for item in predictions]
    report = {
        "version": "asa_quote_render_mechanical_audit_v1",
        "matrix_sha256": sha256_file(args.matrix),
        "predictions_sha256": sha256_file(args.predictions),
        "analyzer_sha256": sha256_file(Path(__file__)),
        "evaluation_only": True, "semantic_review": "pending",
        "notes": [
            "Current parsing and quote validation, not independent semantic judgment.",
            "First round only; no follow-up retrieval or recovery generation is performed.",
            "Literal quotes are reported separately from runtime fuzzy quote acceptance.",
            "The renderer-only intervention preserves selected evidence objects for downstream validation.",
        ],
        "summary": summarize(rows), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
