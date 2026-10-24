"""Canonical quote replay with explicit per-row round and complete denominators.

Run beside frozen before/after sources on 241. Does not simulate retrieval.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_rows(matrix, predictions):
    rows = matrix["rows"]
    by_id = {row["id"]: row for row in rows}
    pred_ids = [row["id"] for row in predictions]
    if (len(rows) != len(by_id) or len(pred_ids) != len(set(pred_ids))
            or set(pred_ids) != set(by_id)):
        raise ValueError("missing or duplicate prediction/matrix rows")
    for pred in predictions:
        row = by_id[pred["id"]]
        if pred["actual_prompt"] != row["runtime_prompt"]:
            raise ValueError("actual input differs from frozen runtime prompt")
        digest = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        if pred["input_row_sha256"] != digest:
            raise ValueError("prediction input row hash mismatch")
        yield row, pred


def round_state(row):
    current = row.get("current_round", 1)
    maximum = row.get("max_retrieval_rounds", 2)
    if (type(current) is not int or type(maximum) is not int
            or not 1 <= current <= maximum):
        raise ValueError("invalid round metadata")
    return current, maximum


def summarize(results):
    summary = {}
    for renderer in sorted({row["renderer"] for row in results}):
        subset = [row for row in results if row["renderer"] == renderer]
        summary[renderer] = {
            "rows": len(subset),
            "normalized_strict_json": sum(r["normalized_strict_json"] for r in subset),
            "parse_or_validation_errors": sum(r["status"] != "ok" for r in subset),
            "raw_actions": dict(Counter(r.get("parsed", {}).get("next_action", "error") for r in subset)),
            "grounded_actions": dict(Counter(r.get("grounded", {}).get("next_action", "error") for r in subset)),
            "length_outputs": sum(r["finish_reason"] == "length" for r in subset),
            "actual_total_exceeds_config10000": sum(r["actual_total_tokens"] > 10000 for r in subset),
            "rounds": dict(Counter(f'{r["current_round"]}/{r["max_retrieval_rounds"]}' for r in subset)),
            "mean_generation_seconds": sum(r["generation_seconds"] for r in subset) / len(subset),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", choices=("before", "after"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--grounding-mode", choices=("quote", "evidence_id"), default="quote")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new artifact")
    run = Path(__file__).resolve().parent
    source_root = args.source_root or run / args.validator / "src"
    sys.path.insert(0, str(source_root))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

    matrix = json.loads(args.matrix.read_text())
    predictions = json.loads(args.predictions.read_text())
    results = []
    for row, pred in matched_rows(matrix, predictions):
        current, maximum = round_state(row)
        final = current >= maximum
        item = {key: row[key] for key in (
            "id", "source_id", "question", "renderer", "template", "evidence_brief",
            "evidence_id_to_corpus_id", "packing_trace",
        )}
        item.update({
            "current_round": current, "max_retrieval_rounds": maximum,
            "round_metadata_defaulted": "current_round" not in row,
            "raw_output": pred["raw_output"], "finish_reason": pred["finish_reason"],
            "input_tokens": pred["prompt_token_count"],
            "generated_tokens": pred["generated_token_count"],
            "actual_total_tokens": pred["prompt_token_count"] + pred["generated_token_count"],
            "generation_seconds": pred["generation_seconds"],
            "normalized_strict_json": False,
        })
        stage = "normalization"
        try:
            normalized = normalize_minimal_conclusion_output(pred["raw_output"], "minimal")
            item["normalized_output"] = normalized
            try:
                item["normalized_strict_json"] = isinstance(json.loads(normalized), dict)
            except json.JSONDecodeError:
                pass
            hyp = HypothesisDocument(**row["hypothesis"])
            stage = "parsing"
            parsed = parse_conclusion_output(
                normalized, question=row["question"], current_hypothesis=hyp,
                max_round_reached=final,
            )
            item["parsed"] = dataclasses.asdict(parsed)
            stage = "grounding"
            grounded = validate_conclusion_grounding(
                question=row["question"], hypothesis=hyp, evidence=copy.deepcopy(row["prompt_evidence"]),
                conclusion=copy.deepcopy(parsed), max_round_reached=final,
                mode=args.grounding_mode, evidence_prompt_text=row["evidence_brief"],
            )
            item["grounded"] = dataclasses.asdict(grounded)
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "parse_or_validation_error"
            item["error_stage"] = stage
            item["error"] = f"{type(exc).__name__}: {exc}"
        if item["status"] == "ok":
            try:
                counterfactual = validate_conclusion_grounding(
                    question=row["question"], hypothesis=hyp,
                    evidence=copy.deepcopy(row["prompt_evidence"]),
                    conclusion=copy.deepcopy(parsed), max_round_reached=True,
                    mode=args.grounding_mode, evidence_prompt_text=row["evidence_brief"],
                )
                item["last_round_validation_counterfactual"] = dataclasses.asdict(counterfactual)
            except Exception as exc:
                item["counterfactual_error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    report = {
        "version": "canonical_grounding_analysis_v4", "evaluation_only": True, "training_ready": False,
        "grounding_mode": args.grounding_mode,
        "runtime_source_root": str(source_root),
        "payload_utils_sha256": sha(source_root / "asa_arknight_story_agent/inference/payload/utils.py"),
        "validator": args.validator, "matrix_sha256": sha(args.matrix),
        "predictions_sha256": sha(args.predictions), "source_manifest_sha256": sha(run / "manifest.json"),
        "analyzer_sha256": sha(__file__), "summary": summarize(results), "results": results,
        "limitations": [
            "Mechanical quote/schema validation is not semantic answer accuracy.",
            "All rows, including normalization/parse/validation failures, remain in the denominator.",
            "Round metadata comes from the frozen input; missing strict9 metadata defaults to 1/2.",
            "Last-round validation is counterfactual, not another retrieval or regeneration.",
            "No production orchestrator recovery/final direct-answer fallback was executed.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"validator": args.validator, "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
