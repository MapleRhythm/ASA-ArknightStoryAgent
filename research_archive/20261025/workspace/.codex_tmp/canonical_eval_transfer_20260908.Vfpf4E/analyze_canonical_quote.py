"""Replay real quote normalization/validation, keeping raw and final answers apart."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", choices=("before", "after"), required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / args.validator / "src"))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

    matrix = json.loads((run / "paired.matrix.json").read_text())
    by_id = {row["id"]: row for row in matrix["rows"]}
    predictions = json.loads(args.predictions.read_text())
    if len(predictions) != len(by_id) or {row["id"] for row in predictions} != by_id.keys():
        raise ValueError("missing or duplicate prediction rows")
    results = []
    for pred in predictions:
        row = by_id[pred["id"]]
        if pred["actual_prompt"] != row["runtime_prompt"]:
            raise ValueError("actual input differs from frozen runtime prompt")
        if pred["input_row_sha256"] != hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest():
            raise ValueError("prediction input row hash mismatch")
        item = {key: row[key] for key in
                ("id", "source_id", "question", "renderer", "template", "evidence_brief",
                 "evidence_id_to_corpus_id", "packing_trace")}
        item.update({
            "raw_output": pred["raw_output"], "finish_reason": pred["finish_reason"],
            "input_tokens": pred["prompt_token_count"],
            "generated_tokens": pred["generated_token_count"],
            "actual_total_tokens": pred["prompt_token_count"] + pred["generated_token_count"],
            "generation_seconds": pred["generation_seconds"],
        })
        normalized = normalize_minimal_conclusion_output(pred["raw_output"], "minimal")
        item["normalized_output"] = normalized
        hyp = HypothesisDocument(**row["hypothesis"])
        try:
            payload = json.loads(normalized)
            item["normalized_strict_json"] = isinstance(payload, dict)
        except json.JSONDecodeError:
            item["normalized_strict_json"] = False
        try:
            parsed = parse_conclusion_output(
                normalized, question=row["question"], current_hypothesis=hyp,
                max_round_reached=False,
            )
            item["parsed"] = dataclasses.asdict(parsed)
            grounded = validate_conclusion_grounding(
                question=row["question"], hypothesis=hyp, evidence=copy.deepcopy(row["prompt_evidence"]),
                conclusion=copy.deepcopy(parsed), max_round_reached=False,
                mode="quote", evidence_prompt_text=row["evidence_brief"],
            )
            item["grounded"] = dataclasses.asdict(grounded)
            final_round = validate_conclusion_grounding(
                question=row["question"], hypothesis=hyp, evidence=copy.deepcopy(row["prompt_evidence"]),
                conclusion=copy.deepcopy(parsed), max_round_reached=True,
                mode="quote", evidence_prompt_text=row["evidence_brief"],
            )
            item["last_round_validation_counterfactual"] = dataclasses.asdict(final_round)
            item["status"] = "ok"
        except Exception as exc:
            item["status"] = "parse_or_validation_error"
            item["error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    summary = {}
    for renderer in ("before", "after"):
        subset = [row for row in results if row["renderer"] == renderer]
        summary[renderer] = {
            "rows": len(subset), "normalized_strict_json": sum(r["normalized_strict_json"] for r in subset),
            "parse_or_validation_errors": sum(r["status"] != "ok" for r in subset),
            "raw_actions": dict(Counter(r.get("parsed", {}).get("next_action", "error") for r in subset)),
            "grounded_actions": dict(Counter(r.get("grounded", {}).get("next_action", "error") for r in subset)),
            "length_outputs": sum(r["finish_reason"] == "length" for r in subset),
            "actual_total_exceeds_config10000": sum(r["actual_total_tokens"] > 10000 for r in subset),
            "mean_generation_seconds": sum(r["generation_seconds"] for r in subset) / len(subset),
        }
    report = {
        "evaluation_only": True, "training_ready": False,
        "validator": args.validator, "matrix_sha256": sha(run / "paired.matrix.json"),
        "predictions_sha256": sha(args.predictions), "source_manifest_sha256": sha(run / "manifest.json"),
        "analyzer_sha256": sha(__file__), "summary": summary, "results": results,
        "limitations": [
            "Mechanical quote/schema validation is not semantic answer accuracy.",
            "Last-round validation is a counterfactual on first-round generated output.",
            "No second retrieval or final direct-answer fallback was executed.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"validator": args.validator, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
