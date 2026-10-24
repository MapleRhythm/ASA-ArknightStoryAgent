"""Read-only CPU view of current native8B outputs; partial views are not rates."""
import argparse
import copy
import dataclasses
import json
import sys

from run_native8b_v1 import HERE, SOURCE, MATRIX, OUTPUT, VALIDATION, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--evidence", action="store_true")
    args = parser.parse_args()
    matrix = load(MATRIX)
    journal = OUTPUT / "predictions.progress.jsonl"
    blob = journal.read_bytes()
    blob = blob[:blob.rfind(b"\n") + 1]
    predictions = {r["id"]: r for r in map(json.loads, blob.splitlines())}
    current = {r["id"]: r for r in load(VALIDATION)["results"]} if VALIDATION.exists() else None
    baseline = {r["id"]: r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]}
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    for index in range(args.start, min(args.start + args.count, len(matrix["rows"]))):
        row = matrix["rows"][index]
        if row["id"] not in predictions:
            print(json.dumps({"index": index, "not_generated_yet": True}))
            continue
        prediction = predictions[row["id"]]
        original = baseline[row["parent_record_id"] + "::original_prompt_v1"]
        item = {"index": index, "question": row["question"], "evidence_sha256": row["evidence_sha256"],
                "original4b_shown": original.get("grounded"),
                "native8b_finish": prediction["finish_reason"],
                "full_validation_completed": current is not None}
        if current is not None:
            result = current[row["id"]]
            item.update(native8b_parsed=result.get("parsed"), native8b_shown=result.get("grounded"),
                        status=result["status"], error=result.get("error"))
        else:
            try:
                hyp = HypothesisDocument(**row["hypothesis"])
                last = row["current_round"] >= row["max_retrieval_rounds"]
                normalized = normalize_minimal_conclusion_output(prediction["raw_output"], "minimal")
                parsed = parse_conclusion_output(normalized, question=row["question"], current_hypothesis=hyp,
                                                 max_round_reached=last)
                grounded = validate_conclusion_grounding(
                    question=row["question"], hypothesis=hyp, evidence=copy.deepcopy(row["prompt_evidence"]),
                    conclusion=copy.deepcopy(parsed), max_round_reached=last, mode="quote",
                    evidence_prompt_text=row["evidence_brief"],
                )
                item.update(native8b_parsed=dataclasses.asdict(parsed), native8b_shown=dataclasses.asdict(grounded))
            except Exception as exc:
                item.update(error=f"{type(exc).__name__}: {exc}", native8b_raw_unshown=prediction["raw_output"])
        print(json.dumps(item, ensure_ascii=False))
        if args.evidence:
            print(row["evidence_brief"])


if __name__ == "__main__":
    main()
