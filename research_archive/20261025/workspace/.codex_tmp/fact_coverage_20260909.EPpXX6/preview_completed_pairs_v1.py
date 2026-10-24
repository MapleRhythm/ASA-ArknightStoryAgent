"""Read-only previews of completed generations through the frozen validator."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import sys

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    matrix = load(HERE / "fact_coverage_pairs116_v1.matrix.json")
    predictions = [
        json.loads(line) for line in
        (HERE / "paired116_ctx12288_v1/predictions.progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_id = {row["id"]: row for row in predictions}
    start, end = args.start * 2, (args.start + args.count) * 2
    for index in range(start, min(end, len(matrix["rows"])), 2):
        rows = matrix["rows"][index:index + 2]
        if not all(row["id"] in by_id for row in rows):
            print(json.dumps({"pair_index": index // 2, "status": "not_yet_complete"}))
            break
        print(json.dumps({
            "pair_index": index // 2, "question": rows[0]["question"],
            "parent_record_id": rows[0]["parent_record_id"],
        }, ensure_ascii=False))
        for row in rows:
            prediction = by_id[row["id"]]
            item = {"variant": row["renderer"], "finish": prediction["finish_reason"]}
            try:
                normalized = normalize_minimal_conclusion_output(prediction["raw_output"], "minimal")
                hypothesis = HypothesisDocument(**row["hypothesis"])
                final = row["current_round"] >= row["max_retrieval_rounds"]
                parsed = parse_conclusion_output(
                    normalized, question=row["question"], current_hypothesis=hypothesis,
                    max_round_reached=final,
                )
                grounded = validate_conclusion_grounding(
                    question=row["question"], hypothesis=hypothesis,
                    evidence=deepcopy(row["prompt_evidence"]), conclusion=deepcopy(parsed),
                    max_round_reached=final, mode="quote", evidence_prompt_text=row["evidence_brief"],
                )
                if args.compact:
                    item.update(action=grounded.next_action, shown_answer=grounded.answer)
                else:
                    item.update(parsed_answer=parsed.answer, shown=asdict(grounded))
            except Exception as exc:
                item.update(error=f"{type(exc).__name__}: {exc}", raw=prediction["raw_output"])
            print(json.dumps(item, ensure_ascii=False))
        if args.evidence:
            print(rows[0]["evidence_brief"])


if __name__ == "__main__":
    main()
