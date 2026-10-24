"""Recompute metrics from all 40 immutable results plus explicit semantic reviews."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from protocol_helpers import sha

HERE = Path(__file__).resolve().parent
CONDITIONS = ["base_off", "base_on", "lora_off", "lora_on"]
GRADES = {"correct", "partial", "wrong", "abstain", "failure"}


def main():
    results_path = HERE / "attempt1/results.jsonl"
    results = [json.loads(x) for x in results_path.read_text().splitlines()]
    reviews = json.loads((HERE / "semantic_review.json").read_text())
    assert reviews["complete"] is True
    assert len(results) == len(reviews["rows"]) == 40
    lookup = {(x["question_id"], x["condition"]): x for x in reviews["rows"]}
    assert len(lookup) == 40
    by_condition = {}
    paired_rows = []
    for condition in CONDITIONS:
        subset = [x for x in results if x["condition"] == condition]
        assert len(subset) == 10
        raw_correct = []
        delivered_correct = []
        for row in subset:
            key = (row["question_id"], condition)
            review = lookup[key]
            assert review["raw_grade"] in GRADES and review["delivered_grade"] in GRADES
            if row["status"] not in {"ok", "protocol_error"}:
                assert review["raw_grade"] == review["delivered_grade"] == "failure"
            if review["delivered_grade"] == "correct":
                assert review["raw_grade"] == "correct"
                assert row["status"] == "ok"
                assert row["grounded"]["answer"].strip()
                assert row["grounded"]["next_action"] in {"answer_directly", "abstain"}
            if review["raw_grade"] == "correct":
                raw_correct.append(row["question_id"])
            if review["delivered_grade"] == "correct":
                delivered_correct.append(row["question_id"])
        seconds = [x["elapsed_sec"] for x in subset]
        tokens = [x.get("output_tokens", 0) for x in subset]
        by_condition[condition] = {
            "n": 10, "raw_strict_correct": len(raw_correct),
            "raw_strict_accuracy": len(raw_correct) / 10,
            "raw_correct_question_ids": raw_correct,
            "delivered_strict_correct": len(delivered_correct),
            "delivered_strict_accuracy": len(delivered_correct) / 10,
            "delivered_correct_question_ids": delivered_correct,
            "raw_grades": Counter(lookup[(x["question_id"], condition)]["raw_grade"] for x in subset),
            "delivered_grades": Counter(lookup[(x["question_id"], condition)]["delivered_grade"] for x in subset),
            "runtime_status": Counter(x["status"] for x in subset),
            "finish_reason": Counter(x.get("finish_reason", "error") for x in subset),
            "normalized_action": Counter(x.get("normalized", {}).get("next_action", "error") for x in subset),
            "grounded_action": Counter(x.get("grounded", {}).get("next_action", "error") for x in subset),
            "mean_seconds": statistics.mean(seconds), "median_seconds": statistics.median(seconds),
            "p90_seconds": sorted(seconds)[math.ceil(.9 * len(seconds)) - 1],
            "total_seconds": sum(seconds), "mean_output_tokens": statistics.mean(tokens),
            "max_output_tokens": max(tokens), "total_output_tokens": sum(tokens),
        }
    for qid in sorted({x["question_id"] for x in results}):
        group = [x for x in results if x["question_id"] == qid]
        paired_rows.append({
            "question_id": qid, "question": group[0]["question"],
            "conditions": {
                x["condition"]: {
                    **lookup[(qid, x["condition"])],
                    "raw_final_text": x.get("final_text", ""),
                    "raw_answer": x.get("normalized", {}).get(
                        "answer", lookup[(qid, x["condition"])].get("raw_final_answer", "")),
                    "delivered_answer": x.get("grounded", {}).get("answer", ""),
                    "raw_action": x.get("normalized", {}).get("next_action"),
                    "delivered_action": x.get("grounded", {}).get("next_action"),
                    "grounding_warnings": x.get("grounded", {}).get("grounding_warnings", []),
                    "missing_slots": x.get("grounded", {}).get("missing_slots", []),
                    "status": x["status"], "seconds": x["elapsed_sec"],
                    "output_tokens": x.get("output_tokens"),
                } for x in group
            },
        })
    pairs = {}
    for model in ("base", "lora"):
        pairs[model] = {}
        for layer in ("raw", "delivered"):
            off = set(by_condition[f"{model}_off"][f"{layer}_correct_question_ids"])
            on = set(by_condition[f"{model}_on"][f"{layer}_correct_question_ids"])
            pairs[model][layer] = {
                "both_correct": sorted(on & off), "thinking_only_correct": sorted(on - off),
                "nonthinking_only_correct": sorted(off - on),
                "net_correct_gain": len(on) - len(off),
            }
    summary = {
        "mode": "fixed_evidence_reader_factorial_not_e2e",
        "question_count": 10, "generation_count": 40,
        "results_sha256": sha(results_path),
        "review_sha256": sha(HERE / "semantic_review.json"),
        "preflight_sha256": sha(HERE / "preflight.json"),
        "boundary_cases": reviews.get("boundary_cases", []),
        "conditions": by_condition, "thinking_pairs": pairs, "questions": paired_rows,
        "all_inputs_exact": all(x["input_token_ids_exact_match"] for x in results),
        "all_lora_requests_correct": all(x["returned_lora_id"] == x["requested_lora_id"] for x in results),
        "input_tokens_min": min(x["input_tokens"] for x in results),
        "input_tokens_max": max(x["input_tokens"] for x in results),
        "full_evidence_body_checks": 40 * 20,
        "notes": [
            "Known-error-enriched development subset; no claim of online accuracy.",
            "Raw reasoning is not scored; only completed final output.",
            "Same native template, tokenizer, sampling and output budget for all four arms.",
            "Not the historical production low-temperature/brace-prefill control.",
            "Assistant source review, not independent human gold; one seed per question.",
        ],
    }
    with (HERE / "summary.json").open("x") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(by_condition, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
