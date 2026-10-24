import hashlib
import json
from pathlib import Path
from collections import Counter
from statistics import mean, median
import math
import re

ROOT = Path(__file__).resolve().parent


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    run = ROOT / "attempt2"
    rows = read_lines(run / "results.jsonl")
    calls = read_lines(run / "calls.jsonl")
    audit = read_lines(run / "audit.jsonl")
    reviews = json.loads((ROOT / "review.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    actual_sha = hashlib.sha256((run / "results.jsonl").read_bytes()).hexdigest()
    assert actual_sha == completion["results_sha256"]
    assert completion["questions_match"] and len(rows) == len(reviews) == 20
    requests = {(x["question_index"], x["call"]): x for x in calls if x["event"] == "request"}
    responses = [x for x in calls if x["event"] == "response"]
    engines = [x for x in calls if x["event"] == "engine_result"]
    snapshots = [x for x in audit if x["event"] == "prompt_evidence"]
    bodies_verified = 0
    for snapshot in snapshots:
        prompt = next(x["prompt"] for x in requests.values()
                      if hashlib.sha256(x["prompt"].encode()).hexdigest() == snapshot["prompt_sha256"])
        assert len(snapshot["documents"]) == 20 and all(x["full"] for x in snapshot["checks"])
        normalized = re.sub(r"\s+", " ", prompt)
        for document in snapshot["documents"]:
            # Original corpora have no internal field markers in these clean_texts.
            assert re.sub(r"\s+", " ", document["clean_text"]).strip() in normalized
            bodies_verified += 1
    for result in engines:
        assert result["actual_input_matches"]
        assert "exx_sft_v55_20260911" in result["lora_path"]
        assert result["input_tokens"] + result["max_tokens"] <= 10000
    assert len(requests) == len(responses) == len(engines)
    parsed = 0
    legacy_fields = []
    for result in responses:
        try:
            payload = json.loads(result["text"])
            parsed += isinstance(payload, dict)
            if isinstance(payload, dict) and "next_action" in payload:
                serialized = json.dumps(payload)
                for name in ("quote", "evidence_refs", "inferred_facts", "final_answer", "answer"):
                    if f'"{name}"' in serialized:
                        legacy_fields.append((result["question_index"], result["call"], name))
        except ValueError:
            pass
    per_question = []
    for index, (row, review) in enumerate(zip(rows, reviews), 1):
        assert review["q"] == index
        conclusion = row["retrieval_trace"][-1]["conclusion"]
        per_question.append({
            **review, "question": row["question"], "answer": row["answer"],
            "elapsed_sec": row["elapsed_sec"], "rounds": len(row["retrieval_trace"]),
            "last_raw": conclusion.get("generation_diagnostics", {}).get("raw_output"),
            "grounding_issues": conclusion.get("generation_diagnostics", {}).get("grounding_issues", []),
            "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
            "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest(),
            "final_action": conclusion["next_action"],
        })
    times = [r["elapsed_sec"] for r in rows]
    summary = {
        "results_sha256": actual_sha, "complete": True, "denominator": 20,
        "categories": dict(Counter(r["category"] for r in reviews)),
        "strict_correct": sum(r["category"] == "correct" for r in reviews),
        "usable": sum(r["usable"] for r in reviews),
        "last_raw_content_strict_correct": sum(r["last_raw_strict_correct"] for r in reviews),
        "false_rejected_strict_correct_last_raw_qids": [8, 13, 18],
        "latency": {"mean_sec": mean(times), "median_sec": median(times),
                    "p90_sec": sorted(times)[math.ceil(len(times)*.9)-1],
                    "warm_mean_sec": mean(times[1:]), "total_sec": sum(times)},
        "retrieval_rounds": dict(Counter(p["rounds"] for p in per_question)),
        "actions": dict(Counter(p["final_action"] for p in per_question)),
        "calls": len(engines), "raw_strict_json_objects": parsed, "legacy_fields": legacy_fields,
        "finish_reasons": dict(Counter(c["finish_reason"] for e in engines for c in e["choices"])),
        "min_input_tokens": min(e["input_tokens"] for e in engines),
        "max_input_tokens": max(e["input_tokens"] for e in engines),
        "max_output_tokens": max(c["tokens"] for e in engines for c in e["choices"]),
        "input_body_checks": bodies_verified, "evidence_prompts": len(snapshots),
        "runtime_errors": sum(bool(r.get("error")) for r in rows),
        "completion": completion, "per_question": per_question,
    }
    with (ROOT / "summary.json").open("x") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k != "per_question"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
