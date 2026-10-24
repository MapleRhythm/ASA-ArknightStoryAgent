"""Verify saved requests and separately document semantic-review boundaries."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_answer_slots_v1 as old
import probe_answer_slots_v2 as new


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    report_path = HERE / "answer_slot_schema_pairs14_v2.json"
    report = load(report_path)
    output = HERE / "answer_slot_schema_pairs14_postflight_v2.json"
    assert not output.exists()
    source = HERE / "answer_slot_coverage_direct39_v1.json"
    original = load(source)
    assert sha(source) == report["source_sha256"]
    assert sha(HERE / "run_slot_schema_pairs14_v2.py") == report["driver_sha256"]
    assert sha(HERE.parent / "probe_claim_entailment_v1.py") == report["transport_sha256"]
    assert sha(HERE / "probe_answer_slots_v1.py") == report["rubric_sha256"]["retry_v1"]
    assert sha(HERE / "probe_answer_slots_v2.py") == report["rubric_sha256"]["schema_clarified_v2"]
    assert sha(HERE / "answer_slot_schema_pairs14_preflight_v2.json") == report["preflight_sha256"]
    assert new.RUBRIC.replace(new.NEW_CLAUSE, new.OLD_CLAUSE) == old.RUBRIC
    failed = {r["id"]: r for r in original["results"] if r["status"] == "error"}
    assert len(failed) == 7 and len(original["results"]) == 39
    results = {r["id"]: r for r in report["results"]}
    assert len(results) == len(report["cases"]) == 14
    pairs = defaultdict(list)
    for case in report["cases"]:
        parent = failed[case["parent_id"]]
        assert case["question"] == parent["source_case_question"]
        assert case["shown_answer"] == parent["source_case_shown_answer"]
        module = old if case["condition"] == "retry_v1" else new
        result = results[case["id"]]
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": module.messages(case)}
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == old.transport.digest(expected)
        assert result["response_model"] == "glm-5.3" and result["finish_reason"] == "stop"
        if result["status"] == "ok":
            assert old.validate(json.loads(result["raw_response"]), case) == result["judgement"]
        else:
            try:
                old.validate(json.loads(result["raw_response"]), case)
            except ValueError as exc:
                assert result["error"] == f"ValueError: {exc}"
            else:
                raise AssertionError("Saved failure no longer reproduces")
        pairs[case["parent_id"]].append(case)
    assert len(pairs) == 7 and all(len(pair) == 2 for pair in pairs.values())
    assert sum(pair[0]["condition"] == "retry_v1" for pair in pairs.values()) == 4
    initial_failure_types = Counter()
    omitted_parents = []
    for parent_id, row in failed.items():
        raw = json.loads(row["raw_response"])
        omitted = [s for s in raw["slots"] if s["status"] == "unfilled" and "answer_spans" not in s]
        if omitted:
            initial_failure_types["unfilled_answer_spans_omitted"] += 1
            omitted_parents.append(parent_id)
        elif "verbatim" in row["error"]:
            initial_failure_types["nonverbatim_answer_span"] += 1
        else:
            assert any(set(s) - {"requirement", "status", "answer_spans", "reason"} for s in raw["slots"])
            initial_failure_types["extra_slot_field"] += 1
    assert dict(initial_failure_types) == {
        "unfilled_answer_spans_omitted": 3, "nonverbatim_answer_span": 3, "extra_slot_field": 1,
    }
    assert all(results[parent + "::schema_clarified_v2"]["status"] == "ok" for parent in omitted_parents)
    assert all(results[parent + "::retry_v1"]["status"] == "error" for parent in omitted_parents)
    notes = {
        "26ec3b7b11e9fcd4": "None is supported as coverage: the shown answer does not identify the disease and does not connect it to the city's history.",
        "466b0df26a5d22ba": "Both requests still changed curly single quotes to ASCII quotes in an answer span. Preserve the validation failure; do not repair the saved JSON and count it successful.",
        "54db38760f359834": "Both conditions now pass and identify the requested definition. Ordinary retry also recovered it, so do not attribute this case to the clause fix.",
        "b9b9339720803699": "The missing relationship to the parents is correctly identified as unfilled. Reasons offered for the emotional motive remain broad; coverage does not certify the story claims.",
        "relabel-2f5d1fc38846acbac8802a43": "Semantic concern remains: the judge insists on the superior's deeper motive and dismisses the superior's order as no reason for a subordinate's action. The question does not explicitly require that deeper motive. Schema success must not become an automatic rejection gate.",
        "relabel-3c4ca0be007bfbbb7dfde508": "Both requests still replace Chinese quotation marks in answer_spans. The answer contains a specific identity, so a coverage-only complete opinion would not establish that the identity belongs to this character; source attribution is a separate known concern.",
        "relabel-77becf1fa96953cb40e226f6": "Both conditions pass the concrete how-information. The extra-field failure also disappears under ordinary retry; no evidence that the one-clause change caused this recovery.",
    }
    reviewed = []
    for parent in pairs:
        matches = [key for key in notes if parent.startswith(key)]
        assert len(matches) == 1
        reviewed.append({"parent_id": parent, "reviewer_note": notes[matches[0]]})
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "pairs": 7, "requests": 14,
        "protocol_verified": True, "old39_file_unchanged": True,
        "only_predeclared_clause_changed": True, "all_failures_reproduced": True,
        "initial_failure_types": dict(initial_failure_types),
        "source_sha256": sha(source), "report_sha256": sha(report_path),
        "summary": report["summary"], "reviewer": "independent_assistant_not_human_gold",
        "reviewer_notes": reviewed,
        "limitations": [
            "No old JSON repaired and no mixed-rubric replacement score over the 39 states.",
            "All seven originally failed inputs are development diagnostics, not a held-out judge-accuracy set.",
            "Three omitted-field failures recovered only in the clarified condition in this trial; no provider determinism assumption.",
            "The remaining semantic scope concern and quote-copying failures prevent production-gate promotion.",
        ],
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"postflight": str(output), "summary": value["summary"], "old39_unchanged": True}))


if __name__ == "__main__":
    main()
