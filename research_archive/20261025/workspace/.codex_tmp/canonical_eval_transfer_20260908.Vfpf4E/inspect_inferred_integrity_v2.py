"""Compare every replay action, visible answer, parsed payload and diagnosis."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
specs = [
    ("fixed79_configured_lora_ctx12288", "validation_quote_integrity_candidate_v1.json", 79),
    ("strict9_vllm_context_guard_ctx12288_stable_idle", "validation_quote_integrity_candidate_v1.json", 9),
    ("same_evidence_retry_paired68_ctx12288_v1", "validation_quote_integrity_v1.json", 68),
]
report = {"evaluation_only": True, "training_ready": False, "production_changed": False, "comparisons": []}
for folder, baseline_name, count in specs:
    baseline_path = run / folder / baseline_name
    candidate_path = run / folder / "validation_inferred_integrity_candidate_v2.json"
    v1_path = run / folder / "validation_inferred_integrity_candidate_v1.json"
    baseline, candidate, v1 = map(load, (baseline_path, candidate_path, v1_path))
    old = {r["id"]: r for r in baseline["results"]}
    new = {r["id"]: r for r in candidate["results"]}
    prior = {r["id"]: r for r in v1["results"]}
    assert len(old) == len(new) == len(prior) == count
    assert old.keys() == new.keys() == prior.keys()
    assert baseline["matrix_sha256"] == candidate["matrix_sha256"] == v1["matrix_sha256"]
    assert baseline["predictions_sha256"] == candidate["predictions_sha256"] == v1["predictions_sha256"]
    assert all(r["status"] == "ok" for r in candidate["results"])
    changes = []
    for key in old:
        before, after = old[key], new[key]
        assert before["raw_output"] == after["raw_output"]
        differences = {
            field: {"before": before["grounded"].get(field), "after": after["grounded"].get(field)}
            for field in set(before["grounded"]) | set(after["grounded"])
            if before["grounded"].get(field) != after["grounded"].get(field)
        }
        if differences:
            changes.append({"id": key, "question": before["question"], "grounded_differences": differences})
    comparison = {
        "folder": folder, "rows": count,
        "baseline_sha256": sha(baseline_path), "candidate_sha256": sha(candidate_path),
        "action_changed_ids": [k for k in old if old[k]["grounded"]["next_action"] != new[k]["grounded"]["next_action"]],
        "answer_changed_ids": [k for k in old if old[k]["grounded"]["answer"] != new[k]["grounded"]["answer"]],
        "parsed_changed_ids": [k for k in old if old[k]["parsed"] != new[k]["parsed"]],
        "grounded_changes": changes,
        "v1_v2_grounded_difference_ids": [k for k in new if new[k]["grounded"] != prior[k]["grounded"]],
        "after_summary": candidate["summary"],
    }
    report["comparisons"].append(comparison)
    print(json.dumps(comparison, ensure_ascii=False))
tests = run / "inferred_integrity_candidate_tests_v2.xml"
suites = list(ET.parse(tests).getroot().iter("testsuite"))
report["candidate_tests"] = sum(int(s.get("tests", 0)) for s in suites)
assert report["candidate_tests"] == 115
assert all(int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0 for s in suites)
old_regression = run / "inferred_integrity_v1_cross_pool_regression.xml"
regression_suites = list(ET.parse(old_regression).getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in regression_suites) == 1
assert sum(int(s.get("failures", 0)) for s in regression_suites) == 1
report["v1_cross_pool_regression_reproduced"] = True
report["test_receipts"] = {str(p.name): sha(p) for p in (tests, old_regression)}
report["limitations"] = [
    "Repeated frozen evidence states, not fresh retrieval or production E2E.",
    "Suppressing an unsupported answer is not producing another correct answer.",
    "Valid matching citations do not prove that a fact or inference follows logically.",
]
with (run / "inferred_integrity_v2_comparison.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
