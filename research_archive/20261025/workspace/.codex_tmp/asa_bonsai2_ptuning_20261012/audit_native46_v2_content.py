"""Read-only aggregation of v2 evidence review and immutable historical scores."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("complete", "partial", "wrong", "no_answer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate(root=ROOT):
    review_path = root / "NATIVE46_V2_CONTENT_REVIEW_20261017.json"
    review = json.loads(review_path.read_text())
    run = root / review["run"]
    manifest = json.loads((run / "manifest.json").read_text())
    old_path = root / review["policy_source"]
    assert sha(old_path) == review["policy_source_sha256"]
    assert sha(old_path) == manifest["historical_content_review_sha256"]
    old = json.loads(old_path.read_text())
    tasks = load_rows(run / "tasks.jsonl")
    assert sha(run / "tasks.jsonl") == review["tasks_sha256"] == manifest["tasks_sha256"]
    baseline = root / "native46_4b_sftkto_vs_bonsai_pt_20261014"
    assert sha(baseline / "tasks.jsonl") == review["tasks_sha256"]
    assert sha(baseline / "manifest.json") == manifest["baseline_manifest_sha256"]
    for name, expected in manifest["historical_baseline_outputs"].items():
        assert sha(baseline / name) == expected, name
    outputs = []
    for name, expected in review["results_sha256"].items():
        assert sha(run / name) == expected, name
        outputs.extend(load_rows(run / name))
    outputs.sort(key=lambda r: r["position"])
    cases = review["cases"]
    assert len(tasks) == len(outputs) == len(cases) == len(old["cases"]) == 46
    assert len({t["id"] for t in tasks}) == 46
    assert review["semantic_adjudication_complete"] is True
    assert review["format_and_citation_binding_excluded"] is True
    for pos, (task, output, case, historical) in enumerate(
        zip(tasks, outputs, cases, old["cases"]), 1
    ):
        assert task["position"] == output["position"] == case["position"] == historical["position"] == pos
        for key in ("id", "question", "round", "question_group"):
            assert task[key] == output[key], (pos, key)
        assert output["prompt_sha256"] == task["bonsai_prompt_sha256"]
        assert output["input_truncated"] is False
        assert output["answer_validation"] is False
        assert output["evidence_count"] == len(task["documents"]) == 20
        assert output["output_reached_length_limit"] is False
        assert case["category"] in CATEGORIES
        assert type(case["core_complete"]) is bool
        assert case["category"] != "complete" or case["core_complete"]
        assert case["note"].strip()
        assert set(case["source_eids"]) <= {d["evidence_id"] for d in task["documents"]}
        if "borderline_alternative" in case:
            assert case["borderline_alternative"] in CATEGORIES
            assert case["borderline_alternative"] != case["category"]
    previous8 = set(json.loads((root / "heldout8_selection_20261014.json").read_text())["ids"])
    gap_positions = {c["position"] for c in old["cases"] if c.get("evidence_gap")}
    arms = {
        "fourb": [c["fourb"] for c in old["cases"]],
        "bonsai_v1": [c["bonsai"] for c in old["cases"]],
        "bonsai_v2": cases,
    }
    summary = {}
    for arm, decisions in arms.items():
        summary[arm] = {}
        for subset, indices in {
            "all46": list(range(46)),
            "previous8": [i for i, t in enumerate(tasks) if t["id"] in previous8],
            "new38": [i for i, t in enumerate(tasks) if t["id"] not in previous8],
            "evidence_gap6": [i for i, t in enumerate(tasks) if t["position"] in gap_positions],
        }.items():
            counts = Counter(decisions[i]["category"] for i in indices)
            summary[arm][subset] = {
                "n": len(indices),
                "categories": {k: counts[k] for k in CATEGORIES},
                "complete_rate": counts["complete"] / len(indices),
                "core_complete_diagnostic_not_accuracy": sum(decisions[i]["core_complete"] for i in indices),
            }
        groups = defaultdict(list)
        for task, decision in zip(tasks, decisions):
            groups[task["question_group"]].append(decision["category"] == "complete")
        summary[arm]["groups43_equal_weight_rate"] = sum(
            sum(values) / len(values) for values in groups.values()
        ) / len(groups)
        assert len(groups) == 43
    paired = {}
    for baseline_arm in ("fourb", "bonsai_v1"):
        categories = defaultdict(list)
        for pos, (v2, before) in enumerate(zip(cases, arms[baseline_arm]), 1):
            a, b = v2["category"] == "complete", before["category"] == "complete"
            key = "both" if a and b else "gained_v2" if a else "lost_v2" if b else "neither"
            categories[key].append(pos)
        paired[baseline_arm] = {
            key: {"count": len(categories[key]), "positions": categories[key]}
            for key in ("both", "gained_v2", "lost_v2", "neither")
        }
    correct = sum(c["category"] == "complete" for c in cases)
    lower = correct - sum(c["category"] == "complete" and c.get("borderline_alternative") == "partial" for c in cases)
    upper = correct + sum(c["category"] != "complete" and c.get("borderline_alternative") == "complete" for c in cases)
    return {
        "integrity_passed": True,
        "review_sha256": sha(review_path),
        "reviewed_v2": len(cases),
        "summary": summary,
        "paired": paired,
        "borderline_sensitivity_not_confidence_interval": {
            "lower_complete": lower, "upper_complete": upper,
            "lower_rate": lower / 46, "upper_rate": upper / 46,
            "positions": [c["position"] for c in cases if "borderline_alternative" in c],
        },
        "generation": {
            "n": len(outputs),
            "parsed_json": sum(r["parsed_output_for_offline_inspection_only"] is not None for r in outputs),
            "input_truncated": sum(r["input_truncated"] for r in outputs),
            "output_length_limit": sum(r["output_reached_length_limit"] for r in outputs),
        },
        "caution": "Assistant review; fixed historical development evidence, not independent human gold, blind generalization, full-chain retrieval or deployment acceptance. Baseline scores reused, not newly generated.",
    }


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
