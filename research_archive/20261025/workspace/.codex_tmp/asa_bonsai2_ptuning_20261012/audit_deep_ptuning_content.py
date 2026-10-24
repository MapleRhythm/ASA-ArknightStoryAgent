"""Read-only identity checks and aggregation; never changes outputs or promotes data."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from audit_native46_v2_content import aggregate as historical_audit

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("complete", "partial", "wrong", "no_answer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_documents(task, prompt=None):
    assert len(task["documents"]) == 20
    assert {d["evidence_id"] for d in task["documents"]} == {f"E{i}" for i in range(1, 21)}
    for doc in task["documents"]:
        assert hashlib.sha256(doc["text"].encode()).hexdigest() == doc["sha256"]
        if prompt is not None:
            assert doc["text"] in prompt


def structure_valid(value):
    if not isinstance(value, dict) or value.get("next_action") != "answer_directly":
        return False
    if set(value) != {"next_action", "supported_facts"}:
        return False
    facts = value["supported_facts"]
    return (
        isinstance(facts, list) and 1 <= len(facts) <= 8
        and all(isinstance(f, dict) and set(f) == {"fact", "evidence_ids"}
                and isinstance(f["fact"], str) and f["fact"].strip()
                and isinstance(f["evidence_ids"], list) and len(f["evidence_ids"]) > 0
                and all(e in {f"E{i}" for i in range(1, 21)} for e in f["evidence_ids"])
                for f in facts)
    )


def aggregate(root=ROOT):
    previous_audit = historical_audit(root)
    assert previous_audit["integrity_passed"]
    review_path = root / "DEEP_PTUNING_CONTENT_REVIEW_20261017.json"
    review = read_json(review_path)
    run = root / review["run"]
    for path, key in (
        (root / review["tasks"], "tasks_sha256"),
        (run / "generated_native46.jsonl", "output_sha256"),
        (run / "generation_receipt.json", "generation_receipt_sha256"),
        (run / "run.json", "run_sha256"),
        (root / review["policy_source"], "policy_source_sha256"),
        (root / review["shallow750_review"], "shallow750_review_sha256"),
        (root / "heldout8_selection_20261014.json", "heldout8_selection_sha256"),
    ):
        assert sha(path) == review[key], path
    receipt = read_json(run / "generation_receipt.json")
    assert receipt["completed"] and receipt["count"] == 46
    assert receipt["output_sha256"] == review["output_sha256"]
    assert receipt["no_input_truncation"] and not receipt["answer_validation_enabled"]
    identity = read_json(run / "run.json")
    assert identity["training_complete"] and identity["optimizer_steps"] == 750
    assert identity["train_examples"] == review["data_examples"] == 750
    assert identity["method"] == review["method"]
    assert identity["backbone_frozen"] and identity["backbone_bytewise_unchanged"]
    assert identity["no_truncation"] and identity["full_evidence"]
    assert not identity["answer_validation_enabled"] and not identity["production_modified"]
    assert identity["trainable_parameters"] == 4456448
    assert identity["full_attention_layers"] == 16 and identity["linear_attention_layers"] == 48
    assert read_json(run / "workflow.json")["stage"] == "complete"
    tasks = rows(root / review["tasks"])
    outputs = rows(run / "generated_native46.jsonl")
    cases = review["cases"]
    old = read_json(root / review["policy_source"])
    shallow = read_json(root / review["shallow750_review"])
    assert review["semantic_adjudication_complete"] and review["format_and_citation_binding_excluded"]
    assert len(tasks) == len(outputs) == len(cases) == 46
    assert len({t["id"] for t in tasks}) == len({r["id"] for r in outputs}) == 46
    for pos, (task, output, case) in enumerate(zip(tasks, outputs, cases), 1):
        assert task["position"] == case["position"] == pos
        assert task["id"] == output["id"] and task["question"] == output["question"]
        assert output["prompt_sha256"] == task["bonsai_prompt_sha256"]
        assert hashlib.sha256(task["bonsai_prompt"].encode()).hexdigest() == output["prompt_sha256"]
        verify_documents(task, task["bonsai_prompt"])
        assert output["evidence_count"] == 20 and not output["truncated"]
        assert output["prefix_tokens"] == identity["prefix_length"]
        assert not output["output_limit_reached"]
        parsed = json.loads(output["raw_text"])
        assert parsed == output["parsed"] and output["valid_json"] is True
        assert case["category"] in CATEGORIES and isinstance(case["core_complete"], bool)
        assert case["category"] != "complete" or case["core_complete"]
        assert case["note"] and case["source_eids"]
        assert set(case["source_eids"]) <= {d["evidence_id"] for d in task["documents"]}
        if "borderline_alternative" in case:
            assert case["borderline_alternative"] in CATEGORIES
            assert case["borderline_alternative"] != case["category"]
    arms = {
        "fourb_sft_kto": [c["fourb"] for c in old["cases"]],
        "bonsai_initial_shallow": [c["bonsai"] for c in old["cases"]],
        "bonsai_shallow750": shallow["cases"],
        "bonsai_deep750": cases,
    }
    previous8 = set(read_json(root / "heldout8_selection_20261014.json")["ids"])
    gap_positions = {c["position"] for c in old["cases"] if c.get("evidence_gap")}
    assert gap_positions == {4, 5, 12, 19, 42, 43}
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
        assert len(groups) == 43
        summary[arm]["groups43_equal_weight_rate"] = sum(
            sum(v) / len(v) for v in groups.values()
        ) / len(groups)
    paired = {}
    for arm in arms:
        if arm == "bonsai_deep750":
            continue
        categories = defaultdict(list)
        for pos, (now, before) in enumerate(zip(cases, arms[arm]), 1):
            a, b = now["category"] == "complete", before["category"] == "complete"
            key = "both" if a and b else "gained_deep" if a else "lost_deep" if b else "neither"
            categories[key].append(pos)
        paired[arm] = {k: {"n": len(categories[k]), "positions": categories[k]}
                       for k in ("both", "gained_deep", "lost_deep", "neither")}
    correct = sum(c["category"] == "complete" for c in cases)
    lower = correct - sum(c["category"] == "complete" and c.get("borderline_alternative") == "partial"
                          for c in cases)
    upper = correct + sum(c["category"] != "complete" and c.get("borderline_alternative") == "complete"
                          for c in cases)
    sample_review_path = root / "READER_ERROR_V3_SPOTCHECK_20261017.json"
    sample_review = read_json(sample_review_path)
    sample_run = root / sample_review["run"]
    assert sha(root / sample_review["sample"]) == sample_review["sample_sha256"]
    assert sha(sample_run / "plan.json") == sample_review["plan_sha256"]
    assert sha(sample_run / "merged_candidate/manifest.json") == sample_review["candidate_manifest_sha256"]
    manifest = read_json(sample_run / "merged_candidate/manifest.json")
    assert manifest["new_train"] == sample_review["new_rows"] == 486
    assert manifest["train"] == sample_review["candidate_train_rows"] == 1236
    assert manifest["ready_for_training"] is False and sample_review["ready_for_training"] is False
    assert sample_review["semantic_adjudication_complete"]
    samples = rows(root / sample_review["sample"])
    assert len(samples) == len(sample_review["cases"]) == 15
    assert len({s["id"] for s in samples}) == 15
    assert set(Counter(s["targeted_kind"] for s in samples).values()) == {3}
    assert len({s["targeted_kind"] for s in samples}) == 5
    spot_counts = Counter()
    for pos, (sample, decision) in enumerate(zip(samples, sample_review["cases"]), 1):
        verify_documents(sample)
        assert sample["id"] == decision["id"] and decision["position"] == pos
        assert sample["targeted_kind"] == decision["targeted_kind"]
        assert not sample["truncated"] and sample["evidence_count"] == 20
        assert set(decision["source_eids"]) <= {d["evidence_id"] for d in sample["documents"]}
        if decision["read_scope"] == "all20_complete_passages":
            assert len(set(decision["source_eids"])) == 20
        assert decision["judgment"] in ("pass", "fail")
        assert not decision["error_tags"] if decision["judgment"] == "pass" else decision["error_tags"]
        # Independently checking retained rows is necessary even when the teacher self-review approved them.
        assert sample["semantic_review"]["label_supported"] is True
        assert sample["semantic_review"]["decision_appropriate"] is True
        assert sample["semantic_review"]["issues"] == []
        spot_counts[decision["judgment"]] += 1
    return {
        "integrity_passed": True,
        "review_sha256": sha(review_path),
        "spotcheck_review_sha256": sha(sample_review_path),
        "summary": summary,
        "paired": paired,
        "borderline_sensitivity_not_confidence_interval": {
            "positions": [c["position"] for c in cases if "borderline_alternative" in c],
            "lower_complete": lower, "upper_complete": upper,
            "lower_rate": lower / 46, "upper_rate": upper / 46,
        },
        "generation_separate_from_semantic_accuracy": {
            "n": len(outputs),
            "parsed_json": sum(r["valid_json"] for r in outputs),
            "native_answer_structure_valid_offline_only": sum(structure_valid(r["parsed"]) for r in outputs),
            "actions": dict(Counter(r["parsed"]["next_action"] for r in outputs)),
            "input_truncated": sum(r["truncated"] for r in outputs),
            "output_length_limit": sum(r["output_limit_reached"] for r in outputs),
        },
        "error_tags_nonexclusive": dict(Counter(tag for c in cases for tag in c["error_tags"])),
        "spotcheck": {
            "n": 15, "semantic_counts": dict(spot_counts),
            "failed_positions": [c["position"] for c in sample_review["cases"] if c["judgment"] == "fail"],
            "style_warning_positions": [c["position"] for c in sample_review["cases"] if "style_warning" in c],
            "ready_for_training": False,
            "representative_accuracy_estimate": False,
            "candidate_manifest_left_unchanged": True,
        },
        "threshold60_requires_complete": 28,
        "threshold60_met": correct >= 28,
        "caution": "Assistant adjudication, not human gold. Fixed historical development prompts; not a blind or end-to-end retrieval test. Historical baseline decisions reused without modification. Audit validates consistency, not semantics independently.",
    }


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
