"""Bind manual content judgments to immutable final outputs; no auto semantic grading."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVIEW = "INPUT_FINAL1117_CONTENT_REVIEW_20261019.json"
CATEGORIES = ("complete", "partial", "wrong", "no_answer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aggregate(root=ROOT):
    review = read(root / REVIEW)
    run = root / review["run"]
    for path, key in (
        (root / review["tasks"], "tasks_sha256"),
        (run / "generated_native46.jsonl", "output_sha256"),
        (run / "generation_receipt.json", "generation_receipt_sha256"),
        (run / "trained/run.json", "run_sha256"),
        (root / review["policy_source"], "policy_source_sha256"),
        (root / review["shallow750_review"], "shallow750_review_sha256"),
    ):
        assert sha(path) == review[key], path
    identity = read(run / "trained/run.json")
    receipt = read(run / "generation_receipt.json")
    workflow = read(run / "workflow.json")
    assert identity["training_complete"] and identity["optimizer_steps"] == 1117
    assert identity["train_examples"] == 1117 and identity["backbone_bytewise_unchanged"]
    assert workflow["stage"] == "complete" and workflow["data_sha256"] == identity["data_sha256"]
    assert receipt["completed"] and receipt["count"] == 46
    assert receipt["adapter_sha256"] == review["adapter_sha256"]
    assert receipt["output_sha256"] == review["output_sha256"]
    assert receipt["no_input_truncation"] and not receipt["answer_validation_enabled"]
    assert review["semantic_adjudication_complete"] and review["format_and_citation_binding_excluded"]
    tasks = rows(root / review["tasks"])
    outputs = rows(run / "generated_native46.jsonl")
    cases = review["cases"]
    previous = read(root / review["shallow750_review"])["cases"]
    assert len(tasks) == len(outputs) == len(cases) == len(previous) == 46
    assert len({r["id"] for r in outputs}) == 46
    bindings = []
    paired = {key: [] for key in ("both_complete", "gained_complete", "lost_complete", "neither_complete")}
    for pos, (task, output, case, before) in enumerate(zip(tasks, outputs, cases, previous), 1):
        assert task["position"] == case["position"] == before["position"] == pos
        assert task["id"] == output["id"] and task["question"] == output["question"]
        assert output["prompt_sha256"] == task["bonsai_prompt_sha256"]
        assert hashlib.sha256(task["bonsai_prompt"].encode()).hexdigest() == output["prompt_sha256"]
        assert len(task["documents"]) == output["evidence_count"] == 20
        assert not output["truncated"] and not output["output_limit_reached"]
        assert output["prefix_tokens"] == 16
        for doc in task["documents"]:
            assert hashlib.sha256(doc["text"].encode()).hexdigest() == doc["sha256"]
            assert doc["text"] in task["bonsai_prompt"]
        try:
            parsed = json.loads(output["raw_text"])
        except ValueError:
            parsed = None
        assert parsed == output["parsed"] and output["valid_json"] == (parsed is not None)
        assert case["category"] in CATEGORIES and type(case["core_complete"]) is bool
        assert case["category"] != "complete" or case["core_complete"]
        assert case["source_eids"] and case["note"]
        assert set(case["source_eids"]) <= {doc["evidence_id"] for doc in task["documents"]}
        a, b = case["category"] == "complete", before["category"] == "complete"
        key = "both_complete" if a and b else "gained_complete" if a else "lost_complete" if b else "neither_complete"
        paired[key].append(pos)
        bindings.append(dict(
            position=pos, id=output["id"], question=output["question"],
            raw_text_sha256=hashlib.sha256(output["raw_text"].encode()).hexdigest(),
            prompt_sha256=output["prompt_sha256"], category=case["category"],
            previous_category=before["category"], note=case["note"]))
    counts = Counter(case["category"] for case in cases)
    old_counts = Counter(case["category"] for case in previous)
    correct = counts["complete"]
    group_values = {}
    for task, case in zip(tasks, cases):
        group_values.setdefault(task["question_group"], []).append(case["category"] == "complete")
    assert len(group_values) == 43
    return dict(
        integrity_passed=True, reviewed_count=46, review_sha256=sha(root / REVIEW),
        output_sha256=review["output_sha256"], final_training_steps=1117,
        categories={key: counts[key] for key in CATEGORIES},
        complete_rate=correct / 46,
        previous_shallow750=dict(
            categories={key: old_counts[key] for key in CATEGORIES},
            complete_rate=old_counts["complete"] / 46,
            reused_historical_outputs_and_judgments=True),
        complete_rate_change_percentage_points=(correct - old_counts["complete"]) * 100 / 46,
        paired_complete_transitions=paired,
        groups43_equal_weight_complete_rate=sum(sum(v) / len(v) for v in group_values.values()) / 43,
        borderline_sensitivity_not_confidence_interval=dict(
            positions=[case["position"] for case in cases if "borderline_alternative" in case],
            strict_complete=correct,
            if_marked_partial_cases_relaxed_to_complete=correct + sum(
                case["category"] != "complete" and case.get("borderline_alternative") == "complete"
                for case in cases)),
        format_separate_from_content=dict(
            valid_json=sum(output["valid_json"] for output in outputs),
            invalid_json_positions=[i for i, output in enumerate(outputs, 1) if not output["valid_json"]],
            no_input_truncation=True, output_limit_reached=0,
            invalid_json_case_content_category=cases[26]["category"]),
        final_teacher_forced_mean_nll=sum(v["nll"] for v in identity["final_validation"]) / 46,
        nll_is_not_answer_accuracy=True, blind_test=False, end_to_end=False,
        independent_human_gold=False, production_modified=False, cases=bindings,
    )


if __name__ == "__main__":
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))
