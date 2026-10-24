"""Offline paired-case integrity/statistics; never judges or rewrites answers."""
import argparse
from collections import Counter
import json
from pathlib import Path

from native46_eval import check_task, file_sha, parse_raw, rows


def audit(out, old_selection=None, review=None, require_complete=False):
    manifest = json.loads((out / "manifest.json").read_text())
    assert file_sha(out / "tasks.jsonl") == manifest["tasks_sha256"]
    tasks = rows(out / "tasks.jsonl")
    assert len(tasks) == len({t["id"] for t in tasks}) == 46
    assert [t["id"] for t in tasks] == manifest["task_order"]
    for task in tasks:
        check_task(task)
    indexed = {t["id"]: t for t in tasks}
    old_ids = set(json.loads(old_selection.read_text())["ids"]) if old_selection else set()
    assert not old_ids or len(old_ids) == 8 and old_ids <= set(indexed)
    outputs, states = [], {}
    for name in ("fourb", "bonsai_shard0", "bonsai_shard1"):
        state_path = out / f"{name}.status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"stage": "not_started"}
        states[name] = state
        result_path = out / f"{name}.results.jsonl"
        values = rows(result_path) if result_path.exists() else []
        expected_ids = manifest["task_order"] if name == "fourb" else manifest["task_order"][int(name[-1])::2]
        assert [r["id"] for r in values] == expected_ids[:len(values)]
        assert len(values) == len({r["id"] for r in values})
        if state["stage"] == "completed":
            assert len(values) == len(expected_ids) == state["completed"]
            assert file_sha(result_path) == state["results_sha256"]
        for output in values:
            task = indexed[output["id"]]
            assert output["question"] == task["question"]
            assert output["round"] == task["round"]
            assert output["position"] == task["position"]
            assert output["evidence_count"] == 20 and not output["input_truncated"]
            assert not output["answer_validation"]
            assert output["prompt_sha256"] == task[f'{output["arm"]}_prompt_sha256']
            assert parse_raw(output["raw_output"], output["protocol_prefill"]) == output["parsed_output_for_offline_inspection_only"]
            assert output["protocol_prefill"] == ("{" if output["arm"] == "fourb" else "")
        outputs.extend(values)
    all_complete = all(s["stage"] == "completed" for s in states.values())
    if require_complete:
        assert all_complete and len(outputs) == 92, "generation not complete"
    summary = {}
    for arm in ("fourb", "bonsai"):
        values = [r for r in outputs if r["arm"] == arm]
        assert len({r["id"] for r in values}) == len(values)
        summary[arm] = dict(
            completed=len(values), expected=46,
            valid_json=sum(isinstance(r["parsed_output_for_offline_inspection_only"], dict) for r in values),
            output_length_limits=sum(r["output_reached_length_limit"] for r in values),
            parsed_actions=dict(Counter(
                r["parsed_output_for_offline_inspection_only"].get("next_action", "missing")
                if isinstance(r["parsed_output_for_offline_inspection_only"], dict) else "invalid_json"
                for r in values)))
    semantic = None
    if review:
        reviewed = json.loads(review.read_text())
        decisions = reviewed["reviews"]
        expected_pairs = {(r["arm"], r["id"]) for r in outputs}
        observed_pairs = [(r["arm"], r["id"]) for r in decisions]
        assert len(observed_pairs) == len(set(observed_pairs)), "duplicate review"
        assert set(observed_pairs) <= expected_pairs
        semantic = {}
        for arm in ("fourb", "bonsai"):
            values = [d for d in decisions if d["arm"] == arm]
            for d in values:
                assert type(d["semantic_complete"]) is bool
                assert type(d["strict_deliverable"]) is bool
                assert not d["strict_deliverable"] or d["semantic_complete"]
                assert isinstance(d["note"], str) and d["note"].strip()
            arm_summary = {}
            for subset, ids in (
                ("all46", set(indexed)),
                ("previous8", old_ids),
                ("new38", set(indexed) - old_ids if old_ids else set()),
            ):
                chosen = [d for d in values if d["id"] in ids]
                arm_summary[subset] = dict(
                    reviewed=len(chosen), denominator=len(ids),
                    semantic_complete=sum(d["semantic_complete"] for d in chosen),
                    strict_deliverable=sum(d["strict_deliverable"] for d in chosen))
            grouped = {}
            for task in tasks:
                grouped.setdefault(task["question_group"], []).append(task["id"])
            if len(values) == 46:
                by_id = {d["id"]: d for d in values}
                arm_summary["question_groups"] = {
                    "n": len(grouped),
                    "semantic_complete_group_mean": sum(
                        sum(by_id[i]["semantic_complete"] for i in ids) / len(ids)
                        for ids in grouped.values()) / len(grouped),
                    "strict_group_mean": sum(
                        sum(by_id[i]["strict_deliverable"] for i in ids) / len(ids)
                        for ids in grouped.values()) / len(grouped)}
            semantic[arm] = arm_summary
    return dict(
        integrity_passed=True, all_generation_complete=all_complete,
        expected_generations=92, observed_generations=len(outputs),
        unique_question_groups=manifest["unique_question_count"],
        workers=states, mechanical_summary_not_accuracy=summary,
        offline_semantic_review=semantic,
        interpretation="No QA accuracy claim without complete evidence adjudication.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old-selection", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.out, args.old_selection, args.review, args.require_complete),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
