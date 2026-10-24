"""Validate exact review receipts and export a separate, not-released repair candidate."""
import argparse
from collections import Counter
import json
from pathlib import Path

import annotation_protocol as p
import expand_input_failures_20261022 as e
import repair_input_failure_data_20261022 as base
import repair_input_failure_followup_20261022 as followup
import repair_input_failure_179_final_20261022 as last

ROOT = Path(__file__).resolve().parent
NOTES = ROOT / "INPUT_FAILURE_REPAIR_POSTREVIEW_20261022.json"


def check_review_run(directory, expected_settings, superseded_positions=()):
    plan = base.read(directory / "plan.json")
    assert plan["settings"] == vars(expected_settings)
    assert p.file_sha(directory / "requests.jsonl") == plan["requests_sha256"]
    for name, sha in plan["inputs_sha256"].items():
        assert p.file_sha(name) == sha
    for name, sha in plan["code_sha256"].items():
        assert p.file_sha(ROOT / name) == sha
    status = base.read(directory / "status.json")
    assert status["stage"] == "batch_finished", "review batch still running"
    tasks = {row["id"]: row for row in p.json_rows(directory / "requests.jsonl")}
    reviews = {row["id"]: row for row in base.read(directory / "reviews.json")}
    receipts = {row["call_id"]: row for row in p.json_rows(directory / "api_receipts.jsonl")}
    for key, result in reviews.items():
        task = tasks[key]
        if result["state"] == "pending":
            assert task["source_position"] in superseded_positions, "non-superseded review pending"
            assert result["result"].get("error", {}).get("kind") == "incomplete_output"
            continue
        assert result["state"] in ("accepted", "quarantined")
        receipt = result["result"]["review"]
        assert receipts[receipt["call_id"]] == receipt
        assert receipt["task_id"] == task["id"] and receipt["phase"] == "review_repair"
        assert receipt["endpoint"] == p.CODING_PLAN_ENDPOINT
        assert receipt["returned_model"] == "glm-5.3" and receipt["response_id"]
        assert receipt["finish_reason"] == "stop" and receipt["accepted"] is True
        assert json.loads(receipt["content"]) == receipt["parsed"]
        assert receipt["cache_key"] == p.sha(p.compact(dict(
            phase="review_repair", messages=e.review_messages(task, task["candidate_target"]),
            settings=expected_settings.cache_settings())))
        assert result["result"]["candidate_target_sha256"] == p.sha(p.compact(task["candidate_target"]))
        verdict = p.check_review(receipt["parsed"])
        good = verdict["label_supported"] and verdict["decision_appropriate"] and not verdict["issues"]
        assert (result["state"] == "accepted") == bool(good)
    assert len(tasks) == len(reviews) == status["total_tasks"]
    return plan, tasks, reviews, status


def finalize(second):
    first = followup.FIRST
    second_settings = base.settings()
    second_settings.max_api_calls, second_settings.max_total_tokens = 6, 300000
    first_run = check_review_run(first, base.settings(), superseded_positions=followup.TARGETS)
    second_run = check_review_run(second, second_settings)
    third_settings = base.settings()
    third_settings.max_api_calls, third_settings.max_total_tokens = 2, 100000
    third_run = check_review_run(last.OUT, third_settings)
    source = Path(first_run[0]["source"])
    combined = second / "combined_candidate"
    assert (combined / "manifest.json").exists(), "export combined candidate first"
    manifest = base.read(combined / "manifest.json")
    assert p.file_sha(combined / "train.jsonl") == manifest["train_sha256"]
    output = list(p.json_rows(combined / "train.jsonl"))
    source_rows = list(p.json_rows(source / "annotation/requests.jsonl"))
    source_labels = {r["id"]: r for r in p.json_rows(source / "annotation/train.jsonl")}
    positions = {row["id"]: i for i, row in enumerate(source_rows, 1)}
    selected = {}
    for directory, (_, tasks, reviews, _) in zip(
            (first, second, last.OUT), (first_run, second_run, third_run)):
        for key, task in tasks.items():
            selected[task["source_id"]] = (task, reviews[key], directory.name)
    final_task = next(iter(third_run[1].values()))
    final_result = third_run[2][final_task["id"]]
    assert final_task["source_position"] == 179 and final_result["state"] == "accepted"
    output = [r for r in output if r["candidate_lineage"]["source_id"] != final_task["source_id"]]
    final_row = {k: v for k, v in final_task.items() if k not in (
        "teacher_messages", "candidate_target", "original_label_receipt")}
    final_receipt = final_result["result"]["review"]
    final_row.update(
        target=p.compact(final_task["candidate_target"]),
        user_prompt=final_task["teacher_messages"][1]["content"],
        source_label="assistant_repaired_GLM_reviewed_full20",
        label_prompt_sha256=final_task["teacher_prompt_sha256"],
        annotation_response_id=None,
        original_annotation_response_id=final_task["original_label_receipt"]["response_id"],
        semantic_review_sampled=True, semantic_review=final_receipt["parsed"],
        repair_review_response_id=final_receipt["response_id"],
        assistant_content_review="pending_post_GLM",
        candidate_lineage=dict(
            origin="assistant_repair", source_id=final_task["source_id"], source_position=179,
            review_run=last.OUT.name))
    output.append(final_row)
    output.sort(key=lambda row: positions[row["candidate_lineage"]["source_id"]])
    notes = {case["position"]: case for case in base.read(NOTES)["cases"]}
    assert set(notes) == {task["source_position"] for task, _, _ in selected.values()}
    previous = {case["id"]: case for case in base.read(base.REVIEW)["cases"]}
    post_cases, reviewed_subset = [], []
    for row in output:
        sid = row["candidate_lineage"]["source_id"]
        position = positions[sid]
        if sid in selected:
            task, result, directory = selected[sid]
            assert result["state"] == "accepted"
            assert row["id"] == task["id"] and json.loads(row["target"]) == task["candidate_target"]
            for field in ("question", "round", "targeted_kind", "documents", "prompt", "prompt_tokens"):
                assert row[field] == task[field]
            assert row["repair_review_response_id"] == result["result"]["review"]["response_id"]
            row["assistant_content_review"] = "pass"
            post_cases.append(dict(notes[position], source_id=sid, candidate_id=row["id"],
                                   target_sha256=p.sha(row["target"]), judgment="pass",
                                   review_run=directory, review_response_id=row["repair_review_response_id"]))
        else:
            original = source_labels[sid]
            for field in ("target", "question", "round", "documents", "prompt", "targeted_kind"):
                assert row[field] == original[field]
            expected = previous.get(sid, {}).get("content_judgment", "pending")
            assert expected != "fail" and row["assistant_content_review"] == expected
        messages, _ = p.prompt_for(row["question"], row["round"], row["documents"])
        p.check_row(dict(row, teacher_messages=messages))
        p.check_target(json.loads(row["target"]), row)
        assert row["prompt_tokens"] + 8192 + 16 <= 32768
        if row["assistant_content_review"] == "pass":
            reviewed_subset.append(row)
    assert len(post_cases) == 14 and len(output) == 180, "missing repair; never silently fill from source"
    assert len({r["id"] for r in output}) == len(output)
    assert len({p.normalized_question(r["question"]) for r in output}) == len(output)
    p.assert_split(dict(train=output, val=list(p.json_rows(ROOT / "curated_shallow_data_20261019_v1/val.jsonl"))))
    dest = second / "final_candidate"
    assert not dest.exists(), "never overwrite a finalized candidate"
    dest.mkdir()
    for name, rows in (("all180.jsonl", output), ("assistant_reviewed24.jsonl", reviewed_subset)):
        with (dest / name).open("x") as stream:
            stream.writelines(p.compact(row) + "\n" for row in rows)
    p.atomic_json(dest / "postrepair_review.json", dict(
        method=base.read(NOTES)["method"], cases=post_cases,
        ready_for_training=False, notes_sha256=p.file_sha(NOTES)))
    result = dict(
        rows=len(output), repaired_or_recovered_reviewed=len(post_cases),
        assistant_reviewed=len(reviewed_subset),
        assistant_pending=len(output) - len(reviewed_subset),
        kind_counts=dict(Counter(r["targeted_kind"] for r in output)),
        action_counts=dict(Counter(json.loads(r["target"])["next_action"] for r in output)),
        first_run_states=first_run[3]["job_states"], followup_run_states=second_run[3]["job_states"],
        final179_run_states=third_run[3]["job_states"],
        api_calls=sum(run[3]["api_calls"] for run in (first_run, second_run, third_run)),
        accounted_tokens=sum(run[3]["accounted_tokens"] for run in (first_run, second_run, third_run)),
        source_export_sha256=p.file_sha(source / "annotation/train.jsonl"),
        source_requests_sha256=p.file_sha(source / "annotation/requests.jsonl"),
        all180_sha256=p.file_sha(dest / "all180.jsonl"),
        reviewed_subset_sha256=p.file_sha(dest / "assistant_reviewed24.jsonl"),
        repaired_labels=10, reclassified=2, question_wording_fixed=1,
        original_pending_reviews_resolved=4, recovered_without_label_change=1,
        source_dataset_modified=False, production_modified=False,
        no_training_started=True, no_base_dataset_merge=True,
        ready_for_training=False, assistant_semantic_review_pending=True,
        new_labels_not_independent_human_gold=True,
        note="Original 171-row export and original pending/quarantined states are retained. "
             "This independent candidate has repairs; 156 rows still await assistant content review.")
    p.atomic_json(dest / "manifest.json", result)
    report = [
        "# 新增输入层数据修复结果",
        "",
        "已另存180条修订候选，原始数据与账本未改动；未并入基准训练集、未训练、未改发布模型。",
        "",
        "- 修复10条内容标签（原7条，加恢复复核时发现的108、144、179三条）。",
        "- 纠正2条题型（45、114），修正1处题面歧义（178）。",
        "- 原4条未完成样本已补齐独立复核；175保留原正确标签，其余3条改正内容。",
        f"- 实际GLM复核调用{result['api_calls']}次；最终采用的14个修改/恢复样本均有干净全文复核回执，助手逐项确认。",
        "- 不把GLM自复核等同独立人工金标：当前24条经助手内容审核，另156条仍待内容审核。",
        "",
        "新增纠错：108不再将另一场惹出停播负债的直播当成致富直播；144不再把同车的灰礼帽当成指定客人；"
        "179将镣铐与虚弱形体归还给望。158保留情境推理但明确不确定性，不强断言B辨色差或确定换壳。",
        "",
        "文件：`final_candidate/all180.jsonl`为全量候选；`assistant_reviewed24.jsonl`为已审子集；"
        "`postrepair_review.json`记录逐条修复依据及真实复核response_id。整批仍为`ready_for_training=false`。",
        "",
    ]
    (dest / "REPORT.md").write_text("\n".join(report))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(p.compact(finalize(args.out.resolve())))
