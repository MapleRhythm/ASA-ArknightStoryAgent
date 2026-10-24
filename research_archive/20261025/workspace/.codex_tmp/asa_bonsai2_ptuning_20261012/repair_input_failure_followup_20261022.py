"""Three evidence-based corrections discovered during truncated-review recovery.

Separate sealed review run, plus combined candidate export. No source mutation.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import annotation_protocol as p
import repair_input_failure_data_20261022 as base

ROOT = Path(__file__).resolve().parent
FIRST = ROOT / "input_failure_repairs_20261022_attempt01"
VERSION = "input_failure_followup_repairs_20261022_v1"
TARGETS = {
    108: {
        "next_action": "abstain",
        "reason": "当前完整20段只交代主播U凭一场直播获得分红并还清贷款，没有交代那场直播具体播了什么。"
                  "批评旅游地并喊“不如多索雷斯”的直播在另一段引出停播、违约金及新贷款；"
                  "证据没有把它与一夜暴富的那场直播明确连在一起，不能将两段经历混为同一场直播。"},
    144: {
        "next_action": "abstain",
        "reason": "全部20段中，诺希斯只确认那位客人已登上开往银心湖的列车，没有揭示其身份或此行目的。"
                  "灰礼帽也出现在列车上，但证据没有把他与诺希斯所指的客人绑定；"
                  "不能因同车出现就将灰礼帽的身份和任务当成所问客人的答案。"},
    179: {
        "next_action": "answer_directly",
        "supported_facts": [
            {"fact": "巨兽所说的是姐妹用梦境欺骗祂，在望的镣铐上造出一个口子，使望刚刚挣脱束缚；"
                     "巨兽接着讥讽望的形体仍然虚弱，不是在说巨兽自己的镣铐或祂已挣脱束缚。",
             "evidence_ids": ["E4", "E15"]},
            {"fact": "这次令和均编织的梦境中，那场对决没有真实发生，但她们借此看清了巨兽的行动、招法、弱点与心理，"
                     "对方也看清了她们。", "evidence_ids": ["E4"]}
        ]}
}


# Keep the original worker file and its hashes intact; use explicitly new settings.
BASE_SETTINGS = base.settings


def bind_settings():
    def bounded():
        value = BASE_SETTINGS()
        value.max_api_calls = 6
        value.max_total_tokens = 300000
        return value
    base.settings = bounded


def prepare(out):
    assert not out.exists()
    plan = base.verify(FIRST)
    source = Path(plan["source"])
    tasks = list(p.json_rows(FIRST / "requests.jsonl"))
    selected = []
    for position, target in TARGETS.items():
        row = dict(next(row for row in tasks if row["source_position"] == position))
        row.update(
            candidate_target=target, candidate_origin="assistant_repair",
            repair_mode="explicit_patch_after_recovery_evidence_review",
            repair_reason={
                108: "一夜暴富直播与引出违约负债的直播不能无据合并",
                144: "同车不是同一身份，取消无据客人指认",
                179: "你的镣铐指受话人望，不是发话的巨兽；纠正束缚的主体"}[position])
        row["id"] = p.sha(p.compact(dict(version=VERSION, source=row["source_id"], target=target)))[:24]
        p.check_row(row)
        p.check_target(target, row)
        selected.append(row)
    out.mkdir(mode=0o700)
    with (out / "requests.jsonl").open("x") as stream:
        stream.writelines(p.compact(row) + "\n" for row in selected)
    bind_settings()
    result = dict(
        version=base.VERSION, followup_version=VERSION, source=str(source),
        inputs_sha256={str(path): p.file_sha(path) for path in (
            FIRST / "plan.json", FIRST / "requests.jsonl",
            source / "annotation/requests.jsonl", source / "annotation/train.jsonl",
            source / "annotation/ledger.sqlite3")},
        code_sha256={name: p.file_sha(ROOT / name) for name in (
            Path(__file__).name, "repair_input_failure_data_20261022.py",
            "annotation_protocol.py", "expand_input_failures_20261022.py")},
        requests_sha256=p.file_sha(out / "requests.jsonl"), settings=vars(base.settings()),
        tasks=3, assistant_read_scope="all20_complete_passages_for_each_task",
        ready_for_training=False, original_data_modified=False, auto_training_enabled=False)
    p.atomic_json(out / "plan.json", result)
    return result


def combined(out):
    bind_settings()
    second_plan = base.verify(out)
    # Independently verify the first run without substituting its sealed settings.
    base.settings = BASE_SETTINGS
    first_plan = base.verify(FIRST)
    assert first_plan["source"] == second_plan["source"]
    source = Path(first_plan["source"])
    dest = out / "combined_candidate"
    assert not dest.exists()
    selected = {}
    for directory in (FIRST, out):
        tasks = list(p.json_rows(directory / "requests.jsonl"))
        reviews = {r["id"]: r for r in base.read(directory / "reviews.json")}
        for task in tasks:
            selected[task["source_id"]] = (task, reviews.get(task["id"]), directory)
    originals = {r["id"]: r for r in p.json_rows(source / "annotation/train.jsonl")}
    original_requests = {r["id"]: r for r in p.json_rows(source / "annotation/requests.jsonl")}
    previous = {r["id"]: r for r in base.read(base.REVIEW)["cases"]}
    output, held = [], []
    for row in originals.values():
        if row["id"] in selected:
            continue
        item = dict(row)
        item.update(assistant_content_review=previous.get(row["id"], {}).get("content_judgment", "pending"),
                    candidate_lineage=dict(origin="unchanged_source_export", source_id=row["id"]))
        output.append(item)
    for source_id, (task, result, directory) in selected.items():
        if not result or result["state"] != "accepted":
            held.append(dict(source_id=source_id, position=task["source_position"], review_run=directory.name,
                             state=result["state"] if result else "unstarted", result=result))
            continue
        receipt = result["result"]["review"]
        p.check_review(receipt["parsed"])
        assert receipt["finish_reason"] == "stop" and receipt["accepted"]
        verdict = receipt["parsed"]
        assert verdict["label_supported"] and verdict["decision_appropriate"] and not verdict["issues"]
        assert result["result"]["candidate_target_sha256"] == p.sha(p.compact(task["candidate_target"]))
        row = {k: v for k, v in task.items() if k not in
               ("teacher_messages", "candidate_target", "original_label_receipt")}
        row.update(target=p.compact(task["candidate_target"]), user_prompt=task["teacher_messages"][1]["content"],
                   label_prompt_sha256=task["teacher_prompt_sha256"], semantic_review=verdict,
                   semantic_review_sampled=True, repair_review_response_id=receipt["response_id"],
                   assistant_content_review="pending_post_GLM",
                   source_label="assistant_repaired_GLM_reviewed_full20" if task["candidate_origin"] == "assistant_repair"
                                else "retained_GLM_label_fresh_review",
                   annotation_response_id=None if task["candidate_origin"] == "assistant_repair"
                                          else task["original_label_receipt"]["response_id"],
                   original_annotation_response_id=task["original_label_receipt"]["response_id"],
                   candidate_lineage=dict(origin=task["candidate_origin"], source_id=source_id,
                                          source_position=task["source_position"], review_run=directory.name))
        output.append(row)
    output.sort(key=lambda r: list(original_requests).index(r["candidate_lineage"]["source_id"]))
    assert len(output) == len({r["id"] for r in output})
    assert len(output) == len({p.normalized_question(r["question"]) for r in output})
    val = list(p.json_rows(ROOT / "curated_shallow_data_20261019_v1/val.jsonl"))
    p.assert_split(dict(train=output, val=val))
    for row in output:
        messages, _ = p.prompt_for(row["question"], row["round"], row["documents"])
        p.check_row(dict(row, teacher_messages=messages))
        p.check_target(json.loads(row["target"]), row)
    dest.mkdir()
    with (dest / "train.jsonl").open("x") as stream:
        stream.writelines(p.compact(row) + "\n" for row in output)
    p.atomic_json(dest / "held.json", held)
    manifest = dict(
        version=VERSION, rows=len(output), held=len(held),
        source_export_rows=171, selected_source_positions=sorted(t["source_position"] for t, _, _ in selected.values()),
        type_counts=dict(Counter(r["targeted_kind"] for r in output)),
        review_states=dict(Counter(r["assistant_content_review"] for r in output)),
        train_sha256=p.file_sha(dest / "train.jsonl"),
        evidence_count_per_example=20, no_truncation=True, protected_split_verified=True,
        source_export_sha256=p.file_sha(source / "annotation/train.jsonl"),
        new_base_training_merge=False, original_data_modified=False,
        ready_for_training=False, assistant_semantic_review_pending=True,
        production_modified=False, training_started=False)
    p.atomic_json(dest / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "launch", "run", "combined", "status"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        result = prepare(out)
    elif args.command == "combined":
        result = combined(out)
    else:
        bind_settings()
        if args.command == "status":
            result = {name: base.read(out / name) for name in ("status.json", "preflight.json")
                      if (out / name).exists()}
        elif args.command == "run":
            result = base.run(out)
        else:
            base.verify(out)
            assert not (out / "launch.json").exists()
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--out", str(out)]
            with (out / "run.log").open("x") as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
            result = dict(pid=process.pid, tasks=3, max_api_calls=6,
                          max_accounted_tokens=300000, ready_for_training=False)
            p.atomic_json(out / "launch.json", result)
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
