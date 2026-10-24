"""Isolated, bounded review of explicit repairs; never alter the sealed source."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import annotation_protocol as p
import expand_input_failures_20261022 as e
from targeted_pt_v2 import read_credential

ROOT = Path(__file__).resolve().parent
VERSION = "input_failure_repair_review_20261022_v1"
SPEC = ROOT / "INPUT_FAILURE_REPAIRS_20261022.json"
REVIEW = ROOT / "INPUT_FAILURE_DATA_REVIEW_20261022.json"
SOURCE_NAME = "input_failure_expansion_20261022_attempt01"


def read(path):
    return json.loads(Path(path).read_text())


def jobs_readonly(source):
    with sqlite3.connect((source / "annotation/ledger.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        return {key: (state, json.loads(value)) for key, state, value in
                db.execute("SELECT task_id,state,result FROM jobs")}


def settings():
    return p.Settings(endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3",
                      workers=2, attempts=1, timeout=360, request_interval=1,
                      max_output_tokens=16384, max_api_calls=32,
                      max_total_tokens=1_200_000, review_fraction=1, seed=20261022)


def target_for(source, row, position, jobs):
    state, result = jobs[row["id"]]
    if state != "pending":
        return result["label"]["parsed"], result["label"]
    # All four incomplete jobs have successful labels but truncated reviews.
    candidates = [
        receipt for receipt in p.json_rows(source / "annotation/api_receipts.jsonl")
        if receipt["task_id"] == row["id"] and receipt["phase"] == "annotate" and receipt["accepted"]]
    assert len(candidates) == 1, f"inspect pending label: {position}"
    receipt = candidates[0]
    assert receipt["finish_reason"] == "stop" and receipt["response_id"]
    expected = p.sha(p.compact(dict(
        phase="annotate", messages=e.annotation_messages(row),
        settings=e.settings("annotation").cache_settings())))
    assert receipt["cache_key"] == expected
    p.check_target(receipt["parsed"], row)
    return receipt["parsed"], receipt


def prepare(source, out, tokenizer_path=None):
    source, out = source.resolve(), out.resolve()
    assert not out.exists(), "never overwrite an existing repair run"
    spec = read(SPEC)
    assert p.file_sha(source / "annotation/requests.jsonl") == spec["requests_sha256"]
    rows = list(p.json_rows(source / "annotation/requests.jsonl"))
    jobs = jobs_readonly(source)
    changes = {r["position"]: r for r in spec["repairs"]}
    selected = sorted(set(changes) | set(spec["recover_pending_positions"]))
    assert len(selected) == 14 and len(changes) == 10
    tokenizer = None
    if tokenizer_path:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tasks = []
    for position in selected:
        original = rows[position - 1]
        p.check_row(original)
        target, original_receipt = target_for(source, original, position, jobs)
        change = changes.get(position, {})
        row = dict(original)
        for key in ("question", "targeted_kind"):
            if key in change:
                row[key] = change[key]
        row["teacher_messages"], row["prompt"] = p.prompt_for(
            row["question"], row["round"], row["documents"])
        row["teacher_prompt_sha256"] = p.sha(p.compact(row["teacher_messages"]))
        if row["question"] != original["question"]:
            assert tokenizer is not None, "changed question requires real tokenizer"
            row["prompt_tokens"] = len(tokenizer.encode(row["prompt"], add_special_tokens=False).ids)
        assert row["prompt_tokens"] + 8192 + 16 <= 32768
        row.update(
            source_id=original["id"], source_position=position,
            source_targeted_kind=original["targeted_kind"],
            source_pipeline_state=jobs[original["id"]][0],
            repair_reason=change.get("reason", "reuse successful annotation; recover truncated review only"),
            repair_mode="explicit_patch" if change else "review_length_recovery",
            candidate_target=change.get("target", target),
            original_label_receipt=original_receipt,
            original_target_sha256=p.sha(p.compact(target)),
            candidate_origin="assistant_repair" if "target" in change else "retained_GLM_label",
            source_question=original["question"])
        row["id"] = p.sha(p.compact(dict(
            version=VERSION, source=original["id"], target=row["candidate_target"],
            question=row["question"], kind=row["targeted_kind"])))[:24]
        p.check_row(row)
        p.check_target(row["candidate_target"], row)
        tasks.append(row)
    val = list(p.json_rows(source / "base_val.jsonl")) if (source / "base_val.jsonl").exists() else list(
        p.json_rows(ROOT / "curated_shallow_data_20261019_v1/val.jsonl"))
    original_by_id = {r["id"]: r for r in rows}
    replaced = {r["source_id"] for r in tasks}
    full = [r for key, r in original_by_id.items() if key not in replaced] + tasks
    assert len(full) == len({p.normalized_question(r["question"]) for r in full}) == 180
    p.assert_split(dict(train=full, val=val))
    out.mkdir(mode=0o700)
    with (out / "requests.jsonl").open("x") as stream:
        stream.writelines(p.compact(row) + "\n" for row in tasks)
    plan = dict(
        version=VERSION, source=str(source), tasks=len(tasks),
        repaired_or_reclassified=10, review_only_recoveries=4,
        inputs_sha256={str(path.resolve()): p.file_sha(path) for path in (
            SPEC, REVIEW, source / "annotation/requests.jsonl",
            source / "annotation/train.jsonl", source / "annotation/api_receipts.jsonl",
            source / "annotation/ledger.sqlite3")},
        code_sha256={name: p.file_sha(ROOT / name) for name in (
            Path(__file__).name, "annotation_protocol.py", "expand_input_failures_20261022.py",
            "expand_reader_errors_v3.py")},
        requests_sha256=p.file_sha(out / "requests.jsonl"),
        settings=vars(settings()), original_data_modified=False, new_annotation_calls=0,
        all_new_calls_review_only=True, ready_for_training=False,
        assistant_semantic_review_pending=True, no_gpu_usage=True,
        auto_training_enabled=False, production_modified=False)
    p.atomic_json(out / "plan.json", plan)
    return plan


def verify(out):
    plan = read(out / "plan.json")
    assert plan["version"] == VERSION
    assert p.file_sha(out / "requests.jsonl") == plan["requests_sha256"]
    assert vars(settings()) == plan["settings"]
    for name, expected in plan["inputs_sha256"].items():
        assert p.file_sha(name) == expected, name
    for name, expected in plan["code_sha256"].items():
        assert p.file_sha(ROOT / name) == expected, name
    return plan


class Runner(p.Runner):
    def fingerprint(self, row):
        return p.sha(p.compact(dict(
            version=VERSION, target=row["candidate_target"],
            messages=e.review_messages(row, row["candidate_target"]),
            settings=self.settings.cache_settings(), kind=row["targeted_kind"])))

    def process(self, row):
        p.check_row(row)
        p.check_target(row["candidate_target"], row)
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        assert not prior or prior[0] == fingerprint
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        try:
            receipt = self.call(row, "review_repair",
                                e.review_messages(row, row["candidate_target"]), p.check_review)
            verdict = receipt["parsed"]
            state = ("accepted" if verdict["label_supported"] and verdict["decision_appropriate"]
                     and not verdict["issues"] else "quarantined")
            self.ledger.save_job(row["id"], fingerprint, state, dict(
                review=receipt, candidate_target_sha256=p.sha(p.compact(row["candidate_target"])),
                source_position=row["source_position"], source_id=row["source_id"],
                independent_human_gold=False))
            return state
        except p.BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending", dict(reason=self.reason or "budget"))
            self.halt(self.reason or "budget")
            return "pending"
        except p.APIError as error:
            self.ledger.save_job(row["id"], fingerprint, "pending", dict(error=error.record()))
            return "failed"

    def export_reviews(self):
        results = []
        for row in self.rows:
            job = self.ledger.job(row["id"])
            if job:
                results.append(dict(id=row["id"], position=row["source_position"],
                                    state=job[1], result=job[2]))
        p.atomic_json(self.out / "reviews.json", results)
        with (self.out / "api_receipts.jsonl").open("w") as stream:
            stream.writelines(p.compact(r) + "\n" for r in self.ledger.receipts())


def run(out):
    verify(out)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not os.environ.get("BIGMODEL_API_KEY"):
        os.environ["BIGMODEL_API_KEY"] = read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    tasks = list(p.json_rows(out / "requests.jsonl"))
    with p.exclusive_run(out):
        runner = Runner(out, tasks, settings())
        try:
            status = runner.run(limit=1)
            runner.export_reviews()
            first = runner.ledger.job(tasks[0]["id"])
            if not first or first[1] not in ("accepted", "quarantined") or runner.reason:
                return status
            p.atomic_json(out / "preflight.json", dict(
                valid_api_review=True, first_state=first[1],
                endpoint=settings().endpoint, returned_model=first[2]["review"]["returned_model"]))
            for _ in range(2):
                status = runner.run()
                runner.export_reviews()
                if runner.reason or not status["remaining_tasks"]:
                    break
            return status
        finally:
            runner.export_reviews()
            runner.ledger.close()


def candidate(out):
    plan = verify(out)
    assert not (out / "candidate").exists(), "candidate already exists"
    source = Path(plan["source"])
    originals = {row["id"]: row for row in p.json_rows(source / "annotation/train.jsonl")}
    previous = {c["id"]: c for c in read(REVIEW)["cases"]}
    tasks = list(p.json_rows(out / "requests.jsonl"))
    results = {r["id"]: r for r in read(out / "reviews.json")}
    changes = {r["source_id"]: r for r in tasks}
    output = []
    held = []
    for original_id, original in originals.items():
        if original_id in changes:
            continue
        row = dict(original)
        row["assistant_content_review"] = previous.get(original_id, {}).get("content_judgment", "pending")
        row["candidate_lineage"] = dict(origin="unchanged_source_export", source_id=original_id)
        output.append(row)
    for task in tasks:
        result = results.get(task["id"])
        if not result or result["state"] != "accepted":
            held.append(dict(source_id=task["source_id"], position=task["source_position"],
                             result=result, state="unstarted" if result is None else result["state"]))
            continue
        receipt = result["result"]["review"]
        row = {key: value for key, value in task.items() if key not in (
            "teacher_messages", "candidate_target", "original_label_receipt")}
        row.update(target=p.compact(task["candidate_target"]),
                   user_prompt=task["teacher_messages"][1]["content"],
                   source_label=("assistant_repaired_GLM_reviewed_full20" if task["candidate_origin"] == "assistant_repair"
                                 else "retained_GLM_label_fresh_review"),
                   label_prompt_sha256=task["teacher_prompt_sha256"],
                   annotation_response_id=(None if task["candidate_origin"] == "assistant_repair" else
                                           task["original_label_receipt"]["response_id"]),
                   original_annotation_response_id=task["original_label_receipt"]["response_id"],
                   semantic_review_sampled=True, semantic_review=receipt["parsed"],
                   repair_review_response_id=receipt["response_id"],
                   assistant_content_review="pending_post_GLM",
                   candidate_lineage=dict(origin=task["candidate_origin"], source_id=task["source_id"],
                                          source_position=task["source_position"]))
        # A teacher review is not an assistant post-review or training release.
        output.append(row)
    assert len(output) == len({r["id"] for r in output})
    assert len(output) == len({p.normalized_question(r["question"]) for r in output})
    for row in output:
        p.check_target(json.loads(row["target"]), row)
    dest = out / "candidate"
    dest.mkdir()
    with (dest / "train.jsonl").open("x") as stream:
        stream.writelines(p.compact(r) + "\n" for r in output)
    p.atomic_json(dest / "held.json", held)
    manifest = dict(
        version=VERSION, rows=len(output), repaired_review_pass=sum(
            r.get("assistant_content_review") == "pending_post_GLM" for r in output),
        unchanged_source_rows=sum(r["candidate_lineage"]["origin"] == "unchanged_source_export" for r in output),
        assistant_review_states=dict(Counter(r["assistant_content_review"] for r in output)),
        kind_counts=dict(Counter(r["targeted_kind"] for r in output)),
        held=len(held), train_sha256=p.file_sha(dest / "train.jsonl"),
        ready_for_training=False, assistant_semantic_review_pending=True,
        original_data_modified=False, base_training_data_merged=False,
        training_started=False, production_modified=False)
    p.atomic_json(dest / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "launch", "run", "candidate", "status"))
    parser.add_argument("--source", type=Path, default=ROOT / SOURCE_NAME)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        result = prepare(args.source, out, args.tokenizer)
    elif args.command == "status":
        result = {name: read(out / name) for name in ("status.json", "preflight.json", "candidate/manifest.json")
                  if (out / name).exists()}
    elif args.command == "verify":
        result = verify(out)
    elif args.command == "candidate":
        result = candidate(out)
    elif args.command == "run":
        result = run(out)
    else:
        verify(out)
        assert not (out / "launch.json").exists(), "already launched"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--out", str(out)]
        with (out / "run.log").open("x") as stream:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        result = dict(pid=process.pid, max_api_calls=32, max_accounted_tokens=1_200_000,
                      review_only=True, no_training=True, original_data_modified=False)
        p.atomic_json(out / "launch.json", result)
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
