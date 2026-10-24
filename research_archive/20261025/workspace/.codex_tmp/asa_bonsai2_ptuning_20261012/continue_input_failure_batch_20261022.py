"""One bounded continuation if unfinished question packs would block all labels.

Never alter the sealed worker/settings, retry terminal failures, or promote data.
Wait for its workflow lock, then only use already accepted generated questions.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import annotation_protocol as p
import expand_input_failures_20261022 as e


def allowed_pending(result):
    error = result.get("error") or {}
    return (not error.get("terminal") and error.get("kind") in
            {"incomplete_output", "invalid_json_or_protocol", "content_filtered"})


def inspect_generation(out):
    uri = (out / "questions/ledger.sqlite3").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        jobs = {key: (state, json.loads(value)) for key, state, value in
                conn.execute("SELECT task_id,state,result FROM jobs")}
    excluded = []
    accepted = 0
    for row in p.json_rows(out / "questions/requests.jsonl"):
        state, result = jobs.get(row["id"], ("unstarted", {}))
        if state == "accepted":
            accepted += 1
        else:
            if state != "pending" or not allowed_pending(result):
                raise RuntimeError("unexpected/terminal generation state; manual inspection required")
            excluded.append(dict(id=row["id"], state=state, reason=result))
    assert accepted > 0
    return accepted, excluded


def run(out):
    e.verify(out)
    with (out / "workflow.lock").open("a") as lock:
        # Detached orchestration only; never overlap the original writer/API worker.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        e.verify(out)
        original = json.loads((out / "workflow.json").read_text())
        if original["stage"] != "pending_inspection":
            return dict(stage="not_needed", original_stage=original["stage"])
        if original.get("error") != "question generation pending; inspect receipts before resume":
            raise RuntimeError("not the question-generation-only boundary; no automatic continuation")
        assert not (out / "annotation").exists(), "do not alter existing annotation tasks"
        accepted, excluded = inspect_generation(out)
        assert excluded, "nothing to salvage"
        p.atomic_json(out / "question_batch_exit_before_continuation.json", original)
        manifest = e.build_requests(out)
        manifest.update(
            accepted_generation_packs=accepted, excluded_generation_groups=excluded,
            partial_generation_batch=True, intended_question_ceiling=180,
            generated_question_ceiling=accepted * 3,
            continuation_code_sha256=p.file_sha(__file__))
        p.atomic_json(out / "annotation/manifest.json", manifest)
        status = dict(
            stage="annotation_preflight", pid=os.getpid(), started_unix=time.time(),
            accepted_generation_packs=accepted, excluded_generation_groups=len(excluded),
            partial_generation_batch=True, ready_for_training=False,
            auto_training_enabled=False, production_modified=False,
            running_training_dataset_changed=False, no_gpu_usage=True)
        p.atomic_json(out / "workflow.json", status)
        if not os.environ.get("BIGMODEL_API_KEY"):
            os.environ["BIGMODEL_API_KEY"] = e.old.previous.read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
        try:
            first = e.stage(out, "annotation", limit=1)
            if not first["accepted"] and not first["quarantined"]:
                raise RuntimeError("annotation preflight pending; inspect before resuming")
            status["stage"] = "annotation_and_all_label_review"
            p.atomic_json(out / "workflow.json", status)
            for _ in range(2):
                result = e.stage(out, "annotation")
                if not result["remaining_tasks"] or result["stop_reason"]:
                    break
            status.update(
                stage="api_finished_awaiting_assistant_review" if not result["remaining_tasks"] else "pending_inspection",
                accepted=result["accepted"], quarantined=result["quarantined"],
                remaining=result["remaining_tasks"], updated_unix=time.time())
        except Exception as exc:
            status.update(stage="pending_inspection", error=str(exc), updated_unix=time.time())
            p.atomic_json(out / "workflow.json", status)
            raise
        p.atomic_json(out / "workflow.json", status)
        return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "run"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.command == "run":
        result = run(out)
    else:
        e.verify(out)
        assert not (out / "continuation_launch.json").exists(), "continuation already launched"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "run", "--out", str(out)]
        with (out / "continuation.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=e.ROOT)
        result = dict(
            pid=process.pid, continuation_code_sha256=p.file_sha(__file__),
            waits_for_primary_workflow_lock=True, no_extra_question_calls=True,
            same_annotation_budget=True, auto_training_enabled=False)
        p.atomic_json(out / "continuation_launch.json", result)
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
