"""Resume immutable v2 inputs with bounded pending-job recovery, not label repair."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import annotation_protocol as p
import targeted_pt_v2 as t
from audit_fresh_subset import require

RETRYABLE_KINDS = {"invalid_json_or_protocol", "network_or_timeout", "rate_limit", "http_error"}


def job_alive(pid, out):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return str(out).encode() in command and any(
        b"targeted_pt_v2" in item for item in command)


def quarantine_exhausted(stage_dir, expected_count):
    """Keep failed payloads, isolate only retryable API failures, never accept them."""
    with p.exclusive_run(stage_dir):
        ledger = p.Ledger(stage_dir / "ledger.sqlite3")
        try:
            rows = t.read_rows(stage_dir / "requests.jsonl")
            pending = []
            for row in rows:
                prior = ledger.job(row["id"])
                require(prior is not None, "unstarted job must not be silently excluded")
                if prior[1] == "pending":
                    error = prior[2].get("error", {})
                    require(error.get("kind") in RETRYABLE_KINDS
                            and error.get("retryable") is True and not error.get("terminal"),
                            "budget/authentication/unknown pending cannot be excluded")
                    pending.append((row["id"], prior))
            require(len(pending) == expected_count, "pending job count changed")
            for row_id, prior in pending:
                result = dict(prior[2], recovery_disposition="quarantined_after_three_recovery_passes",
                              no_target_accepted_or_rewritten=True)
                ledger.save_job(row_id, prior[0], "quarantined", result)
            return [row_id for row_id, _ in pending]
        finally:
            ledger.close()


def drain(out, stage, operation):
    attempts = []
    receipt = out / f"{stage}.recovery.json"
    for index in range(3):
        status = operation(out, 0)
        attempts.append(dict(pass_number=index + 1, status=status))
        p.atomic_json(receipt, dict(attempts=attempts, original_budgets_unchanged=True))
        if status["remaining_tasks"] == 0:
            return status
        require(not status.get("stop_reason"), "cannot retry terminal/budget stop")
    isolated = quarantine_exhausted(out / stage, status["remaining_tasks"])
    status = operation(out, 0)  # Refresh status and immutable accepted-only exports; no API for terminal jobs.
    p.atomic_json(receipt, dict(attempts=attempts, isolated_failed_job_ids=isolated,
                               final_status=status, original_budgets_unchanged=True))
    require(status["remaining_tasks"] == 0, "recovery did not finish")
    return status


def worker(out, gpu):
    question_operation, annotation_operation = t.questions, t.annotate
    t.questions = lambda path, limit: drain(path, "questions", question_operation)
    t.annotate = lambda path, limit: drain(path, "annotation", annotation_operation)
    t.workflow(out, gpu)


def launch(out, gpu):
    t.verify_plan(out)
    require(os.environ.get("BIGMODEL_API_KEY"), "missing GLM credential")
    with (out / ".recovery-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((out / "workflow.json").read_text())
        require(state["stage"] == "blocked_or_failed", "only resume a failed workflow")
        require(not job_alive(state.get("pid"), out), "workflow is already alive")
        prior_path = out / "recovery.launch.json"
        if prior_path.exists():
            previous = json.loads(prior_path.read_text())
            require(not job_alive(previous.get("pid"), out), "recovery is already alive")
        stamp = str(time.time_ns())
        p.atomic_json(out / f"workflow.before-recovery-{stamp}.json", state)
        log = out / f"recovery-{stamp}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                   "--out", str(out), "--gpu", str(gpu)]
        with log.open("x") as stream:
            proc = subprocess.Popen(command, cwd=Path(__file__).resolve().parent,
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    env=dict(os.environ), start_new_session=True)
        receipt = dict(pid=proc.pid, log=str(log), gpu=gpu, launched_unix=time.time(),
                       resume_script_sha256=p.file_sha(__file__),
                       immutable_workflow_sha256=p.file_sha(t.__file__),
                       original_budgets_unchanged=True, production_modified=False,
                       original_labels_unchanged=True)
        p.atomic_json(prior_path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("BIGMODEL_API_KEY"):
        os.environ["BIGMODEL_API_KEY"] = t.read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    result = (launch(args.out.resolve(), args.gpu) if args.command == "launch"
              else worker(args.out.resolve(), args.gpu))
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
