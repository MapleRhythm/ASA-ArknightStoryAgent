"""One-sample live preflight, then detach the remaining bounded batch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pipeline import Settings, atomic_json, file_sha, load_requests, verify_run_settings


def launch(args):
    out, config = Path(args.out).resolve(), Path(args.config).resolve()
    load_requests(out)
    settings = Settings(**json.loads(config.read_text())).validate()
    verify_run_settings(out, settings)
    # Pin both subprocesses to one immutable, credential-free settings snapshot.
    # Editing the shared config while a job is running cannot change its backend.
    snapshot = out / f"launch-config-{file_sha(config)}.json"
    if snapshot.exists():
        if json.loads(snapshot.read_text()) != vars(settings):
            raise ValueError("launch config snapshot changed")
    else:
        atomic_json(snapshot, vars(settings))
    before = json.loads((out / "manifest.json").read_text())
    accepted_before = before.get("labelled_examples", 0)
    if before.get("complete"):
        receipt = {"stage": "already_completed", "started_at_unix": time.time(),
                   "labelled_examples_after_preflight": accepted_before,
                   "config_snapshot": str(snapshot), "production_modified": False,
                   "GPU_used": False, "credentials_logged": False}
        atomic_json(out / "launch.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return
    pipeline = Path(__file__).with_name("pipeline.py").resolve()
    command = [sys.executable, "-u", str(pipeline), "run", "--out", str(out), "--config", str(snapshot)]
    stamp = time.time_ns()
    preflight_log = out / f"preflight-{stamp}.log"
    with preflight_log.open("x") as log:
        code = subprocess.run(command + ["--limit", "1"], stdout=log, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL).returncode
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    manifest = json.loads((out / "manifest.json").read_text())
    accepted = manifest.get("labelled_examples", 0)
    receipt = {"started_at_unix": time.time(), "preflight_exit_code": code,
               "preflight_log": str(preflight_log), "status": status,
               "config_snapshot": str(snapshot),
               "labelled_examples_before_preflight": accepted_before,
               "new_labels_in_preflight": accepted - accepted_before,
               "labelled_examples_after_preflight": accepted, "production_modified": False,
               "GPU_used": False, "credentials_logged": False}
    if code or (accepted <= accepted_before and not manifest.get("complete")):
        receipt["stage"] = "blocked"
        atomic_json(out / "launch.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        raise SystemExit(code or 2)
    if args.preflight_only or manifest.get("complete"):
        receipt["stage"] = "preflight_only" if args.preflight_only else "already_completed"
    else:
        worker_log = out / f"worker-{stamp}.log"
        with worker_log.open("x") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True,
                                    env=dict(os.environ))
        receipt.update(stage="running", worker_pid=proc.pid, worker_log=str(worker_log))
    atomic_json(out / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    with (out / ".launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another launcher is active for this dataset") from None
        previous = out / "launch.json"
        if previous.exists():
            receipt = json.loads(previous.read_text())
            pid = receipt.get("worker_pid")
            if receipt.get("stage") == "running" and isinstance(pid, int):
                try:
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    command = []
                if str(out).encode() in command and any(c.endswith(b"/pipeline.py") for c in command):
                    print(json.dumps(dict(receipt, already_running=True), ensure_ascii=False), flush=True)
                    return
        launch(args)


if __name__ == "__main__":
    main()
