"""Create-only closure receipt for the interrupted DeepSeek experiment."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    run = Path(sys.argv[1])
    assert run == Path("${REMOTE_STORE}/exx_grounding_v1/experiments/asa_v58_deepseek_pilot_20260912")
    assert not (run / "run_summary.json").exists()
    assert not (run / "comparison.json").exists()
    assert not (run / "closed_interrupted.json").exists()
    # Never terminate anything; verify the exact old process is no longer present.
    try:
        os.kill(316146, 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Original PID exists; inspect before closing")
    events = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
    assert len(events) == 1 and events[0]["event"] == "request_started"
    log = (run / "api.log").read_text()
    assert "KeyboardInterrupt" in log and "sock.connect(sa)" in log
    pre = json.loads((run / "preflight.json").read_text())
    for file, expected in pre["bound_files"].items():
        assert sha(Path(file)) == expected, Path(file).name
    files = sorted(p for p in run.iterdir() if p.is_file())
    record = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "interrupted_no_answers_do_not_retry",
        "old_pid": 316146, "old_pid_absent": True,
        "request_started_markers": 1, "request_finished_markers": 0,
        "successful_answers": 0, "remaining_requests_not_started": 27,
        "last_recorded_stack": "TCP socket.connect; interrupted by operator",
        "no_recorded_http_response": True, "billing_status": "not verified",
        "reason": "Credential echoed by SSH PTY; process interrupted and credential retired from agent use",
        "credential_rotation_required": True,
        "source_secret_not_in_receipt": True,
        "full_glm_evaluation_launched": False,
        "production_changed": False,
        "artifact_sha256": {p.name: sha(p) for p in files},
    }
    with (run / "closed_interrupted.json").open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    archive = run / "interrupted_receipt.tgz"
    with tarfile.open(archive, "x:gz") as tar:
        for p in files + [run / "closed_interrupted.json"]:
            tar.add(p, arcname=p.name, recursive=False)
    print(json.dumps({"archive_sha256": sha(archive), "status": record["status"],
                      "successful_answers": 0, "archive_bytes": archive.stat().st_size}))


if __name__ == "__main__":
    main()
