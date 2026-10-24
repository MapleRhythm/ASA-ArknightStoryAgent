"""Isolated recovery of position472; original run/results remain untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import annotation_protocol as p
import glm_review_original_20261018 as original
from targeted_pt_v2 import read_credential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in original.load(args.source) if r["review_position"] == 472]
    assert len(rows) == 1
    status = json.loads((args.parent / "status.json").read_text())
    assert status["remaining_tasks"] == 1 and status["api_calls"] == 190
    reviews = json.loads((args.parent / "reviews.json").read_text())
    pending = [r for r in reviews if r["state"] == "pending"]
    assert len(pending) == 1 and pending[0]["id"] == rows[0]["id"]
    assert pending[0]["result"]["error"]["kind"] == "incomplete_output"
    if args.command == "launch":
        args.out.mkdir(mode=0o700)
        p.atomic_json(args.out / "manifest.json", {
            "parent": str(args.parent.resolve()), "position": 472,
            "source_sha256": original.SOURCE_SHA,
            "reason": "two prior finish_reason=length at 8192 output tokens",
            "max_output_tokens": 16384, "max_api_calls": 2,
            "max_total_tokens": 150000, "ready_for_training": False,
            "original_run_modified": False,
        })
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "run",
                   "--source", str(args.source.resolve()),
                   "--parent", str(args.parent.resolve()),
                   "--out", str(args.out.resolve())]
        with (args.out / "run.log").open("x") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(pid=proc.pid, command=command,
                       launched_at=p.datetime.now(p.timezone.utc).isoformat())
        p.atomic_json(args.out / "launch.json", receipt)
        print(p.compact(receipt))
        return
    if not os.environ.get("BIGMODEL_API_KEY"):
        os.environ["BIGMODEL_API_KEY"] = read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    settings = p.Settings(endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3",
                          workers=1, attempts=2, timeout=360,
                          max_output_tokens=16384, max_api_calls=2,
                          max_total_tokens=150000, review_fraction=1, seed=20261018)
    with p.exclusive_run(args.out):
        runner = original.Runner(args.out, rows, settings)
        try:
            status = runner.run()
            if status["remaining_tasks"]:
                runner.halt("length_recovery_pending_requires_inspection")
                p.atomic_json(args.out / "status.json", runner.progress("blocked"))
        finally:
            runner.export_reviews()
            runner.ledger.close()


if __name__ == "__main__":
    main()
