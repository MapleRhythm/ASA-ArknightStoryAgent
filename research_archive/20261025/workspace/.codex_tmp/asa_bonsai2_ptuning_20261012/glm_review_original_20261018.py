"""Review immutable labels only; reuse Coding Plan transport and durable ledger."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import annotation_protocol as p
from expand_reader_errors_v3 import review_messages
from targeted_pt_v2 import read_credential

SOURCE_SHA = "5fc6e331c9d02a24003c0aa89ea9b402c08f1b6907d3fb624bf31961f73a8ca4"
START, END = 301, 486
VERSION = "original_labels_glm_review_20261018_v1"


def load(source):
    assert p.file_sha(source) == SOURCE_SHA, "immutable source hash mismatch"
    rows = list(p.json_rows(source))
    assert len(rows) == len({r["id"] for r in rows}) == 486
    selected = []
    for position, original in enumerate(rows, 1):
        if START <= position <= END:
            row = dict(original, review_position=position)
            # Retained training exports omit the canonical teacher messages.
            # Reconstruct only this in-memory view; check_row verifies the
            # original prompt and original message hash before any API call.
            if "teacher_messages" not in row:
                row["teacher_messages"], _ = p.prompt_for(
                    row["question"], row["round"], row["documents"])
            p.check_row(row)
            p.check_target(json.loads(row["target"]), row)
            selected.append(row)
    assert len(selected) == 186
    return selected


class Runner(p.Runner):
    def fingerprint(self, row):
        return p.sha(p.compact({
            "version": VERSION, "source": SOURCE_SHA,
            "position": row["review_position"],
            "original_target_sha256": p.sha(row["target"]),
            "messages": review_messages(row, json.loads(row["target"])),
            "settings": self.settings.cache_settings(),
        }))

    def process(self, row):
        fingerprint = self.fingerprint(row)
        prior = self.ledger.job(row["id"])
        assert not prior or prior[0] == fingerprint, "changed review task"
        if prior and prior[1] in ("accepted", "quarantined"):
            return prior[1]
        try:
            receipt = self.call(row, "review_original",
                                review_messages(row, json.loads(row["target"])),
                                p.check_review)
            review = receipt["parsed"]
            state = ("accepted" if review["label_supported"]
                     and review["decision_appropriate"] and not review["issues"]
                     else "quarantined")
            self.ledger.save_job(row["id"], fingerprint, state, {
                "position": row["review_position"], "review": receipt,
                "original_target_sha256": p.sha(row["target"]),
                "independent_human_gold": False,
            })
            return state
        except p.BudgetReached:
            self.ledger.save_job(row["id"], fingerprint, "pending",
                                 {"reason": self.reason or "budget_limit"})
            self.halt(self.reason or "budget_limit")
            return "pending"
        except p.APIError as error:
            # An API/schema failure is not a semantic rejection of the label.
            self.ledger.save_job(row["id"], fingerprint, "pending",
                                 {"error": error.record()})
            return "failed"

    def export_reviews(self):
        results = []
        for row in self.rows:
            job = self.ledger.job(row["id"])
            if job:
                results.append(dict(id=row["id"], position=row["review_position"],
                                    state=job[1], result=job[2]))
        p.atomic_json(self.out / "reviews.json", results)
        with (self.out / "api_receipts.jsonl").open("w") as stream:
            for receipt in self.ledger.receipts():
                stream.write(p.compact(receipt) + "\n")


def execute(source, out):
    rows = load(source)
    settings = p.Settings(
        endpoint=p.CODING_PLAN_ENDPOINT, model="glm-5.3", workers=2,
        attempts=2, timeout=240, request_interval=1,
        max_api_calls=400, max_total_tokens=6_000_000,
        review_fraction=1, seed=20261018,
    )
    if not os.environ.get("BIGMODEL_API_KEY"):
        os.environ["BIGMODEL_API_KEY"] = read_credential(Path("${REMOTE_HOME}/.bigmodel_env"))
    with p.exclusive_run(out):
        runner = Runner(out, rows, settings)
        try:
            # A resumed run must still check the existing first result.
            runner.run(limit=1)
            runner.export_reviews()
            first = runner.ledger.job(rows[0]["id"])
            if not first or first[1] not in ("accepted", "quarantined"):
                runner.halt("preflight_no_valid_review")
                p.atomic_json(out / "status.json", runner.progress("blocked"))
                return
            if runner.reason:
                return
            p.atomic_json(out / "preflight.json", {
                "valid_api_review": True, "source_sha256": SOURCE_SHA,
                "position": START, "state": first[1],
                "endpoint": settings.endpoint, "model": settings.model,
            })
            # Run twice at most so transient pending failures can recover.
            for _ in range(2):
                status = runner.run()
                runner.export_reviews()
                if runner.reason or not status["remaining_tasks"]:
                    break
            if status["remaining_tasks"] and not runner.reason:
                runner.halt("pending_tasks_require_inspection")
                p.atomic_json(out / "status.json", runner.progress("blocked"))
        finally:
            runner.export_reviews()
            runner.ledger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "launch", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    rows = load(source)
    if args.command == "validate":
        print(p.compact({"rows": len(rows), "start": START, "end": END,
                         "source_sha256": SOURCE_SHA, "full20": True}))
    elif args.command == "launch":
        out.mkdir(mode=0o700)  # Refuse duplicate launch or existing run.
        p.atomic_json(out / "manifest.json", {
            "version": VERSION, "source": str(source), "source_sha256": SOURCE_SHA,
            "start": START, "end": END, "rows": len(rows),
            "source_modified": False, "ready_for_training": False,
            "independent_human_gold": False, "training_started": False,
        })
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "run",
                   "--source", str(source), "--out", str(out)]
        with (out / "run.log").open("x") as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(pid=proc.pid, out=str(out), source_sha256=SOURCE_SHA,
                       command=command, stage="launched",
                       launched_at=p.datetime.now(p.timezone.utc).isoformat())
        p.atomic_json(out / "launch.json", receipt)
        print(p.compact(receipt))
    else:
        execute(source, out)


if __name__ == "__main__":
    main()
