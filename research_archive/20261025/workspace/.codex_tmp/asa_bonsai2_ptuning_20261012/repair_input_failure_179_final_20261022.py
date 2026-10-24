"""Final minimal wording for 179; avoid ambiguous reciprocal-knowledge claim."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

import annotation_protocol as p
import repair_input_failure_data_20261022 as base

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "input_failure_repairs_followup_20261022_attempt01"
OUT = ROOT / "input_failure_repairs_179_final_20261022_attempt01"
BASE_SETTINGS = base.settings


def settings():
    value = BASE_SETTINGS()
    value.max_api_calls, value.max_total_tokens = 2, 100000
    return value


def prepare():
    assert not OUT.exists()
    row = copy.deepcopy(next(r for r in p.json_rows(PARENT / "requests.jsonl") if r["source_position"] == 179))
    row["candidate_target"]["supported_facts"][1]["fact"] = (
        "这次令和均编织的梦境中，那场对决没有真实发生；"
        "她们借此看清了巨兽的行动、招法、弱点与心理。")
    row["repair_reason"] += "；删除不必要且代词归属不确定的对方也看清她们"
    row["id"] = p.sha(p.compact(dict(
        version="input_failure_179_final_20261022", target=row["candidate_target"], source=row["source_id"])))[:24]
    p.check_row(row)
    p.check_target(row["candidate_target"], row)
    parent_plan = base.read(PARENT / "plan.json")
    OUT.mkdir(mode=0o700)
    (OUT / "requests.jsonl").write_text(p.compact(row) + "\n")
    plan = dict(
        version=base.VERSION, source=parent_plan["source"], tasks=1,
        inputs_sha256={str(path): p.file_sha(path) for path in (
            PARENT / "requests.jsonl", PARENT / "plan.json",
            Path(parent_plan["source"]) / "annotation/train.jsonl")},
        code_sha256={name: p.file_sha(ROOT / name) for name in (
            Path(__file__).name, "repair_input_failure_data_20261022.py",
            "annotation_protocol.py", "expand_input_failures_20261022.py")},
        requests_sha256=p.file_sha(OUT / "requests.jsonl"),
        settings=vars(settings()), ready_for_training=False, original_data_modified=False)
    p.atomic_json(OUT / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "launch", "run", "status"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    else:
        base.settings = settings
        if args.command == "run":
            result = base.run(OUT)
        elif args.command == "status":
            result = base.read(OUT / "status.json") if (OUT / "status.json").exists() else {}
        else:
            base.verify(OUT)
            assert not (OUT / "launch.json").exists()
            with (OUT / "run.log").open("x") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "run"],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, cwd=ROOT)
            result = dict(pid=process.pid, max_api_calls=2, max_accounted_tokens=100000,
                          ready_for_training=False)
            p.atomic_json(OUT / "launch.json", result)
    print(p.compact(result), flush=True)
