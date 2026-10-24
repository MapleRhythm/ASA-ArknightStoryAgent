"""Read-only views for evidence-based manual review; never repairs model output."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "native46_4b_sftkto_vs_bonsai_pt_20261014"
tasks = [json.loads(line) for line in (RUN / "tasks.jsonl").open()]
outputs = {}
for stem in ("fourb", "bonsai_shard0", "bonsai_shard1"):
    for line in (RUN / f"{stem}.results.jsonl").open():
        row = json.loads(line)
        outputs[row["arm"], row["id"]] = row

parser = argparse.ArgumentParser()
parser.add_argument("positions", help="comma-separated 1-based case numbers")
parser.add_argument("--outputs", action="store_true")
parser.add_argument("--docs", help="comma-separated E-IDs, or all")
parser.add_argument("--search", help="print complete docs containing any | separated terms")
parser.add_argument("--index", action="store_true")
args = parser.parse_args()
for position in map(int, args.positions.split(",")):
    task = tasks[position - 1]
    print(f"\nCASE {position} [{task['round']}] {task['question']}")
    if args.outputs:
        for arm in ("fourb", "bonsai"):
            row = outputs[arm, task["id"]]
            print(arm, row["protocol_prefill"] + row["raw_output"])
    for doc in task["documents"]:
        selected = args.docs == "all" or (
            args.docs and doc["evidence_id"] in args.docs.split(",")
        ) or (args.search and any(term in doc["text"] for term in args.search.split("|")))
        if selected:
            print(doc["evidence_id"], doc["document_id"], "\n" + doc["text"])
        elif args.index:
            print(doc["evidence_id"], doc["document_id"], repr(doc["text"][:85]))
