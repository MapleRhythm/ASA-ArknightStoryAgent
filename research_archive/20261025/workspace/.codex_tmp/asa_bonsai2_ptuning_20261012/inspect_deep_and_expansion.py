"""Read-only evidence views; no answer repair, teacher calls, or grading."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("deep", "expansion"))
    parser.add_argument("positions")
    parser.add_argument("--docs", help="all, cited, or comma-separated evidence IDs")
    parser.add_argument("--search", help="OR-separated literal search terms")
    parser.add_argument("--historical-support", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    if args.suite == "deep":
        tasks = rows(ROOT / "native46_bonsai_pt_v2_20261016/tasks.jsonl")
        outputs = {r["id"]: r for r in rows(
            ROOT / "ptuning_v2_deep_20261017_attempt01/generated_native46.jsonl")}
        old = json.loads((ROOT / "NATIVE46_V2_CONTENT_REVIEW_20261017.json").read_text())["cases"]
    else:
        tasks = rows(ROOT / "reader_errors_v3_20261017_formal01/spotcheck_requests.jsonl")
        outputs = {}
        old = []
    seen = set()
    for position in map(int, args.positions.split(",")):
        task = tasks[position - 1]
        label = outputs[task["id"]].get("parsed") if outputs else json.loads(task["target"])
        print(f"\nCASE {position} [{task['round']}] {task['id']} {task['question']}")
        print("LABEL", json.dumps(label, ensure_ascii=False))
        selected = set()
        if args.docs == "cited" and label:
            selected.update(e for f in label.get("supported_facts", []) for e in f["evidence_ids"])
        elif args.docs and args.docs != "all":
            selected.update(args.docs.split(","))
        if args.historical_support:
            selected.update(old[position - 1]["source_eids"])
        for doc in task["documents"]:
            eid = doc["evidence_id"]
            show = args.docs == "all" or eid in selected or (
                args.search and any(term in doc["text"] for term in args.search.split("|")))
            if show:
                key = doc.get("sha256", doc["document_id"])
                print("DOC", eid, doc["document_id"])
                if key in seen:
                    print("(identical complete document already printed in this invocation)")
                else:
                    print(doc["text"])
                    seen.add(key)
            elif args.index:
                print("INDEX", eid, doc["document_id"], repr(doc["text"][:80]))


if __name__ == "__main__":
    main()
