"""Read a live annotation ledger without opening a writer or changing jobs."""
import argparse
import json
from pathlib import Path
import sqlite3


def inspect(out, limit=6, full_evidence=False, kind=None):
    directory = out / "annotation"
    requests = {r["id"]: r for r in (
        json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines())}
    database = directory / "ledger.sqlite3"
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        jobs = connection.execute(
            "SELECT task_id,state,result FROM jobs ORDER BY task_id").fetchall()
        result = []
        for identifier, state, value in jobs:
            value, row = json.loads(value), requests[identifier]
            if kind is not None and row["targeted_kind"] != kind:
                continue
            label = value.get("label", {}).get("parsed")
            if label is None:
                continue
            evidence_ids = {eid for fact in label.get("supported_facts", [])
                            for eid in fact.get("evidence_ids", [])}
            result.append(dict(
                id=identifier, state=state, question=row["question"], round=row["round"],
                kind=row["targeted_kind"], label=label,
                review=value.get("review", {}).get("parsed"),
                documents=row["documents"] if full_evidence else [
                    d for d in row["documents"] if d["evidence_id"] in evidence_ids],
                full_evidence_included=full_evidence,
            ))
        return result[:limit]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--full-evidence", action="store_true")
    parser.add_argument("--kind")
    args = parser.parse_args()
    print(json.dumps(inspect(args.out, args.limit, args.full_evidence, args.kind),
                     ensure_ascii=False, indent=2))
