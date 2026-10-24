"""Read-only views and coverage audit of assistant-reviewed expansion labels.

No teacher calls, source edits, label repairs, promotion, or training.
Passing integrity checks is not an independent semantic judgment.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"
SOURCE_SHA = "5fc6e331c9d02a24003c0aa89ea9b402c08f1b6907d3fb624bf31961f73a8ca4"
PREVIOUS = "READER_ERROR_V3_SPOTCHECK_20261017.json"
PREVIOUS_SHA = "ae491f9db50296cf9e331cb334e62983a0387286acc9280da9562386ef1b76bc"
REVIEW = "READER_ERROR_V3_INDIVIDUAL_REVIEW_20261017.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root=ROOT):
    source = root / SOURCE
    if not source.exists():
        # Remote uses the original retained dataset; hash must be identical.
        source = root / "reader_errors_v3_20261017_formal01/reviewed_new/train.jsonl"
    assert sha(source) == SOURCE_SHA, "source changed"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    assert len(rows) == len({r["id"] for r in rows}) == 486
    for row in rows:
        assert row["evidence_count"] == len(row["documents"]) == 20
        assert not row["truncated"]
        assert {d["evidence_id"] for d in row["documents"]} == {f"E{i}" for i in range(1, 21)}
        for doc in row["documents"]:
            assert hashlib.sha256(doc["text"].encode()).hexdigest() == doc["sha256"]
    return rows


def audit(root=ROOT):
    rows = load(root)
    by_id = {r["id"]: (i, r) for i, r in enumerate(rows, 1)}
    previous_path = root / PREVIOUS
    assert sha(previous_path) == PREVIOUS_SHA
    previous = json.loads(previous_path.read_text())
    sample_path = root / previous["sample"]
    assert sha(sample_path) == previous["sample_sha256"]
    previous_samples = {
        r["id"]: r for r in (
            json.loads(line) for line in sample_path.read_text().splitlines() if line.strip())
    }
    assert len(previous_samples) == len(previous["cases"]) == 15
    review = json.loads((root / REVIEW).read_text())
    assert review["source_sha256"] == SOURCE_SHA
    assert review["previous_review_sha256"] == PREVIOUS_SHA
    assert review["ready_for_training"] is False
    assert review["all486_semantically_reviewed"] is False
    candidate = root / "reader_errors_v3_20261017_formal01/merged_candidate/manifest.json"
    assert sha(candidate) == review["candidate_manifest_sha256"]
    assert json.loads(candidate.read_text())["ready_for_training"] is False
    decisions = {}
    for case in previous["cases"]:
        assert case["id"] in by_id and case["id"] not in decisions
        row = by_id[case["id"]][1]
        assert row["targeted_kind"] == case["targeted_kind"]
        for field in ("question", "round", "target", "documents", "targeted_kind"):
            assert row[field] == previous_samples[case["id"]][field], "previous sample mismatch"
        decisions[case["id"]] = case
    for case in review["cases"]:
        assert case["id"] not in decisions, "duplicate review"
        position, row = by_id[case["id"]]
        assert case["position"] == position
        assert case["targeted_kind"] == row["targeted_kind"]
        assert case["judgment"] in ("pass", "fail", "needs_review")
        assert case["note"] and case["read_eids"]
        eids = {d["evidence_id"] for d in row["documents"]}
        assert len(set(case["read_eids"])) == len(case["read_eids"])
        assert set(case["source_eids"]) <= set(case["read_eids"]) <= eids
        if case["read_scope"] == "all20_complete_passages":
            assert set(case["read_eids"]) == eids
        assert case["read_scope"] in ("all20_complete_passages", "relevant_complete_passages")
        assert bool(case["error_tags"]) == (case["judgment"] == "fail")
        assert case["subquestions_checked"]
        if case["judgment"] == "fail":
            assert case["repair_direction"]
        if case["judgment"] == "needs_review":
            assert case["resolution_needed"], "unresolved review needs an explicit resolution"
        decisions[case["id"]] = case
    counts = Counter(c["judgment"] for c in decisions.values())
    coverage = defaultdict(Counter)
    pending = []
    for position, row in enumerate(rows, 1):
        state = decisions.get(row["id"], {}).get("judgment", "pending")
        coverage[row["targeted_kind"]][state] += 1
        if state == "pending":
            pending.append(position)
    return {
        "integrity_passed": True,
        "source_sha256": SOURCE_SHA,
        "total": 486,
        "previous_reviewed": len(previous["cases"]),
        "new_reviewed": len(review["cases"]),
        "reviewed": len(decisions),
        "judgments": dict(counts),
        "pending": len(pending),
        "next_pending_positions": pending[:20],
        "by_kind": {k: dict(v) for k, v in coverage.items()},
        "failed_positions": sorted(by_id[k][0] for k, v in decisions.items() if v["judgment"] == "fail"),
        "ready_for_training": False,
        "all486_semantically_reviewed": len(decisions) == 486 and not counts["needs_review"],
        "caution": "Assistant review, not human gold. Integrity audit does not independently grade semantics; unreviewed rows remain pending.",
    }


def view(rows, packs, docs):
    selected = set(packs.split(","))
    groups = defaultdict(list)
    for position, row in enumerate(rows, 1):
        if row["source_row_id"] in selected:
            groups[row["source_row_id"]].append((position, row))
    assert set(groups) == selected
    for pack, items in groups.items():
        print("\nPACK", pack)
        for position, row in items:
            print("CASE", position, row["id"], row["targeted_kind"], row["round"])
            print("QUESTION", row["question"])
            print("LABEL", row["target"])
        first = items[0][1]
        assert all([d["sha256"] for d in r["documents"]] ==
                   [d["sha256"] for d in first["documents"]] for _, r in items)
        if docs == "cited":
            selected_eids = {eid for _, r in items
                             for f in json.loads(r["target"]).get("supported_facts", [])
                             for eid in f["evidence_ids"]}
        else:
            selected_eids = set(docs.split(","))
        for doc in first["documents"]:
            if docs == "all" or doc["evidence_id"] in selected_eids:
                print("DOC", doc["evidence_id"], doc["document_id"])
                print(doc["text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("view", "audit"))
    parser.add_argument("--packs")
    parser.add_argument("--docs", default="all")
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit(), ensure_ascii=False, indent=2))
    else:
        assert args.packs
        view(load(), args.packs, args.docs)
