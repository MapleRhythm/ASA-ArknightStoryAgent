"""Read-only revalidation of the candidate merge; never promotes it to training."""
from collections import Counter
import argparse
import json
from pathlib import Path

import annotation_protocol as p
import expand_reader_errors_v3 as e
from audit_fresh_subset import read_rows, require, verify


def audit(out):
    plan = e.verify_plan(out)
    merged = out / "merged_candidate"
    manifest = json.loads((merged / "manifest.json").read_text())
    require(manifest["kind"] == e.VERSION and manifest["candidate_data_only"]
            and not manifest["ready_for_training"]
            and manifest["review_required_before_training"], "candidate readiness overstated")
    require(manifest["expansion_plan_sha256"] == p.file_sha(out / "plan.json"), "plan changed")
    reviewed = out / "reviewed_new"
    source = reviewed / "source"
    reviewed_manifest = verify(reviewed)
    require(reviewed_manifest["audit_sha256"] == manifest["new_audit_sha256"], "audit changed")
    new = read_rows(reviewed / "train.jsonl")
    e.verify_reviews(source, new)
    base = Path(plan["base"])
    for split in ("train", "val"):
        require(p.file_sha(merged / f"{split}.jsonl") == manifest["data_sha256"][split],
                "candidate hash changed")
        expected = read_rows(base / f"{split}.jsonl") + (new if split == "train" else [])
        require(read_rows(merged / f"{split}.jsonl") == expected, "candidate rows not originals")
        require(len(expected) == manifest[split], "candidate count mismatch")
    require(p.file_sha(merged / "val.jsonl") == plan["base_data_sha256"]["val"], "fixed eval changed")
    counts = Counter(row["targeted_kind"] for row in new)
    require(counts == manifest["targeted_counts"] and len(new) == manifest["new_train"]
            and len(new) >= 350 and all(counts[k] >= 40 for k in e.KINDS),
            "retained coverage mismatch")
    train, val = (read_rows(merged / f"{s}.jsonl") for s in ("train", "val"))
    p.assert_split(dict(train=train, val=val))
    old_questions = {p.normalized_question(r["question"])
                     for r in read_rows(base / "train.jsonl")}
    questions = [p.normalized_question(r["question"]) for r in new]
    require(len(set(questions)) == len(questions) and not set(questions) & old_questions,
            "duplicate new questions")
    by_kind_action = {
        kind: dict(Counter(json.loads(r["target"])["next_action"]
                           for r in new if r["targeted_kind"] == kind))
        for kind in e.KINDS
    }
    return dict(candidate_audit_passed=True, train=len(train), new_train=len(new), val=len(val),
                targeted_counts=dict(counts), actions_by_design_kind=by_kind_action,
                design_kind_is_not_action_gold=True, ready_for_training=False,
                assistant_spot_check_pending=True, semantic_gold_claim=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(p.compact(audit(args.out.resolve())))
