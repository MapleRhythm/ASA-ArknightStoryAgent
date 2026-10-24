"""Build an isolated corrected-label candidate; never promote, train, or edit source."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import review_reader_errors_individually as review
from reader_error_label_repairs_20261018 import REPAIRS

OUTPUT_NAME = "reader_errors_v3_20261018_assistant_repaired_candidate_v5"
MUTABLE_FIELDS = {
    "target", "annotation_model", "annotation_response_id", "old_labels_used",
    "semantic_review", "semantic_review_sampled", "source_label", "label_revision",
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_target(target, row):
    action = target["next_action"]
    if action == "answer_directly":
        assert set(target) == {"next_action", "supported_facts"}
        facts = target["supported_facts"]
        assert 1 <= len(facts) <= 8
        available = {d["evidence_id"] for d in row["documents"]}
        for item in facts:
            assert set(item) == {"fact", "evidence_ids"}
            assert isinstance(item["fact"], str) and item["fact"].strip()
            ids = item["evidence_ids"]
            assert 1 <= len(ids) <= 2 and len(ids) == len(set(ids))
            assert set(ids) <= available
    elif action == "retrieve_more":
        assert set(target) == {"next_action", "follow_up_hypothesis"}
        assert row["round"] == "1/2", "no retrieval on last round"
        hyp = target["follow_up_hypothesis"]
        required = {"question", "query_type", "entities", "keywords", "expected_answer_type"}
        assert required <= set(hyp) <= required | {"dialogue_context"}
        for name, value in hyp.items():
            if name in ("entities", "keywords"):
                assert isinstance(value, list) and value
                assert all(isinstance(s, str) and s.strip() for s in value)
            else:
                assert isinstance(value, str) and value.strip()
    else:
        raise AssertionError("corrections must retain supported parts or retrieve a missing cause")


def prepare():
    audit = review.audit()
    rows = review.load()
    original = copy.deepcopy(rows)
    repairs = sorted(REPAIRS, key=lambda c: c["position"])
    assert len(repairs) == len({c["id"] for c in repairs})
    assert [c["position"] for c in repairs] == audit["failed_positions"]
    history = json.loads((review.ROOT / review.PREVIOUS).read_text())["cases"]
    current = json.loads((review.ROOT / review.REVIEW).read_text())["cases"]
    decisions = {c["id"]: c for c in history + current}
    changes = []
    for correction in repairs:
        position = correction["position"]
        row = rows[position - 1]
        assert row["id"] == correction["id"], "repair position/id mismatch"
        decision = decisions[row["id"]]
        assert decision["judgment"] == "fail", "cannot repair pending or ambiguous labels"
        new_target = correction["target"]
        validate_target(new_target, row)
        before = json.loads(row["target"])
        assert before != new_target
        changes.append({
            "position": position, "id": row["id"], "question": row["question"],
            "round": row["round"], "targeted_kind": row["targeted_kind"],
            "original_target": before, "revised_target": copy.deepcopy(new_target),
            "original_target_sha256": digest(row["target"].encode()),
            "error_tags": decision["error_tags"],
            "original_review_note": decision["note"],
            "repair_direction": decision["repair_direction"],
            "revision_status": "assistant_repaired_pending_independent_review",
            "not_independent_gold": True,
        })
        row["label_revision"] = {
            "author": "codex_assistant", "date": "2026-10-18",
            "source_sha256": review.SOURCE_SHA,
            "original_target_sha256": digest(row["target"].encode()),
            "original_metadata": {
                k: copy.deepcopy(row[k]) for k in sorted(MUTABLE_FIELDS - {"target", "label_revision"})
                if k in row
            },
            "status": "assistant_repaired_pending_independent_review",
            "ready_for_training": False,
        }
        row["target"] = json.dumps(new_target, ensure_ascii=False, separators=(",", ":"))
        row["annotation_model"] = "codex_assistant_evidence_repair"
        row["annotation_response_id"] = None
        row["old_labels_used"] = True
        row["source_label"] = "assistant_evidence_repair_20261018"
        row["semantic_review"] = {
            "status": "pending_independent_review",
            "label_supported": None, "decision_appropriate": None,
            "not_independent_gold": True,
        }
        row["semantic_review_sampled"] = False
    repaired_ids = {c["id"] for c in repairs}
    for before, after in zip(original, rows):
        if before["id"] not in repaired_ids:
            assert before == after, "unapproved record changed"
        else:
            assert {k: v for k, v in before.items() if k not in MUTABLE_FIELDS} == {
                k: v for k, v in after.items() if k not in MUTABLE_FIELDS
            }, "evidence, prompt, question, or identity changed"
    train = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    change_bytes = json_bytes(changes)
    manifest = {
        "kind": "assistant_corrected_labels_isolated_candidate",
        "source_sha256": review.SOURCE_SHA,
        "source_rows": len(rows), "repaired": len(repairs),
        "unchanged": len(rows) - len(repairs),
        "original_labels_reviewed": audit["reviewed"],
        "original_label_judgments": audit["judgments"],
        "original_labels_pending": audit["pending"],
        "repaired_positions": [c["position"] for c in repairs],
        "repairs_sha256": review.sha(review.ROOT / "reader_error_label_repairs_20261018.py"),
        "review_sha256": review.sha(review.ROOT / review.REVIEW),
        "history_sha256": review.PREVIOUS_SHA,
        "train_sha256": digest(train),
        "changes_sha256": digest(change_bytes),
        "ready_for_training": False, "candidate_data_only": True,
        "all486_semantically_reviewed": False,
        "all_retained_labels_semantically_certified": False,
        "revised_labels_independently_reviewed": False,
        "production_modified": False,
        "running_training_dataset_changed": False,
        "teacher_api_called": False,
        "not_human_gold": True,
        "scope": "486 new rows only; not the 1236-row merged candidate; not selected by a trainer",
        "protocol_note": "Partial support uses answer_directly with supported facts and explicit unknowns; no unsupported hybrid action or synthetic reasoning chain.",
    }
    return {
        "train.jsonl": train,
        "changes.json": change_bytes,
        "manifest.json": json_bytes(manifest),
    }


def build(output):
    files = prepare()
    assert not output.exists(), "output exists; use --verify, do not overwrite"
    output.mkdir(parents=False)
    for name, data in files.items():
        (output / name).write_bytes(data)
    verify(output)


def verify(output):
    expected = prepare()
    assert {p.name for p in output.iterdir()} == set(expected)
    for name, data in expected.items():
        assert (output / name).read_bytes() == data, f"candidate mismatch: {name}"
    return json.loads(expected["manifest.json"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert args.build != args.verify, "choose --build or --verify"
    destination = review.ROOT / OUTPUT_NAME
    if args.build:
        build(destination)
    print(json.dumps(verify(destination), ensure_ascii=False, indent=2))
