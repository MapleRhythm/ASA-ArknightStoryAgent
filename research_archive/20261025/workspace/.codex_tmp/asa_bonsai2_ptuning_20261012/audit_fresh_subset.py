"""Create/verify a hash-bound accepted subset. Offline checks, never answer filters."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from annotation_protocol import check_row, check_target, normalized_question, prompt_for


VERSION = "reviewed_GLM_accepted_subset_v1"
SOURCE_FILES = ("manifest.json", "train.jsonl", "val.jsonl", "api_receipts.jsonl",
                "status.json", "quarantine.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def inspect_source(source):
    manifest = json.loads((source / "manifest.json").read_text())
    require(manifest["kind"] == "fresh_GLM_full20_annotation", "wrong source kind")
    require(manifest["run_status"] == "batch_finished", "batch still running")
    status = json.loads((source / "status.json").read_text())
    quarantine = json.loads((source / "quarantine.json").read_text())
    require(status["stage"] == "batch_finished" and status["remaining_tasks"] == 0,
            "source not fully processed")
    require(status["total_tasks"] == manifest["planned_train"] + manifest["planned_val"] and
            status["accepted"] == manifest["labelled_examples"] and
            status["accepted"] + status["quarantined"] == status["total_tasks"],
            "processed status count mismatch")
    require(len(quarantine) == status["quarantined"], "source quarantine count mismatch")
    require(not manifest["old_labels_used"] and manifest["new_api_calls"] > 0, "old labels")
    require(manifest["teacher_endpoint"] ==
            "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions", "wrong endpoint")
    require(manifest["evidence_count_per_example"] == 20 and
            manifest["truncated_documents"] == 0, "source truncation")
    receipts = read_rows(source / "api_receipts.jsonl")
    require(len(receipts) == manifest["new_api_calls"] == status["api_calls"], "call count mismatch")
    accepted = {}
    for receipt in receipts:
        if receipt.get("accepted") and receipt.get("phase") == "annotate":
            key = (receipt["task_id"], receipt["response_id"])
            require(key not in accepted, "duplicate response")
            accepted[key] = receipt
    rows = {}
    seen = set()
    for split in ("train", "val"):
        require(file_sha(source / f"{split}.jsonl") == manifest["data_sha256"][split],
                f"source hash mismatch: {split}")
        rows[split] = read_rows(source / f"{split}.jsonl")
        require(len(rows[split]) == manifest[split], "source row count mismatch")
        for row in rows[split]:
            require(row["id"] not in seen, "duplicate sample id")
            seen.add(row["id"])
            require(row["split"] == split and not row["old_labels_used"], "bad row lineage")
            require(row["annotation_model"] == manifest["teacher_model"] and
                    row["source_label"] == "fresh_GLM_API_full20", "wrong teacher")
            messages, prompt = prompt_for(row["question"], row["round"], row["documents"])
            check_row(dict(row, teacher_messages=messages))
            require(row["user_prompt"] == messages[1]["content"] and row["prompt"] == prompt,
                    "user evidence prompt mismatch")
            require(row["label_prompt_sha256"] == row["teacher_prompt_sha256"],
                    "label prompt lineage mismatch")
            target = check_target(json.loads(row["target"]), row)
            receipt = accepted.get((row["id"], row["annotation_response_id"]))
            require(receipt is not None, "missing accepted API receipt")
            require(receipt["parsed"] == target and json.loads(receipt["content"]) == target,
                    "teacher response does not match target")
            require(receipt["teacher_prompt_sha256"] == row["teacher_prompt_sha256"] and
                    receipt["endpoint"] == manifest["teacher_endpoint"] and
                    receipt["requested_model"] == manifest["teacher_model"], "receipt lineage mismatch")
    require(sum(map(len, rows.values())) == manifest["labelled_examples"], "accepted count mismatch")
    overlaps = {}
    for field, extractor in (
        ("source_family", lambda r: r["source_families"]),
        ("document", lambda r: [d["document_id"] for d in r["documents"]]),
        ("question", lambda r: [normalized_question(r["question"])]),
    ):
        values = {s: {v for row in rows[s] for v in extractor(row)} for s in rows}
        overlaps[field] = sorted(values["train"] & values["val"])
        require(not overlaps[field], f"train/val {field} leakage")
    return manifest, rows, overlaps


def build(source, decisions, out):
    require(not out.exists(), "refusing to overwrite dataset")
    manifest, rows, overlaps = inspect_source(source)
    review = json.loads(decisions.read_text())
    exclusions = review["excluded"]
    retained_checks = review["additional_retained_spot_checks"]
    all_rows = {row["id"]: row for values in rows.values() for row in values}
    require(set(exclusions) <= set(all_rows), "unknown exclusion id")
    require(set(retained_checks) <= set(all_rows), "unknown review id")
    require(not set(exclusions) & set(retained_checks), "conflicting review decisions")
    require(review["labels_manually_rewritten"] == 0 and review["not_QA_accuracy"],
            "invalid review scope")
    require(all(isinstance(reason, str) and reason.strip() for reason in exclusions.values()),
            "missing exclusion reason")
    out.mkdir()
    (out / "source").mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(source / name, out / "source" / name)
    shutil.copy2(decisions, out / "review.json")
    selected = {s: [r for r in values if r["id"] not in exclusions] for s, values in rows.items()}
    for split, values in selected.items():
        # Preserve complete row payloads and original order. Never repair teacher labels.
        with (out / f"{split}.jsonl").open("x") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (out / "excluded.jsonl").open("x") as stream:
        for row_id, reason in exclusions.items():
            stream.write(json.dumps({"id": row_id, "reason": reason,
                                     "source_row": all_rows[row_id]}, ensure_ascii=False) + "\n")
    audit = {
        "version": VERSION, "offline_structural_and_lineage_checks_passed": True,
        "source_complete": manifest["complete"], "source_run_status": manifest["run_status"],
        "source_planned": manifest["planned_train"] + manifest["planned_val"],
        "source_accepted": manifest["labelled_examples"],
        "source_not_accepted": manifest["planned_train"] + manifest["planned_val"] -
                               manifest["labelled_examples"],
        "source_file_sha256": {name: file_sha(source / name) for name in SOURCE_FILES},
        "review_sha256": file_sha(out / "review.json"),
        "excluded_sha256": file_sha(out / "excluded.jsonl"),
        "additional_excluded": len(exclusions),
        "assistant_spot_checked_targets": len(exclusions) + len(retained_checks),
        "review_scope": review["scope"],
        "all_retained_labels_semantically_certified": False,
        "data_sha256": {s: file_sha(out / f"{s}.jsonl") for s in selected},
        "selected_ids": {s: [r["id"] for r in selected[s]] for s in selected},
        "split_overlap": overlaps,
        "protocol_sha256": file_sha(Path(__file__).with_name("annotation_protocol.py")),
        "auditor_sha256": file_sha(__file__),
        "not_QA_accuracy": True,
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    reviewed_manifest = {
        "kind": VERSION, "ready_for_training": True, "accepted_subset": True,
        "source_complete": manifest["complete"], "source_batch_finished": True,
        "old_labels_used": False, "new_api_calls": manifest["new_api_calls"],
        "review_required_before_training": False,
        "review_scope": review["scope"], "not_human_gold": True, "not_unseen_benchmark": True,
        "not_full_chain_retrieval_evaluation": True,
        "all_retained_labels_semantically_certified": False,
        "evidence_count_per_example": 20, "truncated_documents": 0,
        "train_val_story_overlap": [], "train_val_source_family_overlap": [],
        "train_val_document_overlap": [], "train_val_question_overlap": [],
        "train": len(selected["train"]), "val": len(selected["val"]),
        "data_sha256": audit["data_sha256"], "audit_sha256": file_sha(out / "audit.json"),
        "action_counts": {s: dict(Counter(json.loads(r["target"])["next_action"] for r in values))
                          for s, values in selected.items()},
        "source_accepted": manifest["labelled_examples"], "additional_excluded": len(exclusions),
        "production_modified": False, "no_QA_accuracy_claim": True,
    }
    (out / "manifest.json").write_text(json.dumps(reviewed_manifest, ensure_ascii=False, indent=2))
    return verify(out)


def verify(data):
    manifest = json.loads((data / "manifest.json").read_text())
    require(manifest["kind"] == VERSION and manifest["accepted_subset"] and
            manifest["ready_for_training"], "not a reviewed accepted subset")
    require(file_sha(data / "audit.json") == manifest["audit_sha256"], "audit tampered")
    audit = json.loads((data / "audit.json").read_text())
    require(audit["auditor_sha256"] == file_sha(__file__), "auditor version changed")
    require(audit["protocol_sha256"] == file_sha(Path(__file__).with_name("annotation_protocol.py")),
            "protocol version changed")
    for name, digest in audit["source_file_sha256"].items():
        require(file_sha(data / "source" / name) == digest, f"source copy changed: {name}")
    require(file_sha(data / "review.json") == audit["review_sha256"], "review changed")
    require(file_sha(data / "excluded.jsonl") == audit["excluded_sha256"], "exclusions changed")
    source_manifest, original, overlaps = inspect_source(data / "source")
    review = json.loads((data / "review.json").read_text())
    require(manifest["source_complete"] == source_manifest["complete"] == audit["source_complete"],
            "source complete status misrepresented")
    require(manifest["old_labels_used"] is False and manifest["evidence_count_per_example"] == 20 and
            manifest["truncated_documents"] == 0 and not manifest["review_required_before_training"],
            "unsafe manifest")
    require(not manifest["all_retained_labels_semantically_certified"], "overclaimed review")
    require(audit["split_overlap"] == overlaps, "overlap receipt mismatch")
    excluded_rows = read_rows(data / "excluded.jsonl")
    originals = {r["id"]: r for values in original.values() for r in values}
    require({r["id"] for r in excluded_rows} == set(review["excluded"]), "exclusion set mismatch")
    require(len(excluded_rows) == len(review["excluded"]) == manifest["additional_excluded"],
            "exclusion count mismatch")
    for item in excluded_rows:
        require(item["source_row"] == originals[item["id"]] and
                item["reason"] == review["excluded"][item["id"]], "exclusion payload changed")
    for split, values in original.items():
        require(file_sha(data / f"{split}.jsonl") == manifest["data_sha256"][split] ==
                audit["data_sha256"][split], "selected data changed")
        rows = read_rows(data / f"{split}.jsonl")
        expected = [r for r in values if r["id"] not in review["excluded"]]
        require(rows == expected and len(rows) == manifest[split], "not unchanged accepted subset")
        require([r["id"] for r in rows] == audit["selected_ids"][split], "selected id mismatch")
        require(manifest["action_counts"][split] ==
                dict(Counter(json.loads(r["target"])["next_action"] for r in rows)),
                "action count mismatch")
    return {"train": manifest["train"], "val": manifest["val"],
            "excluded": manifest["additional_excluded"], "audit_sha256": manifest["audit_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
    else:
        result = build(args.source, args.review, args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
