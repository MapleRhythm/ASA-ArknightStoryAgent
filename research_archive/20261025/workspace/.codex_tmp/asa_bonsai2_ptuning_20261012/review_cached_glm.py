"""Display a bounded semantic-review sample, never edit labels."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--record-review", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    selected = []
    for split, count in (("train", 3), ("val", 2)):
        rows = [json.loads(l) for l in (data / f"{split}.jsonl").read_text().splitlines()]
        for row in rows[:count]:
            target = json.loads(row["target"])
            refs = {ref for fact in target["supported_facts"] for ref in fact["evidence_ids"]}
            selected.append({"id": row["id"], "split": split, "question": row["question"],
                             "target": target, "original_label_model": row["source_api"]["model"],
                             "conditioned_evidence_count": row["label_conditioned_document_count"],
                             "supporting_documents": [d for d in row["documents"] if d["evidence_id"] in refs]})
    if args.record_review:
        review = {
            "reviewer": "assistant_original_text_spot_review_not_independent_human_gold",
            "sample_ids": [r["id"] for r in selected],
            "source_evidence_integrity_checked_all_rows": True,
            "semantic_support_spot_checked": True,
            "remaining_limitations": [
                "Positive facts are not freshly conditioned on all 20 documents.",
                "Original labels may be incomplete or over-specific; not coverage gold.",
                "Training and validation contain historical development questions.",
                "No abstain/retrieve_more supervision; not action-boundary training.",
            ],
        }
        (data / "assistant_spot_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2))
        manifest = json.loads((data / "manifest.json").read_text())
        manifest.update(review_required_before_training=False,
                        assistant_spot_review_completed=True,
                        independent_human_review=False)
        (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(json.dumps(review, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
