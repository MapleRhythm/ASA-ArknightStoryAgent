"""Keep source-review disagreements separate from unmodified API judgments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
analysis_path = HERE / "canonical_fixed79_validation_v1.json"
analysis = json.loads(analysis_path.read_text())
by_id = {row["id"]: row for row in analysis["results"]}
row_id = "0b3f0002117b06cb__teacher_state_grounded_action_sft::current_quote_fixed79"
row = by_id[row_id]
claim = "乔迪的父母来自阿戈尔，来自海洋的深处。"
correction = "你的父母，严格来说，是你的祖父母，来自阿戈尔，来自海洋的深处"
assert claim in row["grounded"]["answer"] and correction in row["evidence_brief"]
assert row["grounded"]["supported_facts"][1]["fact"] == claim
judgments = [
    json.loads(line) for line in (HERE / "semantic79_content_v2.progress.jsonl").read_text().splitlines()
    if line.strip()
]
judge = next(result for result in judgments if result["id"] == row_id)
assert judge["status"] == "ok" and judge["judgement"]["verdict"] == "supported_complete"
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "reviewer_type": "assistant_source_review_not_human_gold",
    "training_ready": False, "auto_judgments_modified": False,
    "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
    "reviewed_states": 1, "notes": [{
        "id": row_id, "question": row["question"], "shown_answer": row["grounded"]["answer"],
        "review_verdict": "unsupported", "review_confidence": "high",
        "disputed_span": claim, "supporting_correction_evidence_id": "E8",
        "verbatim_correction": correction,
        "reason": (
            "The source explicitly corrects parents to grandparents for the origin claim. "
            "The displayed answer asserts parents without quoting a character's loose usage or retaining the correction. "
            "Treating this as inherited colloquial nomenclature benevolently changes the asserted subject. "
            "The first maintenance/sacrifice claim has separate support in E9/E10; it is not the challenged claim."
        ),
        "auto_verdict": judge["judgement"]["verdict"],
        "auto_reason": judge["judgement"]["reason"],
        "failure_type": "explicit_correction_ignored_and_judge_rationalization",
    }],
    "limitations": [
        "One source-checked discrepancy is not a complete audit of all 79 states.",
        "Do not silently replace model labels or subtract one from a headline score and call it true accuracy.",
        "No training target, shown answer, protocol or model was modified.",
    ],
}
with (HERE / "source_review_notes_v1.json").open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
print(json.dumps({"reviewed_states": 1, "auto_review_disagreements": 1}))
