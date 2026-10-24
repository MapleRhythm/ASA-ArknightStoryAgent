"""Audit the complete cohort of proposed answers rejected by quote validation.

This is a diagnostic upper bound on same-evidence repair opportunities, not
user-visible accuracy and not permission to bypass the existing validator.
"""
import copy
import hashlib
import json
from pathlib import Path


run = Path(__file__).resolve().parent
source_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
source = json.loads(source_path.read_text())
source_rows = source["results"]
if len(source_rows) != 79 or len({r["id"] for r in source_rows}) != 79:
    raise ValueError("expected all 79 unique source rows")
selected = []
for row in source_rows:
    if (row.get("status") == "ok" and row["parsed"]["next_action"] == "answer_directly"
            and row["grounded"]["next_action"] != "answer_directly"):
        target = copy.deepcopy(row)
        target["production_grounded_result"] = target["grounded"]
        target["grounded"] = copy.deepcopy(target["parsed"])
        target["audit_target"] = "raw_proposed_answer_before_quote_validation_NOT_user_visible"
        selected.append(target)
if len(selected) != 34:
    raise ValueError(f"unexpected rejected cohort size: {len(selected)}")
report = {
    "version": "canonical_rejected_proposals34_v1",
    "evaluation_only": True, "training_ready": False,
    "audit_target": "raw_proposed_answer_before_quote_validation_NOT_user_visible",
    "source_analysis_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_record_count": 79, "selected_record_count": len(selected),
    "selection_rule": "parsed action answer_directly AND grounded action != answer_directly",
    "results": selected,
    "limitations": [
        "All 34 rejected proposed answers are included, not a favorable hand-picked subset.",
        "No rejected answer is released to the user or used for training.",
        "A supported raw answer with invalid quotes is only a potential repair opportunity, not a validated answer.",
        "Cohort proportions are not current 79-record or production accuracy.",
    ],
}
with (run / "fixed79_configured_lora_ctx12288/rejected34.raw_analysis.json").open("x") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_records": 79, "all_rejected_proposed_answers": len(selected),
                  "user_visible": False}))
