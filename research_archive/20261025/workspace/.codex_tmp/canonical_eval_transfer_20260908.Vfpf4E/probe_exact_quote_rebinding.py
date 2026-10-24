"""Counterfactual exact-quote ID repair; never bypass semantic validation.

Only previously rejected answer proposals are considered. No fact or quote
text is changed, and a quote must occur verbatim in exactly one visible block.
All 79 rows remain; newly released answers require separate source review.
"""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys


def rebind(proposal, visible):
    updated = copy.deepcopy(proposal)
    edits = []
    for field in ("supported_facts", "inferred_facts"):
        for index, fact in enumerate(updated.get(field, [])):
            for ref_index, ref in enumerate(fact.get("evidence_refs", [])):
                quote = ref.get("quote")
                old = ref.get("evidence_id")
                if not isinstance(quote, str) or not 20 <= len(quote) <= 80:
                    continue
                if quote in visible.get(old, ""):
                    continue
                matches = [eid for eid, body in visible.items() if quote in body]
                if len(matches) != 1:
                    continue
                ref["evidence_id"] = matches[0]
                edits.append({
                    "field": field, "fact_index": index, "ref_index": ref_index,
                    "old_evidence_id": old, "new_evidence_id": matches[0], "quote_unchanged": quote,
                    "visible_exact_match_count": 1,
                })
    return updated, edits


def main():
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "after/src"))
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
    source_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
    matrix_path = run / "fixed79.current_quote.matrix.json"
    original = json.loads(source_path.read_text())
    matrix = {r["id"]: r for r in json.loads(matrix_path.read_text())["rows"]}
    rows = copy.deepcopy(original["results"])
    if len(rows) != 79 or {r["id"] for r in rows} != set(matrix):
        raise ValueError("expected fixed79 complete identity match")
    newly_answered = []
    changed_rows = []
    for row in rows:
        row["experimental_quote_rebinding"] = []
        if row.get("status") != "ok" or row["parsed"]["next_action"] != "answer_directly" or row["grounded"]["next_action"] == "answer_directly":
            continue
        visible = evidence_id_text_map_from_prompt(row["evidence_brief"])
        proposal, edits = rebind(row["parsed"], visible)
        if not edits:
            continue
        frozen = matrix[row["id"]]
        row["production_grounded_result"] = copy.deepcopy(row["grounded"])
        row["rebound_proposal"] = proposal
        row["experimental_quote_rebinding"] = edits
        changed_rows.append(row["id"])
        try:
            result = validate_conclusion_grounding(
                question=row["question"], hypothesis=HypothesisDocument(**frozen["hypothesis"]),
                evidence=copy.deepcopy(frozen["prompt_evidence"]),
                conclusion=ConclusionResult(**proposal),
                max_round_reached=row["current_round"] >= row["max_retrieval_rounds"],
                mode="quote", evidence_prompt_text=row["evidence_brief"],
            )
            row["grounded"] = dataclasses.asdict(result)
            if result.next_action == "answer_directly":
                newly_answered.append(row["id"])
        except Exception as exc:
            # Retain the actual original user-facing output on experiment error.
            row["experimental_revalidation_error"] = f"{type(exc).__name__}: {exc}"
    report = {
        "version": "exact_quote_rebinding_counterfactual_v1",
        "evaluation_only": True, "training_ready": False, "production_behavior_changed": False,
        "source_analysis_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": len(rows), "changed_binding_record_ids": changed_rows,
        "newly_answered_record_ids": newly_answered,
        "results": rows,
        "limitations": [
            "Counterfactual only; repaired answers are not returned to a production user.",
            "No facts or quote strings changed; only unique exact visible quote bindings of length 20-80 are repaired.",
            "A matching quote does not establish semantic entailment. All newly answered cases require source review.",
            "Do not count additional validator acceptance as improved answer correctness.",
        ],
    }
    output = run / "fixed79_configured_lora_ctx12288/exact_rebinding79.counterfactual.json"
    with output.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "rows_with_rebound_ids": len(changed_rows),
                      "newly_answered": len(newly_answered), "newly_answered_ids": newly_answered}))


if __name__ == "__main__":
    main()
