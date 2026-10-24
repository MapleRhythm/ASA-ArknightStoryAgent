"""Inspect every real pointer output; never repair/rehydrate rejected payloads."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha, digest
from verify_empty_quote_candidate_v1 import save
from quote_pointer_protocol_v1 import strict_loads, validate_catalog, bind


def main():
    destination = HERE / "quote_pointer8b58_failure_diagnosis_v1.json"
    assert not destination.exists()
    matrixpath = HERE / "quote_pointer8b58_v1.matrix.json"
    analysispath = HERE / "quote_pointer8b58_validation_v1.json"
    matrix, report = load(matrixpath), load(analysispath)
    assert report["matrix_sha256"] == sha(matrixpath)
    rows = []
    raw_actions, ref_defects, raw_fact_counts, warnings = Counter(), Counter(), Counter(), Counter()
    total_refs = 0
    for index, (source, result) in enumerate(zip(matrix["rows"], report["results"])):
        assert source["id"] == result["id"]
        value = strict_loads(result["raw_output"])
        assert isinstance(value, dict)
        validate_catalog(source["evidence_brief"], source["quote_catalog"])
        lookup = {q["quote_id"]: q for q in source["quote_catalog"]}
        issues, refs = [], []
        raw_actions[value.get("next_action")] += 1
        raw_fact_counts[len(value.get("supported_facts", []))] += 1
        for kind in ("supported_facts", "inferred_facts"):
            for fi, fact in enumerate(value.get(kind, [])):
                if not fact.get("evidence_refs"):
                    issues.append({"kind": "empty_refs", "field": kind, "fact_index": fi})
                for ri, ref in enumerate(fact.get("evidence_refs", [])):
                    total_refs += 1
                    q = lookup.get(ref.get("quote_id"))
                    record = {"field": kind, "fact_index": fi, "ref_index": ri, "raw_ref": ref}
                    if set(ref) != {"evidence_id", "quote_id"}:
                        record["kind"] = "unexpected_ref_fields"
                    elif q is None:
                        record["kind"] = "unknown_quote_id"
                    elif ref["evidence_id"] != q["evidence_id"]:
                        record.update(kind="cross_evidence", catalog_evidence_id=q["evidence_id"])
                    else:
                        record.update(kind="valid_pointer", quote_sha256=digest(q["quote"]))
                    refs.append(record)
                    ref_defects[record["kind"]] += 1
        expected_error = None
        try:
            expanded = bind(value, source["evidence_brief"], source["quote_catalog"])
        except Exception as exc:
            expected_error = f"{type(exc).__name__}: {exc}"
            assert result["status"] != "ok" and result["error"] == expected_error
        else:
            assert expanded == result["expanded_payload"] and result["status"] == "ok"
        for warning in result.get("grounded", {}).get("grounding_warnings", []):
            if result.get("grounded", {}).get("next_action") != "answer_directly":
                warnings[warning.split(":", 1)[0]] += 1
        rows.append({
            "index": index, "id": result["id"], "question": result["question"],
            "raw_output_sha256": digest(result["raw_output"]), "raw_action": value.get("next_action"),
            "recorded_status": result["status"], "recorded_error": expected_error,
            "all_ref_inspections": refs, "additional_fact_issues": issues,
            "grounded_action": result.get("grounded", {}).get("next_action"),
        })
    assert len(rows) == 58
    summary = {
        "states": 58, "raw_actions": dict(raw_actions), "supported_fact_count_distribution": dict(raw_fact_counts),
        "all_ref_occurrences": total_refs, "ref_inspection_counts": dict(ref_defects),
        "first_recorded_errors": dict(Counter(r["recorded_error"] for r in rows if r["recorded_error"])),
        "states_with_additional_empty_fact_refs": sum(bool(r["additional_fact_issues"]) for r in rows),
        "nondirect_grounding_warning_occurrences": dict(warnings),
    }
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "summary": summary, "results": rows,
        "rejected_outputs_repaired": False, "new_generations_or_audit_calls": 0,
        "source_artifacts_sha256": {str(p): sha(p) for p in [matrixpath, analysispath, Path(__file__),
                                                            HERE / "quote_pointer_protocol_v1.py"]},
        "limits": [
            "Reference defect counts are occurrences, not independent questions or semantic errors.",
            "Correct pointer does not imply the associated fact follows from that quote.",
            "Empty-ref issues may be hidden by an earlier cross-E exception; full58 denominator retained.",
        ],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
