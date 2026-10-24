"""Explain actual changed fields; do not call all withheld samples incorrect."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(a.keys() | b.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in a:
                out.append({"path": path, "kind": "added", "after": b[key]})
            elif key not in b:
                out.append({"path": path, "kind": "removed", "before": a[key]})
            else:
                out.extend(differences(a[key], b[key], path))
        return out
    if a != b:
        return [{"path": prefix, "kind": "changed", "before": a, "after": b}]
    return []


def main():
    replay = json.loads((HERE / "stored_followup_replay_v1.json").read_text())
    builder = load_module("explain_followups", HERE / "build_soda_api_verifier_dataset.py")
    groups, reviews = {}, {}
    for path_string, digest in replay["source_sha256"].items():
        path = Path(path_string)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        if path.name == "audit_records.jsonl":
            groups.update(builder.load_audit_groups(path.parent))
        else:
            reviews.update(builder.load_existing_verifier(path))
    counts, rows = Counter(), []
    for entry in replay["dropped"]:
        key = entry["prompt_key"]
        verdict = reviews[key]["verifier"]
        candidates = []
        for role, tag in (("teacher", True), ("student", False)):
            record = builder.record_for_role(groups[key], kto_tag=tag)
            if record is None:
                continue
            strict = builder.strict_json_object(builder.response_text(record))
            parsed = builder.parse_jsonish(builder.response_text(record))
            if (parsed or {}).get("next_action") != "retrieve_more":
                continue
            candidate = {
                "role": role, "id": record["id"], "strict_json": strict is not None,
                "error": verdict.get(role + "_action_error"),
            }
            if strict is not None and verdict.get(role + "_action_error") == "none":
                normalized = builder.normalize_grounded_action_payload(
                    strict, question=builder.extract_question(builder.prompt_text(record)),
                    prompt=builder.prompt_text(record), force_action="retrieve_more",
                    missing_slots=verdict.get("missing_slots", []),
                )
                candidate["changes"] = differences(strict, normalized)
                counts.update(d["path"] + ":" + d["kind"] for d in candidate["changes"])
            candidates.append(candidate)
        rows.append({"prompt_key": key, "question": entry["question"], "original_candidates": candidates})
    report = {
        "training_ready": False, "dropped_states": len(rows),
        "changed_fields_in_approved_strict_candidates": dict(counts),
        "states": rows,
        "interpretation": [
            "Withheld means review provenance/target identity needs checking, not that the query is factually false.",
            "Structural rewrites are distinguished from query-term rewrites and missing reviewed candidates.",
            "No original records, outputs or verdicts are changed.",
        ],
    }
    with (HERE / "followup_drop_explanations_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "changed_fields": dict(counts), "examples": rows[:5],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
