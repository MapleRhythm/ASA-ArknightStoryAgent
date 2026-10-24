"""Reapply unchanged v42 diagnostic to every carried/new staged raw output.

Post-hoc analysis, not a new gate or independent validation dataset. Includes
intermediate rejected primary calls and terminal replies with explicit scopes.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V45 = HERE.parent / "staged_round2_reader_20260910.v45"
V42 = HERE.parent / "named_subject_reference_diagnosis_20260910.v42"
sys.path[:0] = [str(V45), str(V42)]
import run_second_round_reader as probe
from subject_reference_diagnostic import diagnose
shared = probe.shared


def rows():
    pre45 = shared.load(V45 / "preflight.json")
    gen45 = shared.load(V45 / "generations.json")
    pre41 = shared.load(probe.V41 / "preflight.json")
    gen41 = shared.load(probe.V41 / "generations.json")
    primary = {
        (r["pair_id"], r["condition"]): r
        for r in shared.load(V45 / "primary_fallback_audit_v1.json")["rows"]
    }
    first_inputs = {(r["id"], r["condition"]): r for r in pre41["conditions"]}
    first_results = {(r["pair_id"], r["condition"]): r for r in gen41["attempts"]}
    for item, result in zip(pre45["records"], gen45["records"], strict=True):
        key = item["pair_id"], item["condition"]
        assert key == (result["pair_id"], result["condition"])
        base = {k: item[k] for k in ("pair_id", "condition", "id", "question")}
        if item["terminal_after_round1"]:
            first, source = first_results[key], first_inputs[(item["id"], item["condition"])]
            yield {
                **base, "request_kind": "carried_v41_round1", "raw": first["sanitized_raw_output"],
                "hypothesis": source["hypothesis"], "visible_evidence": source["visible_evidence"],
                "max_round_reached": False,
                "recorded_request_action": first["conclusion"]["next_action"],
                "recorded_request_action_scope": "unchanged_v41_terminal",
                "final_endpoint_action": result["conclusion"]["next_action"],
                "final_endpoint_answer": result["final_answer"],
            }
            continue
        assert result["status"] == "ok"
        for index, request in enumerate(result["requests"]):
            planned = item["planned_requests"][index]
            assert request["status"] == "ok" and request["kind"] == planned["kind"]
            yield {
                **base, "request_kind": request["kind"], "raw": request["sanitized_raw_output"],
                "hypothesis": item["hypothesis"], "visible_evidence": planned["visible_evidence"],
                "max_round_reached": True,
                "recorded_request_action": (
                    primary[key]["primary_action"] if index == 0 else result["conclusion"]["next_action"]
                ),
                "recorded_request_action_scope": (
                    "canonical_primary_before_optional_fallback" if index == 0 else "canonical_final_direct_fallback"
                ),
                "final_endpoint_action": result["conclusion"]["next_action"],
                "final_endpoint_answer": result["final_answer"],
            }


def prepare():
    assert not (HERE / "preflight.json").exists()
    pre45 = shared.load(V45 / "preflight.json")
    probe.reader.verify(pre45)
    post = shared.load(V45 / "postflight_v1.json")
    assert post["round2_canonical_replays"] == 14 and post["terminal_carry_unchanged"] == 10
    inputs = list(rows())
    assert len(inputs) == 10 + post["actual_new_llm_calls"]
    assert len({(r["pair_id"], r["condition"]) for r in inputs}) == 24
    paths = [
        Path(__file__), V42 / "subject_reference_diagnostic.py", V42 / "checkpoint_v42.json",
        V45 / "run_second_round_reader.py", V45 / "preflight.json", V45 / "generations.json",
        V45 / "postflight_v1.json", V45 / "primary_fallback_audit_v1.json",
        probe.V41 / "preflight.json", probe.V41 / "generations.json",
        probe.reader.ROOT / "indexes/arknights_story/operator_aliases.json",
        *sorted(probe.reader.SRC.rglob("*.py")),
    ]
    shared.save(HERE / "preflight.json", {
        "version": "unchanged_subject_diagnostic_staged_raw_outputs_v46",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs, "sha256": {str(p): shared.sha(p) for p in paths},
        "protocol": [
            "Unmodified v42 diagnose function and original alias catalog; no new threshold or subject rule.",
            "Include all10 carried first-round raws and every actual v45 primary/final-direct raw, not only displayed answers.",
            "One endpoint can contribute two new raw outputs; count request scopes separately, not independent questions.",
            "Existing request action comes from exact canonical replay; schema parsing for diagnosis does not revalidate actions.",
            "Presence anywhere in an E block is not relation support or speaker identity, especially in long multi-member chains.",
            "Post-hoc read-only diagnosis after observing a possible alias-presence false negative, not a blind confirmation test.",
            "No output mutation, diagnostic gate, new generation/retrieval/GPU, source or label changes, training or promotion.",
        ],
        "production_changed": False, "training_ready": False, "gpu_used": False,
    })
    print({"raw_outputs": len(inputs), "endpoints": 24, "new_model_calls": 0}, flush=True)


def run():
    pre = shared.load(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    probe.staged.imports()
    from asa_arknight_story_agent.inference.generation.conclusion_generation import normalize_minimal_conclusion_output, parse_conclusion_output
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    catalog = shared.load(probe.reader.ROOT / "indexes/arknights_story/operator_aliases.json")
    outputs = []
    for item in pre["inputs"]:
        original = deepcopy(item)
        output = {k: item[k] for k in (
            "pair_id", "condition", "id", "request_kind", "recorded_request_action",
            "recorded_request_action_scope", "final_endpoint_action", "final_endpoint_answer",
        )}
        try:
            normalized = normalize_minimal_conclusion_output(item["raw"], "minimal")
            parsed = parse_conclusion_output(
                normalized, question=item["question"], current_hypothesis=HypothesisDocument(**item["hypothesis"]),
                max_round_reached=item["max_round_reached"],
            )
            diagnostics = []
            for group in ("supported_facts", "inferred_facts"):
                diagnostics.extend({"group": group, **finding} for finding in diagnose(
                    question=item["question"], hypothesis=item["hypothesis"],
                    facts=getattr(parsed, group), visible_evidence=item["visible_evidence"], catalog=catalog,
                ))
            output.update(
                status="parsed", parsed=asdict(parsed), diagnostics=diagnostics,
                absence_flags=sum(d["named_subject_absent_in_available_cited_blocks"] for d in diagnostics),
            )
        except Exception as exc:
            output.update(status="error", error=f"{type(exc).__name__}: {exc}")
        assert item == original
        outputs.append(output)
    summary = {}
    for kind in sorted({r["request_kind"] for r in outputs}):
        selected = [r for r in outputs if r["request_kind"] == kind]
        summary[kind] = {
            "raw_outputs": len(selected), "parse_status": dict(Counter(r["status"] for r in selected)),
            "outputs_with_eligible_checks": sum(bool(r.get("diagnostics")) for r in selected),
            "fact_subject_checks": sum(len(r.get("diagnostics", [])) for r in selected),
            "flagged_raw_outputs": sum(bool(r.get("absence_flags")) for r in selected),
            "absence_flags": sum(r.get("absence_flags", 0) for r in selected),
        }
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    shared.save(HERE / "diagnosis_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "outputs": outputs, "summary_not_accuracy": summary,
        "gpu_used": False, "new_llm_calls": 0, "retrieval_calls": 0,
        "production_changed": False, "training_ready": False,
        "diagnostic_only_no_production_gate": True,
    })
    print(summary, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_staged_subject_diagnosis.py prepare|run")
    operations[sys.argv[1]]()
