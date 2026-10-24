"""Extend unchanged v42 name-presence diagnostic to all fixed58 final steps.

Read-only diagnostic, not a new validator; reconstructed historical visible
packets and historical recorded actions keep their original scope.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V42 = HERE.parent / "named_subject_reference_diagnosis_20260910.v42"
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
sys.path.insert(0, str(V27))
import run_reader_router as shared
sys.path.insert(0, str(V42))
from subject_reference_diagnostic import diagnose


def prepare():
    assert not (HERE / "preflight.json").exists()
    previous = shared.load(V42 / "preflight.json")
    shared.load(V42 / "checkpoint_v42.json")
    assert all(shared.sha(Path(p)) == h for p, h in previous["sha256"].items())
    source = V27 / "results.jsonl"
    data = [json.loads(line) for line in source.read_text().splitlines()]
    visible = shared.load(V27 / "visible_answer_analysis.json")
    assert visible["source_results_sha256"] == shared.sha(source)
    assert len(data) == len(visible["results"]) == 58
    inputs = []
    for index, (result, packed) in enumerate(zip(data, visible["results"], strict=True), start=1):
        assert result["question"] == packed["question"] and packed["id"] == f"fixed58-{index:03d}"
        assert packed["status"] == "ok" and packed["evidence_checks"]
        assert all(c["status"] == "exact_length_match" for c in packed["evidence_checks"])
        trace = result["retrieval_trace"][-1]
        conclusion = trace["conclusion"]
        raw = conclusion["generation_diagnostics"]["raw_output"]
        assert raw and not result["error"]
        inputs.append({
            "id": packed["id"], "question": result["question"], "hypothesis": trace["hypothesis"],
            "raw": raw, "evidence": packed["evidence_brief"], "evidence_checks": packed["evidence_checks"],
            "current_round": packed["current_round"], "max_retrieval_rounds": packed["max_retrieval_rounds"],
            "recorded_action": conclusion["next_action"], "recorded_answer": result["answer"],
            "recorded_action_scope": "historical_fixed58_last_step_not_new_current_validator_replay",
        })
    paths = [
        Path(__file__), V42 / "subject_reference_diagnostic.py", V42 / "test_subject_reference_diagnostic.py",
        V42 / "checkpoint_v42.json", source, V27 / "visible_answer_analysis.json",
        ROOT / "indexes/arknights_story/operator_aliases.json", *sorted(SRC.rglob("*.py")),
    ]
    shared.save(HERE / "preflight.json", {
        "version": "unchanged_subject_diagnostic_broad_fixed58_v43",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False, "gpu_used": False,
        "inputs": inputs, "sha256": {str(p): shared.sha(p) for p in paths},
        "protocol": [
            "All 58 last-step raw outputs from the frozen no-MiniRAG fixed58 run, not only outputs already known to fail.",
            "Reuse unchanged v42 diagnostic and current canonical parser; do not tune on new results.",
            "Historical rendered evidence is the existing source-bound reconstruction with all packing lengths checked, not a fresh retrieved context.",
            "Original actions/answers remain historical observations; neither old nor new names are human-gold quality labels.",
            "No new generation/retrieval, output correction, source rewriting, reference expansion, alias edits, or decision changes.",
            "Broader development coverage, not a held-out corpus; repeat questions across v35/v41/v43 must not be counted as independent tests.",
            "Investigate every absence flag and known misses before considering any new gate; this script does not implement one.",
        ],
    })
    print({"inputs": 58, "gpu_used": False, "production_changed": False,
           "preflight_sha256": shared.sha(HERE / "preflight.json")})


def run():
    pre = shared.load(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    sys.path.insert(0, str(SRC))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import normalize_minimal_conclusion_output, parse_conclusion_output
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    catalog = shared.load(ROOT / "indexes/arknights_story/operator_aliases.json")
    rows = []
    for source in pre["inputs"]:
        row = {k: source[k] for k in ("id", "question", "recorded_action", "recorded_action_scope")}
        try:
            parsed = parse_conclusion_output(
                normalize_minimal_conclusion_output(source["raw"], "minimal"),
                question=source["question"], current_hypothesis=HypothesisDocument(**source["hypothesis"]),
                max_round_reached=source["current_round"] >= source["max_retrieval_rounds"],
            )
            findings = []
            for group in ("supported_facts", "inferred_facts"):
                findings.extend({"group": group, **r} for r in diagnose(
                    question=source["question"], hypothesis=source["hypothesis"], facts=getattr(parsed, group),
                    visible_evidence=source["evidence"], catalog=catalog,
                ))
            row.update(status="parsed", diagnostics=findings, absence_flags=sum(
                r["named_subject_absent_in_available_cited_blocks"] for r in findings
            ))
        except Exception as exc:
            row.update(status="parse_or_diagnostic_error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    assert len(rows) == 58
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    summary = {
        "outputs": 58, "status": dict(Counter(r["status"] for r in rows)),
        "outputs_with_absence_flags": sum(bool(r.get("absence_flags")) for r in rows),
        "fact_subject_absence_flags": sum(r.get("absence_flags", 0) for r in rows),
    }
    shared.save(HERE / "diagnosis_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "results": rows, "summary_not_accuracy": summary, "model_calls": 0, "retrieval_calls": 0,
        "production_changed": False, "training_ready": False, "gpu_used": False,
        "semantic_review_complete": False,
    })
    print(summary)
    for row in rows:
        if row.get("absence_flags") or row["status"] != "parsed":
            print(row)


if __name__ == "__main__":
    actions = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit("usage: run_broad_subject_diagnosis.py prepare|run")
    actions[sys.argv[1]]()
