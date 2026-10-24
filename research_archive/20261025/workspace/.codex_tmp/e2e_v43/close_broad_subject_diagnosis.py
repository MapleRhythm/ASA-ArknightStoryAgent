from datetime import datetime, timezone
from pathlib import Path
import sys
import run_broad_subject_diagnosis as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared
sys.path.insert(0, str(probe.V42))
from subject_reference_diagnostic import evidence_blocks


def main():
    pre, result = shared.load(HERE / "preflight.json"), shared.load(HERE / "diagnosis_v1.json")
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert result["summary_not_accuracy"] == {
        "outputs": 58, "status": {"parsed": 58},
        "outputs_with_absence_flags": 1, "fact_subject_absence_flags": 2,
    }
    flagged = [r for r in result["results"] if r["absence_flags"]]
    assert [r["id"] for r in flagged] == ["fixed58-017"]
    checked = [r for r in result["results"] if r["diagnostics"]]
    assert len(checked) == 9 and sum(len(r["diagnostics"]) for r in checked) == 16
    source = next(r for r in pre["inputs"] if r["id"] == "fixed58-017")
    blocks = evidence_blocks(source["evidence"])
    bindings = []
    for eid, quote in (
        ("E2", "雪绒：......那个糖的问题......还没有回答我呢......"),
        ("E3", "母亲从雪地中拽出雪绒，拽出他的法杖。"),
    ):
        assert quote in blocks[eid] and quote in source["raw"]
        assert all(name not in blocks[eid] for name in ("初雪", "圣聆初雪", "恩雅"))
        bindings.append({
            "id": source["id"], "eid": eid, "quote": quote,
            "block_sha256": shared.digest(blocks[eid]),
            "assistant_observation": "Qanipalaat's scene does not establish Pramanix's family events",
        })
    files = [HERE / name for name in (
        "preflight.json", "diagnosis_v1.json", "run_broad_subject_diagnosis.py", "close_broad_subject_diagnosis.py",
    )]
    shared.save(HERE / "checkpoint_v43.json", {
        "version": "broad_subject_reference_diagnosis_closed_v43",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False, "gpu_used": False,
        "summary_not_accuracy": result["summary_not_accuracy"],
        "outputs_with_in_scope_name_checks": 9, "fact_subject_name_checks": 16,
        "assistant_source_bindings": bindings,
        "promotion_decision": "diagnostic_only_no_production_gate",
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": [
            "Only 9 of 58 outputs contain eligible named-subject checks; do not claim other 57 outputs passed a semantic validator.",
            "Absence flag repeats the same Pramanix subject mistake found in v35/v41, not a new independent discovery.",
            "Alias/name presence cannot validate relation direction, motive, event, time, or creator identity.",
            "Historical actions and reconstructed evidence remain historical; no live two-round pipeline or correctness score was measured.",
            "No diagnostic threshold was tuned after observing the broader outputs.",
        ],
    })
    print({"outputs": 58, "in_scope_outputs": 9, "named_fact_checks": 16,
           "flagged_questions": 1, "promotion": "diagnostic only"})


if __name__ == "__main__":
    main()
