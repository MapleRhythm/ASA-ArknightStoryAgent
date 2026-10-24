from datetime import datetime, timezone
from pathlib import Path
import run_subject_diagnosis as audit
from subject_reference_diagnostic import diagnose, evidence_blocks

HERE = Path(__file__).resolve().parent
shared = audit.shared


def main():
    pre, result = shared.load(HERE / "preflight.json"), shared.load(HERE / "diagnosis_v1.json")
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    assert result["summary_not_accuracy"] == {
        "v41": {"outputs": 24, "status": {"parsed": 24}, "outputs_with_absence_flags": 2, "fact_subject_absence_flags": 4},
        "v35": {"outputs": 32, "status": {"parsed": 32}, "outputs_with_absence_flags": 2, "fact_subject_absence_flags": 4},
    }
    catalog = shared.load(audit.ROOT / "indexes/arknights_story/operator_aliases.json")
    bindings = []
    indexed = {}
    for source, row in zip(pre["inputs"], result["results"], strict=True):
        assert source["id"] == row["id"] and source["condition"] == row["condition"]
        replay = []
        for group in ("supported_facts", "inferred_facts"):
            replay.extend({"group": group, **r} for r in diagnose(
                question=source["question"], hypothesis=source["hypothesis"],
                facts=row["parsed"][group], visible_evidence=source["evidence"], catalog=catalog,
            ))
        assert replay == row["diagnostics"]
        key = (row["source_experiment"], row["pair_id"], row["condition"])
        indexed[key] = row
        if row["absence_flags"]:
            assert row["id"] == "fixed58-017" and row["recorded_action"] == "answer_directly"
            blocks = evidence_blocks(source["evidence"])
            for eid, quote in (
                ("E2", "雪绒的思维在涣散，他好像又看到了那成团的温柔羽毛"),
                ("E3", "母亲从雪地中拽出雪绒，拽出他的法杖。"),
            ):
                assert quote in blocks[eid]
                assert all(name not in blocks[eid] for name in ("初雪", "圣聆初雪", "恩雅"))
                bindings.append({
                    "source_experiment": row["source_experiment"], "pair_id": row["pair_id"],
                    "condition": row["condition"], "eid": eid, "quote": quote,
                    "block_sha256": shared.digest(blocks[eid]),
                    "raw_output_sha256": shared.digest(source["raw"]),
                    "assistant_observation": "cited scene names Qanipalaat, not Pramanix; wrong-subject rescue is not repaired or relabeled",
                })
    assert len(bindings) == 8
    known_misses = [
        ("v41", "fixed58-007:seed1", "control"),
        ("v41", "fixed58-049:seed1", "control"),
        ("v41", "fixed58-049:seed1", "candidate"),
    ]
    # All listed v41 wrong/uncertified relationship outputs stay unflagged.
    for key in known_misses:
        assert indexed[key]["absence_flags"] == 0
    files = [HERE / name for name in (
        "preflight.json", "diagnosis_v1.json", "subject_reference_diagnostic.py",
        "test_subject_reference_diagnostic.py", "run_subject_diagnosis.py", "close_subject_diagnosis.py",
    )]
    shared.save(HERE / "checkpoint_v42.json", {
        "version": "subject_presence_diagnosis_closed_v42",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "summary_not_accuracy": result["summary_not_accuracy"],
        "diagnostic_replays": 56, "assistant_source_bindings": bindings,
        "known_relationship_outputs_unflagged": [
            {"source_experiment": k[0], "pair_id": k[1], "condition": k[2]} for k in known_misses
        ],
        "promotion_decision": "diagnostic_only_no_production_gate; broaden_source_review_before_considering_any_reject_only_candidate",
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": [
            "Four flagged outputs are repetitions of ONE question, not four independent error discoveries.",
            "Only absence of known multi-character question/fact names in cited blocks is measured.",
            "Explicit aliases, pronouns, uncited identity bridges and broad chains make absence/presence different from truth.",
            "Synthetic tests deliberately show both a potential false alarm and a semantic false negative.",
            "Wrong mother/daughter direction and unsupported creator relationship are not detected.",
            "v35 recorded actions keep their historical empty-object standardized-reader scope; not relabeled as current E2E.",
            "All source/output reviews are assistant observations, not human gold or certified accuracy.",
        ],
    })
    print({"outputs_replayed": 56, "source_bindings": 8, "distinct_flagged_questions": 1,
           "promotion": "diagnostic only"})


if __name__ == "__main__":
    main()
