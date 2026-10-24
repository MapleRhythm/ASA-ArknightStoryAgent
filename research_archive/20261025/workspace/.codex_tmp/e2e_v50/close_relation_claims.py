from datetime import datetime, timezone
from pathlib import Path
import audit_relation_claims as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v50.json").exists()
    pre, saved = shared.load(HERE / "preflight.json"), shared.load(HERE / "audit_v1.json")
    probe.verify(pre)
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    replay = probe.compute()
    assert all(saved[key] == value for key, value in replay.items())
    assert replay["exact_endpoint_replays"] == 3 and replay["cached_raw_calls_replayed"] == 4
    projections = [p for row in saved["rows"] for p in row["single_fact_projections_not_model_answers"]]
    controls = [c for row in saved["rows"] for c in row["invalid_reference_controls"]]
    assert len(projections) == len(controls) == 6
    assert all(p["result"]["next_action"] == "answer_directly" and p["claim_displayed_verbatim"]
               and not p["quote_issues"] and not p["fact_required_terms_missing"] for p in projections)
    assert all(p["quote_warnings"] for p in projections)
    assert all(c["quote_issues"] and c["result"]["next_action"] != "answer_directly"
               and not c["claim_displayed_verbatim"] for c in controls)
    report = probe.integration.ROOT / "docs/CANONICAL_RELATION_CLAIM_SENSITIVITY_20260910_v50.md"
    paths = [HERE / name for name in (
        "README.md", "audit_relation_claims.py", "close_relation_claims.py",
        "preflight.json", "prepare.log", "audit_v1.json", "run_v1.log",
    )] + [report]
    probe.verify(pre)
    shared.save(HERE / "checkpoint_v50.json", {
        "version": "source_bound_relation_claim_sensitivity_closed_v50",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "exact_endpoint_replays": 3, "cached_calls_per_replay": 4,
        "original_bad_fact_lineages_raw_parsed_validated_displayed": 3,
        "assistant_authored_single_fact_projections": 6,
        "projection_actions_not_accuracy": {"answer_directly": 6},
        "all_projections_have_lexical_warnings_not_relation_rejection": True,
        "invalid_reference_controls_rejected": 6,
        "independent_fresh_process_replay_equal": True,
        "production_changed": False, "training_ready": False,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "human_gold": False, "semantic_accuracy_gain_claimed": False,
        "v49_installed_source_mapping_reverified": True,
        "decision": "record_semantic_boundary_no_name_or_keyword_gate_promotion",
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
        "limitations": [
            "Three known failures were deliberately selected after observing outputs; no incidence or accuracy estimate.",
            "Synthetic paired claims are not actual model outputs or recommended answers to the original questions.",
            "The Juanna-consistent projection is not an answer to the Pramanix question; source consistency alone is insufficient.",
            "Creator relation is unsupported by the bound source, not proved false in all possible source material.",
            "Lexical warnings are present in all six projections; do not report completely warning-free acceptance.",
            "No original model output, label, evidence, E-ID, validator, alias or production prompt was altered.",
        ],
    })
    print({"exact_endpoint_replays": 3, "paired_projection_accepts_not_accuracy": 6,
           "invalid_reference_controls_rejected": 6, "production_changed": False}, flush=True)


if __name__ == "__main__":
    main()
