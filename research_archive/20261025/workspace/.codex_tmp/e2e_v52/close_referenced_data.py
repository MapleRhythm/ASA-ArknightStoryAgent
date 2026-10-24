from datetime import datetime, timezone
from pathlib import Path
import audit_referenced_data as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v52.json").exists()
    pre, saved = shared.load(HERE / "preflight.json"), shared.load(HERE / "audit_v1.json")
    probe.verify(pre)
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    replay = probe.compute()
    assert all(saved[key] == value for key, value in replay.items())
    assert saved["records_total"] == len(saved["rows"]) == 208
    desired = [r for r in saved["summary_by_split_task_and_kto_tag"]
               if r["task"] == "conclusion_generation" and r["kto_tag"]]
    assert [(r["split"], r["records"]) for r in desired] == [("train", 63), ("val", 6)]
    for row in desired:
        assert row["strict_object_targets"] == row["answer_field_rows"] == row["records"]
        assert row["prompt_schema_labels"] == {"conclusion_v2": row["records"]}
        assert row["rows_with_numbered_evidence"] == row["records"]
        assert not any(row[k] for k in (
            "supported_fact_field_rows", "inferred_fact_field_rows", "evidence_refs_key_rows",
            "final_answer_field_rows", "rows_with_E_headings",
        ))
    overlap = saved["current_registered_split_overlap_not_accuracy"]
    assert overlap["distinct_questions"] == {"train": 49, "val": 8}
    assert len(overlap["shared_question_texts"]) == 7
    assert len(overlap["val_questions_absent_from_train"]) == 1
    assert overlap["val_records_with_question_in_train"] == 14
    assert overlap["val_records_with_exact_prompt_in_train"] == 0
    assert not overlap["actual_prompt_envelope_shared_sha256"]
    assert all(not r["exact_question_train_rows"] and not r["exact_question_val_rows"]
               for r in saved["current_runtime_contract_and_exact_question_overlap"])
    anomalies = [r for r in saved["rows"] if
                 r["target_parse_status"] != "object"
                 or (r["task_type"] == "conclusion_generation" and r["target_action"] is None)]
    assert len(anomalies) == 2 and all(r["kto_tag"] is False for r in anomalies)
    stale_role = [r for r in saved["rows"]
                  if r["kto_tag"] is False and r["preference_role_metadata"] == "teacher_positive"]
    assert len(stale_role) == 12
    report = probe.ROOT / "docs/CANONICAL_REFERENCED_TRAINING_CONTRACT_20260910_v52.md"
    paths = [HERE / name for name in (
        "README.md", "audit_referenced_data.py", "close_referenced_data.py",
        "preflight.json", "prepare.log", "audit_v1.json", "run_v1.log",
    )] + [report]
    probe.verify(pre)
    shared.save(HERE / "checkpoint_v52.json", {
        "version": "current_lora_referenced_data_contract_closed_v52",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_registered_records": 208, "train_records": 190, "val_records": 18,
        "positive_conclusion_train_records": 63, "positive_conclusion_val_records": 6,
        "positive_conclusion_rows_with_structured_fact_or_reference_fields": 0,
        "independent_current_file_replay_equal": True,
        "shared_train_val_question_texts": 7,
        "val_records_with_question_in_train": 14,
        "val_records_with_exact_prompt_in_train": 0,
        "exact_v51_question_overlap_with_registered_dataset": 0,
        "negative_only_schema_or_json_anomalies": 2,
        "negative_kto_rows_with_stale_teacher_positive_metadata": 12,
        "historical_training_data_bytes_proven": False,
        "complete_parent_adapter_training_history_audited": False,
        "production_changed": False, "training_ready": False, "labels_modified": False,
        "training_config_executed": False, "training_args_bin_loaded": False,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "semantic_accuracy_or_causality_claimed": False, "human_gold": False,
        "basis_sha256": {str(p): shared.sha(p) for p in paths},
        "decision": "record_current_referenced_schema_gap_no_training_or_relabeling",
        "limitations": [
            "Current modelcard/config/registry agreement is not training-time dataset SHA proof.",
            "Parent adapter is declared; final-stage208 rows do not describe all model training.",
            "Same-question overlap is not exact-prompt leakage; no exact train/val prompt duplicates were found.",
            "No exact overlap of the six v51 questions does not prove semantic novelty or independence from earlier training/development.",
            "Absence of structured fact targets in this registered stage does not prove the cause of current model errors.",
            "Rejected malformed/wrong-task outputs are negative KTO examples, not desired positive targets.",
        ],
    })
    print({"records_replayed": 208, "positive_conclusion_train": 63,
           "structured_fact_targets_in_this_group": 0,
           "val_same_question_records": 14, "val_same_prompt_records": 0,
           "historical_bytes_proven": False, "training_started": False}, flush=True)


if __name__ == "__main__":
    main()
