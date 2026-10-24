"""Current declared final-stage dataset contract, not historical byte proof."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
V49 = HERE.parent / "fallback_trace_retention_integration_20260910.v49"
V51 = HERE.parent / "ownership_rule_prompt_20260910.v51"
sys.path.insert(0, str(V49))
import run_trace_integration as integration

shared = integration.shared
ROOT = integration.ROOT
CONFIG = ROOT / "src/config/llama_factory_soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_config.yaml"
DATA = ROOT / "data/processed/llama_factory/soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_merged"
TRAIN_NAME = "soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_gpu3_train"


def digest(value):
    return shared.digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def verify(pre):
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    assert all(shared.sha(Path(path)) == expected for path, expected in pre["sha256"].items())


def config():
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def prepare():
    assert not (HERE / "preflight.json").exists()
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    cfg = config()
    model = Path(shared.load(V51 / "preflight.json")["adapter"])
    registry = shared.load(DATA / "dataset_info.json")
    assert cfg["stage"] == "kto"
    assert (ROOT / cfg["output_dir"]).resolve() == model.resolve()
    assert (ROOT / cfg["dataset_dir"]).resolve() == DATA.resolve()
    assert cfg["dataset"] == TRAIN_NAME
    assert registry[TRAIN_NAME]["file_name"] == "train.json"
    assert registry[cfg["eval_dataset"]]["file_name"] == "val.json"
    assert TRAIN_NAME in (model / "README.md").read_text()
    files = [
        Path(__file__), HERE / "README.md", CONFIG, model / "README.md",
        model / "trainer_state.json", model / "all_results.json",
        DATA / "dataset_info.json", DATA / "build_summary.json", DATA / "train.json", DATA / "val.json",
        V49 / "preflight.json", V49 / "checkpoint_v49.json", V49 / "run_trace_integration.py",
        V51 / "preflight.json", V51 / "checkpoint_v51.json",
        ROOT / "docs/current_full_flow_and_datasets_for_review.md",
    ]
    shared.save(HERE / "preflight.json", {
        "version": "current_lora_referenced_data_contract_v52",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {str(p): shared.sha(p) for p in files},
        "current_adapter": str(model), "current_config": str(CONFIG),
        "current_dataset_dir": str(DATA), "current_dataset_name": TRAIN_NAME,
        "parent_adapter_declared_by_config": cfg.get("adapter_name_or_path"),
        "config_not_executed": True, "training_args_bin_not_loaded": True,
        "protocol": [
            "Current model card/config/registry agreement is a reference chain, not a training-time file-content attestation.",
            "Read current merged208 records with exclusive new audit receipts; never rebuild, relabel or train.",
            "Keep positive/negative KTO labels separate using kto_tag, not potentially stale preference_role metadata.",
            "Strict JSON target inspection preserves every malformed output as a finding rather than repairing it.",
            "Actual question text and prompt-envelope digests define overlap; never join unrelated datasets by numeric ID suffix.",
            "Compare the current registered final-stage contract with frozen v51 canonical requested schema, not semantic accuracy.",
            "Earlier adapter is declared as input; do not claim these208 rows are the model's entire training history.",
            "Historical June3 document contains obsolete MiniRAG/weak runtime examples; never apply them.",
            "No training_args unpickling, model/GPU/retrieval, source/prompt/default changes, restart, commit or push.",
        ],
        "production_changed": False, "training_ready": False,
    })
    print({"config_registry_modelcard_reference_consistent": True,
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def inspect_row(row, split, index):
    assert isinstance(row["kto_tag"], bool)
    messages = row["conversations"]
    assistants = [i for i, m in enumerate(messages) if m["from"] == "gpt"]
    assert len(assistants) == 1 and assistants[0] == len(messages) - 1
    human = [m["value"] for m in messages[:assistants[0]] if m["from"] == "human"]
    assert len(human) == 1
    prompt, target = human[0], messages[assistants[0]]["value"]
    task = re.search(r"(?m)^task:\s*([^\n]+)", prompt)
    question = re.search(r"(?m)^question:[ \t]*([^\n]+)", prompt)
    assert task and question, row["id"]
    assert task.group(1).strip() == row["task_type"] == row["meta"]["task_type"]
    schema = re.search(r"(?m)^output_schema:[ \t]*([^\n]+)", prompt)
    envelope = {
        "system": row.get("system"), "tools": row.get("tools"),
        "messages": messages[:assistants[0]],
    }
    try:
        parsed = json.loads(target)
    except (json.JSONDecodeError, TypeError) as exc:
        parsed, status = None, f"{type(exc).__name__}: {exc}"
    else:
        status = "object" if isinstance(parsed, dict) else "non_object_json"
    evidence = None
    if "\nevidence_brief:\n" in prompt:
        evidence = prompt.split("\nevidence_brief:\n", 1)[1]
        evidence = re.split(r"\n(?:output_schema|action_set|rules|answer_directly|retrieve_more|abstain):", evidence, maxsplit=1)[0]
    fields = sorted(parsed) if isinstance(parsed, dict) else []
    out = {
        "split": split, "row_index": index, "id": row["id"], "row_sha256": digest(row),
        "question": question.group(1).strip(), "task_type": row["task_type"],
        "question_key_metadata": row["meta"].get("question_key"),
        "prompt_key_metadata": row["meta"].get("prompt_key"),
        "actual_prompt_envelope_sha256": digest(envelope), "target_sha256": shared.digest(target),
        "kto_tag": row["kto_tag"], "preference_role_metadata": row["meta"].get("preference_role"),
        "api_verifier_reason_metadata": row["meta"].get("api_verifier_reason"),
        "prompt_schema_label": schema.group(1).strip() if schema else None,
        "target_parse_status": status, "target_top_level_fields": fields,
        "target_action": parsed.get("next_action") if isinstance(parsed, dict) else None,
        "has_supported_facts_field": "supported_facts" in fields,
        "has_inferred_facts_field": "inferred_facts" in fields,
        "has_final_answer_field": "final_answer" in fields, "has_answer_field": "answer" in fields,
        "has_any_evidence_refs_key_in_target": bool(re.search(r'"evidence_refs"\s*:', target)),
        "evidence_brief_present": evidence is not None,
        "evidence_numbered_line_count": len(re.findall(r"(?m)^\d+\.\s", evidence or "")),
        "evidence_E_heading_count": len(re.findall(r"(?m)^\[E\d+\]", evidence or "")),
    }
    # Output only, not metadata.source.gold or the full evidence packet. These
    # matches are retrieval aids for future source review, not semantic labels.
    if re.search(r"娜斯提|初雪|槐天裴|保存者", out["question"]):
        out["selected_topic_target_for_read_only_review"] = target
    return out


def compute():
    cfg = config()
    datasets = {split: shared.load(DATA / f"{split}.json") for split in ("train", "val")}
    records = [inspect_row(row, split, i) for split, rows in datasets.items() for i, row in enumerate(rows)]
    build = shared.load(DATA / "build_summary.json")
    assert {split: len(rows) for split, rows in datasets.items()} == {
        "train": build["records_train"], "val": build["records_val"],
    }
    assert len(records) == build["records_total"]
    assert dict(Counter(str(r["kto_tag"]) for r in records)) == build["kto_tags"]
    assert dict(Counter(r["task_type"] for r in records)) == build["task_counts"]
    groups = defaultdict(list)
    for row in records:
        groups[(row["split"], row["task_type"], row["kto_tag"])].append(row)
    summary = []
    for (split, task, tag), rows in sorted(groups.items()):
        summary.append({
            "split": split, "task": task, "kto_tag": tag, "records": len(rows),
            "strict_object_targets": sum(r["target_parse_status"] == "object" for r in rows),
            "non_object_or_parse_errors": sum(r["target_parse_status"] != "object" for r in rows),
            "target_actions": dict(Counter(str(r["target_action"]) for r in rows)),
            "schema_field_sets": dict(Counter(",".join(r["target_top_level_fields"]) for r in rows)),
            "prompt_schema_labels": dict(Counter(str(r["prompt_schema_label"]) for r in rows)),
            "supported_fact_field_rows": sum(r["has_supported_facts_field"] for r in rows),
            "inferred_fact_field_rows": sum(r["has_inferred_facts_field"] for r in rows),
            "evidence_refs_key_rows": sum(r["has_any_evidence_refs_key_in_target"] for r in rows),
            "answer_field_rows": sum(r["has_answer_field"] for r in rows),
            "final_answer_field_rows": sum(r["has_final_answer_field"] for r in rows),
            "rows_with_numbered_evidence": sum(bool(r["evidence_numbered_line_count"]) for r in rows),
            "rows_with_E_headings": sum(bool(r["evidence_E_heading_count"]) for r in rows),
        })
    questions, envelopes, states = {}, {}, {}
    for split in datasets:
        subset = [r for r in records if r["split"] == split]
        questions[split] = {r["question"] for r in subset}
        envelopes[split] = {r["actual_prompt_envelope_sha256"] for r in subset}
        states[split] = {r["prompt_key_metadata"] for r in subset}
    shared_questions = questions["train"] & questions["val"]
    shared_envelopes = envelopes["train"] & envelopes["val"]
    overlap = {
        "distinct_questions": {s: len(q) for s, q in questions.items()},
        "shared_question_texts": sorted(shared_questions),
        "val_questions_absent_from_train": sorted(questions["val"] - questions["train"]),
        "actual_prompt_envelope_shared_sha256": sorted(shared_envelopes),
        "metadata_prompt_keys_shared": sorted(states["train"] & states["val"]),
        "val_records_with_question_in_train": sum(r["split"] == "val" and r["question"] in shared_questions for r in records),
        "val_records_with_exact_prompt_in_train": sum(r["split"] == "val" and r["actual_prompt_envelope_sha256"] in shared_envelopes for r in records),
        "exact_prompt_overlap_rows": [
            {k: r[k] for k in ("split", "row_index", "id", "kto_tag", "question", "actual_prompt_envelope_sha256", "target_sha256")}
            for r in records if r["actual_prompt_envelope_sha256"] in shared_envelopes
        ],
    }
    current = [r for r in shared.load(V51 / "preflight.json")["conditions"] if r["condition"] == "baseline"]
    runtime = []
    for item in current:
        lines = item["canonical_prompt"].splitlines()
        schema_line = next(line for line in lines if line.startswith("output_schema:"))
        requested = json.loads(next(line for line in lines if line.startswith("answer_directly:")).split(":", 1)[1])
        runtime.append({
            "id": item["id"], "question": item["question"],
            "prompt_sha256": item["canonical_prompt_sha256"],
            "schema_line": schema_line, "requested_answer_fields": sorted(requested),
            "exact_question_train_rows": [r["id"] for r in records if r["split"] == "train" and r["question"] == item["question"]],
            "exact_question_val_rows": [r["id"] for r in records if r["split"] == "val" and r["question"] == item["question"]],
        })
    model = Path(shared.load(V51 / "preflight.json")["adapter"])
    trainer = shared.load(model / "trainer_state.json")
    return {
        "current_config_reference": {
            k: cfg[k] for k in ("stage", "dataset_dir", "dataset", "eval_dataset", "output_dir",
                               "adapter_name_or_path", "template", "cutoff_len", "num_train_epochs", "pref_beta")
        },
        "modelcard_names_current_registry_entry": TRAIN_NAME in (model / "README.md").read_text(),
        "trainer_state_selected_metadata": {
            k: trainer.get(k) for k in ("epoch", "global_step", "max_steps", "num_train_epochs")
        },
        "rows": records, "records_total": len(records), "summary_by_split_task_and_kto_tag": summary,
        "current_registered_split_overlap_not_accuracy": overlap,
        "current_runtime_contract_and_exact_question_overlap": runtime,
        "production_changed": False, "training_ready": False,
        "historical_training_data_bytes_proven": False,
        "complete_parent_adapter_training_history_audited": False,
        "training_args_bin_loaded": False, "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "labels_modified": False, "semantic_accuracy_or_causality_claimed": False,
    }


def run():
    assert not (HERE / "audit_v1.json").exists()
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    result = compute()
    verify(pre)
    shared.save(HERE / "audit_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"), **result,
    })
    overlap = result["current_registered_split_overlap_not_accuracy"]
    print({
        "records": result["records_total"],
        "desired_conclusion_schema": [r for r in result["summary_by_split_task_and_kto_tag"]
                                      if r["task"] == "conclusion_generation" and r["kto_tag"]],
        "split_questions": overlap["distinct_questions"],
        "val_records_with_question_in_train": overlap["val_records_with_question_in_train"],
        "val_records_with_exact_prompt_in_train": overlap["val_records_with_exact_prompt_in_train"],
        "historical_bytes_proven": False,
    }, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: audit_referenced_data.py prepare|run")
    operations[sys.argv[1]]()
