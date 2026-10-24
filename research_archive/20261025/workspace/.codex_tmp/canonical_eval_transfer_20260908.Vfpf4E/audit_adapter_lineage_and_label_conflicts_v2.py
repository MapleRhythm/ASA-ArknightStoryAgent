"""Audit all three recorded stages, split overlap and exact contradictory KTO labels."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import yaml


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_key(row):
    for message in row["conversations"]:
        if message.get("from") == "human":
            match = re.search(r"(?m)^question:\s*(.+)$", message["value"])
            if match:
                return re.sub(r"\s+", "", match.group(1)).strip("？?。！!，,；;：:\"'")
    return None


def sample_key(row, normalize_assistant_json=False):
    conversations = []
    for message in row["conversations"]:
        value = message["value"]
        if normalize_assistant_json and message.get("from") == "gpt":
            try:
                value = {"parsed_assistant_json": json.loads(value)}
            except json.JSONDecodeError:
                pass
        conversations.append({"from": message["from"], "value": value})
    return digest({"system": row.get("system"), "tools": row.get("tools"), "conversations": conversations})


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
names = [
    "llama_factory_soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_config.yaml",
    "llama_factory_soda_blackbox_deepseek_v1_550_parallel_config.yaml",
    "llama_factory_online_teacher_chain_v2_quality_fix3_plus_detail_conclusion_patch_v1_config.yaml",
]
configs = [yaml.safe_load((repo / "src/config" / n).read_text()) for n in names]
for child, parent in zip(configs, configs[1:]):
    assert (repo / child["adapter_name_or_path"]).resolve() == (repo / parent["output_dir"]).resolve()
assert "adapter_name_or_path" not in configs[-1] and configs[-1]["stage"] == "sft"
adapter = (repo / "release/ASA-ArknightStoryAgent/model/lora/asa-arknightstoryagent-4b-lora").resolve()
assert adapter == (repo / configs[0]["output_dir"]).resolve()
matrix = json.loads((run / "fixed79.current_quote.matrix.json").read_text())
eval_keys = {re.sub(r"\s+", "", r["question"]).strip("？?。！!，,；;：:\"'") for r in matrix["rows"]}
all_train_questions = set()
report = {
    "version": 2, "evaluation_only": True, "training_ready": False,
    "canonical_host": "${REMOTE_HOST}", "adapter": str(adapter),
    "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
    "prior_audit_sha256": sha(run / "configured_adapter_contract_audit_v1.json"),
    "stages": [],
}
for name, config in zip(names, configs):
    folder = repo / config["dataset_dir"]
    registry_path = folder / "dataset_info.json"
    registry = json.loads(registry_path.read_text())
    stats = {"config": name, "config_sha256": sha(repo / "src/config" / name), "stage": config["stage"],
             "registry_sha256": sha(registry_path), "splits": {}}
    rows_by_split, questions_by_split = {}, {}
    for split, registry_key in (("train", "dataset"), ("val", "eval_dataset")):
        path = folder / registry[config[registry_key]]["file_name"]
        rows = json.loads(path.read_text())
        rows_by_split[split] = rows
        questions = {q for row in rows if (q := question_key(row))}
        questions_by_split[split] = questions
        groups = defaultdict(Counter)
        for row in rows:
            group = (row.get("task_type", "unknown"), str(row.get("kto_tag", "sft")))
            counts = groups[group]
            counts["rows"] += 1
            for message in row["conversations"]:
                if message.get("from") != "gpt":
                    continue
                try:
                    payload = json.loads(message["value"])
                except json.JSONDecodeError:
                    counts["invalid_json_outputs"] += 1
                    continue
                counts["valid_json_outputs"] += 1
                if isinstance(payload, dict):
                    for key in ("answer", "final_answer", "supported_facts", "inferred_facts"):
                        if key in payload:
                            counts[f"has_{key}"] += 1
                    if payload.get("next_action") == "answer_directly":
                        counts["answer_directly"] += 1
        split_stats = {
            "file": str(path), "sha256": sha(path), "rows": len(rows),
            "distinct_questions": len(questions),
            "question_parse_failed": sum(question_key(row) is None for row in rows),
            "development_question_overlap": sorted(questions & eval_keys),
            "task_and_tag_counts": [
                {"task": task, "tag": tag, **dict(counts)} for (task, tag), counts in sorted(groups.items())
            ],
        }
        for normalized in (False, True):
            samples = defaultdict(list)
            for row in rows:
                samples[sample_key(row, normalized)].append(row)
            conflicts = [
                {"sample_sha256": key, "rows": [
                    {"id": r["id"], "task": r.get("task_type"), "kto_tag": r.get("kto_tag")}
                    for r in sample_rows
                ]}
                for key, sample_rows in samples.items()
                if {r.get("kto_tag") for r in sample_rows} == {False, True}
            ]
            split_stats["json_normalized_label_conflicts" if normalized else "exact_label_conflicts"] = conflicts
        stats["splits"][split] = split_stats
    all_train_questions.update(questions_by_split["train"])
    stats["train_val_question_overlap"] = sorted(questions_by_split["train"] & questions_by_split["val"])
    train_samples = {sample_key(row) for row in rows_by_split["train"]}
    stats["train_val_exact_sample_overlap"] = sum(sample_key(row) in train_samples for row in rows_by_split["val"])
    report["stages"].append(stats)
report["all_recorded_train_question_overlap_with_fixed79"] = sorted(all_train_questions & eval_keys)
report["fixed79_distinct_questions"] = len(eval_keys)
report["limitations"] = [
    "Recorded repository configuration/data lineage, not authenticated historical preprocessor cache or CLI overrides.",
    "Only exact question text and exact input/output collisions are checked; no fuzzy/semantic overlap inference.",
    "KTO rejected responses are intentional negative examples, not positive-label corruption.",
    "No contradiction is inferred just because chosen and rejected answers address the same question differently.",
    "Legacy target schemas establish a contract difference; causal impact requires controlled generation experiments.",
    "No training, labels, original datasets, checkpoints or production configuration are changed.",
]
with (run / "adapter_lineage_and_label_conflicts_v2.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
for stage in report["stages"]:
    print(json.dumps({
        "config": stage["config"], "stage": stage["stage"],
        "train_val_question_overlap_count": len(stage["train_val_question_overlap"]),
        "train_val_exact_sample_overlap": stage["train_val_exact_sample_overlap"],
        "splits": {name: {
            "rows": s["rows"], "questions": s["distinct_questions"],
            "eval_question_overlap": len(s["development_question_overlap"]),
            "exact_label_conflicts": len(s["exact_label_conflicts"]),
            "json_normalized_label_conflicts": len(s["json_normalized_label_conflicts"]),
            "task_and_tag_counts": s["task_and_tag_counts"],
        } for name, s in stage["splits"].items()},
    }, ensure_ascii=False))
print(json.dumps({"all_recorded_training_overlap_questions":
                  len(report["all_recorded_train_question_overlap_with_fixed79"]),
                  "fixed79_distinct_questions": len(eval_keys)}))
