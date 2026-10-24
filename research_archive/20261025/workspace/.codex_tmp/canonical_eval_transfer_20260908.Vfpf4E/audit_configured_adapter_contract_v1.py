"""Read-only schema/overlap audit of configured adapter's recorded training chain."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_question(text):
    return re.sub(r"\s+", "", text).strip("？?。！!，,；;：:\"'")


def question_of(row):
    for message in row.get("conversations", []):
        if message.get("from") != "human":
            continue
        match = re.search(r"(?m)^question:\s*(.+)$", message.get("value", ""))
        if match:
            return match.group(1).strip()
    return None


def inspect_split(rows):
    summaries, details, question_ids = {}, [], {}
    for row in rows:
        group = (row.get("task_type", "missing"), str(row.get("kto_tag")))
        stats = summaries.setdefault(group, Counter())
        stats["rows"] += 1
        question = question_of(row)
        if question:
            question_ids.setdefault(norm_question(question), []).append(row["id"])
        else:
            stats["question_parse_failed"] += 1
        replies = [m["value"] for m in row["conversations"] if m.get("from") == "gpt"]
        item = {"id": row["id"], "task": group[0], "kto_tag": row.get("kto_tag"), "question": question, "outputs": []}
        for text in replies:
            record = {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text)}
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                stats["invalid_json_outputs"] += 1
                record.update(strict_json=False, error=f"JSONDecodeError: {exc.msg}", tail=text[-120:])
            else:
                stats["strict_json_outputs"] += 1
                record["strict_json"] = True
                if isinstance(value, dict):
                    record["keys"] = sorted(value)
                    record["action"] = value.get("next_action")
                    for key in ("answer", "final_answer", "supported_facts", "inferred_facts", "evidence_refs", "evidence_ids"):
                        if key in value:
                            stats[f"top_level_{key}"] += 1
                    if value.get("next_action") == "answer_directly":
                        stats["answer_directly_outputs"] += 1
                    if any(isinstance(f, dict) and f.get("evidence_refs") for f in value.get("supported_facts", []) or []):
                        stats["supported_fact_quote_refs"] += 1
                    if any(isinstance(f, dict) and f.get("evidence_ids") for f in value.get("supported_facts", []) or []):
                        stats["supported_fact_exx_ids"] += 1
            item["outputs"].append(record)
        details.append(item)
    return {
        "rows": len(rows), "distinct_questions": len(question_ids),
        "by_task_and_tag": [
            {"task_type": task, "kto_tag": tag, **dict(stats)}
            for (task, tag), stats in sorted(summaries.items())
        ],
        "question_ids": question_ids, "details": details,
    }


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
adapter = (repo / "release/ASA-ArknightStoryAgent/model/lora/asa-arknightstoryagent-4b-lora").resolve()
configs = [
    repo / "src/config/llama_factory_soda_eval50_len1800_api_verifier_v2_scoped_sweep_soda_lora_config.yaml",
    repo / "src/config/llama_factory_soda_blackbox_deepseek_v1_550_parallel_config.yaml",
]
config_values = [yaml.safe_load(path.read_text()) for path in configs]
assert (repo / config_values[0]["output_dir"]).resolve() == adapter
assert (repo / config_values[0]["adapter_name_or_path"]).resolve() == (repo / config_values[1]["output_dir"]).resolve()
matrix_path = run / "fixed79.current_quote.matrix.json"
matrix = json.loads(matrix_path.read_text())
eval_questions = {norm_question(r["question"]): r["question"] for r in matrix["rows"]}
report = {
    "evaluation_only": True, "training_ready": False, "canonical_host": "${REMOTE_HOST}",
    "adapter": str(adapter), "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
    "adapter_card_sha256": sha(adapter / "README.md"),
    "development_matrix_sha256": sha(matrix_path), "development_states": len(matrix["rows"]),
    "development_distinct_questions": len(eval_questions), "stages": [],
}
for config_path, config in zip(configs, config_values):
    dataset_dir = repo / config["dataset_dir"]
    registry_path = dataset_dir / "dataset_info.json"
    registry = json.loads(registry_path.read_text())
    stage = {
        "config": str(config_path), "config_sha256": sha(config_path),
        "stage": config["stage"], "input_adapter": config["adapter_name_or_path"],
        "output_adapter": config["output_dir"], "template": config["template"], "cutoff_len": config["cutoff_len"],
        "registry_sha256": sha(registry_path), "splits": {},
    }
    for split, key in (("train", "dataset"), ("val", "eval_dataset")):
        path = dataset_dir / registry[config[key]]["file_name"]
        stats = inspect_split(json.loads(path.read_text()))
        stats.update(path=str(path), sha256=sha(path))
        stats["literal_development_question_overlap"] = [
            {"question": eval_questions[q], "training_row_ids": stats["question_ids"][q]}
            for q in sorted(set(stats["question_ids"]) & set(eval_questions))
        ]
        stage["splits"][split] = stats
    train_questions, val_questions = [stage["splits"][s]["question_ids"] for s in ("train", "val")]
    stage["train_val_question_overlap_count"] = len(set(train_questions) & set(val_questions))
    stage["train_val_question_overlap"] = [
        {"normalized_question": q, "train_ids": train_questions[q], "val_ids": val_questions[q]}
        for q in sorted(set(train_questions) & set(val_questions))
    ]
    report["stages"].append(stage)
report["limitations"] = [
    "Matches repository configs, dataset registry and model-card lineage; does not prove historical CLI overrides or cache contents.",
    "Current datasets are hashed; their historical training-time bytes are not independently authenticated here.",
    "Old answer schema without explicit fact citations demonstrates a recorded contract gap, not by itself causality of wrong answers.",
    "KTO rejected examples are intentionally negative; their defects must not be counted as positive training-label corruption.",
    "Question overlap is normalized exact text, not a semantic near-duplicate or full contamination audit.",
    "No model, data, training labels or production configuration is modified.",
]
with (run / "configured_adapter_contract_audit_v1.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
for stage in report["stages"]:
    print(json.dumps({
        "config": stage["config"], "cutoff": stage["cutoff_len"],
        "train_val_question_overlap": stage["train_val_question_overlap_count"],
        "splits": {name: {k: v for k, v in split.items() if k in
            ("rows", "distinct_questions", "by_task_and_tag")} | {
                "development_question_overlap": len(split["literal_development_question_overlap"])
            } for name, split in stage["splits"].items()},
    }, ensure_ascii=False))
