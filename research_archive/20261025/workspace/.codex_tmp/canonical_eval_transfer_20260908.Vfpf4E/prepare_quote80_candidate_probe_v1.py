"""Freeze an existing quote-SFT adapter probe against the exact recent controls."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

import yaml


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def qkey(text):
    return re.sub(r"\s+", "", text).strip("？?。！!，,；;：:\"'")


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
release = repo / "release/ASA-ArknightStoryAgent"
config_path = repo / "src/config/llama_factory_teacher_state_grounded_action_sft_all_v2_quote80_drop_unfixable_hybrid_a100_rank8_cutoff10000_config.yaml"
config = yaml.safe_load(config_path.read_text())
adapter = (repo / config["output_dir"]).resolve()
base = (repo / config["model_name_or_path"]).resolve()
assert config["stage"] == "sft" and "adapter_name_or_path" not in config
assert base == (release / "model/qwen3.5-4b").resolve()
adapter_config = load(adapter / "adapter_config.json")
assert (repo / adapter_config["base_model_name_or_path"]).resolve() == base
state = load(adapter / "trainer_state.json")
assert state["global_step"] == state["max_steps"] == 80 and state["epoch"] == 2.0
control_folder = run / "question_focus_paired18_ctx12288_v1"
control_matrix_path = run / "question_focus_paired18_v1.matrix.json"
control_matrix = load(control_matrix_path)
controls = [r for r in control_matrix["rows"] if r["renderer"] == "focus_control_v1"]
assert len(controls) == 9
control_predictions_path = control_folder / "predictions.json"
predictions = {r["id"]: r for r in load(control_predictions_path)}
seed_receipt = {r["id"]: r for r in load(control_folder / "sampling_seed_override.json")["rows"]}
rows = []
for control in controls:
    pred = predictions[control["id"]]
    assert pred["actual_prompt"] == control["runtime_prompt"]
    assert seed_receipt[control["id"]]["seed"] == control["sampling_seed"]
    row = copy.deepcopy(control)
    row["control_prediction_id"] = control["id"]
    row["id"] = row["parent_record_id"] + "::quote80_sft_adapter_probe_v1"
    row["renderer"] = "quote80_sft_adapter_probe_v1"
    row["control_prompt_token_ids_sha256"] = pred["prompt_token_ids_sha256"]
    row["control_prompt_token_count"] = pred["prompt_token_count"]
    rows.append(row)

# Check actual tokenizer vocabulary/merges, not merely tokenizer file byte size.
control_meta = load(control_folder / "metadata.json")
control_tokenizer = Path(control_meta["adapter"]) / "tokenizer.json"
candidate_tokenizer = adapter / "tokenizer.json"
original_tok = load(control_tokenizer)
candidate_tok = load(candidate_tokenizer)
assert candidate_tok["model"]["vocab"] == original_tok["model"]["vocab"]
assert candidate_tok["model"]["merges"] == original_tok["model"]["merges"]
assert candidate_tok["added_tokens"] == original_tok["added_tokens"]

dataset_dir = repo / config["dataset_dir"]
registry_path = dataset_dir / "dataset_info.json"
registry = load(registry_path)
split_report, questions = {}, {}
all_development = {qkey(r["question"]) for r in load(run / "fixed79.current_quote.matrix.json")["rows"]}
for split, key in (("train", "dataset"), ("val", "eval_dataset")):
    path = dataset_dir / registry[config[key]]["file_name"]
    records = load(path)
    counts = Counter()
    questions[split] = set()
    for record in records:
        counts["rows"] += 1
        found_question = False
        for message in record["conversations"]:
            if message.get("from") == "human":
                match = re.search(r"(?m)^(?:question|用户原问题|问题)[:：]\s*(.+)$", message["value"])
                if match:
                    questions[split].add(qkey(match.group(1)))
                    found_question = True
            if message.get("from") != "gpt":
                continue
            try:
                value = json.loads(message["value"])
            except json.JSONDecodeError:
                counts["invalid_json_outputs"] += 1
                continue
            if not isinstance(value, dict):
                counts["non_object_outputs"] += 1
                continue
            counts["json_object_outputs"] += 1
            if value.get("next_action") == "answer_directly":
                counts["answer_directly"] += 1
                if value.get("supported_facts"):
                    counts["answer_with_supported_facts"] += 1
                if any(isinstance(f, dict) and any(isinstance(ref, dict) and "quote" in ref
                       for ref in f.get("evidence_refs", [])) for f in value.get("supported_facts", [])):
                    counts["answer_with_fact_quotes"] += 1
        if not found_question:
            counts["question_parse_failed"] += 1
    assert counts["question_parse_failed"] == 0, f"Question extraction incomplete: {split}"
    split_report[split] = {
        "path": str(path), "sha256": sha(path), "counts": dict(counts),
        "distinct_questions": len(questions[split]),
        "fixed79_question_overlap": sorted(questions[split] & all_development),
        "strict9_question_overlap": sorted(questions[split] & {qkey(r["question"]) for r in rows}),
    }
manifest = {
    "evaluation_only": True, "training_ready": False, "production_changed": False,
    "selected_adapter": str(adapter), "base_model": str(base),
    "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
    "adapter_config_sha256": sha(adapter / "adapter_config.json"),
    "trainer_state_sha256": sha(adapter / "trainer_state.json"),
    "training_config": str(config_path), "training_config_sha256": sha(config_path),
    "registry_sha256": sha(registry_path), "recorded_dataset_splits": split_report,
    "recorded_train_val_question_overlap": sorted(questions["train"] & questions["val"]),
    "tokenizer_vocab_merges_added_tokens_match_control": True,
    "candidate_tokenizer_sha256": sha(candidate_tokenizer), "control_tokenizer_sha256": sha(control_tokenizer),
    "control_matrix_sha256": sha(control_matrix_path),
    "control_predictions_sha256": sha(control_predictions_path),
    "control_metadata_sha256": sha(control_folder / "metadata.json"),
    "control_base_assets": control_meta["base_model_assets"],
    "limitations": [
        "Existing completed checkpoint only; no training or deployment is performed.",
        "Recorded data/config lineage is not authenticated historical cache provenance.",
        "Schema compatibility does not establish correct labels or semantic answers.",
        "Training/evaluation question overlaps are retained explicitly; this is not independent generalization.",
        "Old controls are the recent explicitly seeded baseline generations, not unseeded or different-engine results.",
    ],
}
with (run / "quote80_sft_adapter_probe_manifest_v1.json").open("x") as output:
    output.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
matrix = {
    "version": "quote80_sft_adapter_probe_v1", "evaluation_only": True, "training_ready": False,
    "manifest_sha256": sha(run / "quote80_sft_adapter_probe_manifest_v1.json"),
    "rows": rows, "builder_sha256": sha(Path(__file__)),
}
with (run / "quote80_sft_adapter_probe9_v1.matrix.json").open("x") as output:
    output.write(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "candidate": str(adapter), "rows": len(rows),
    "dataset_splits": split_report,
    "train_val_question_overlap_count": len(manifest["recorded_train_val_question_overlap"]),
    "production_changed": False,
}, ensure_ascii=False))
