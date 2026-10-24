"""Read-only preflight of the existing quote-SFT-followup KTO checkpoint."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import yaml


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qkey(text):
    return re.sub(r"\s+", "", text).strip("？?。！!，,；;：:\"'")


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
config_path = repo / "src/config/llama_factory_student_rollout_quote80_sft_v1_200_api_verifier_v1_kto_config.yaml"
config = yaml.safe_load(config_path.read_text())
adapter = (repo / config["output_dir"]).resolve()
parent = (repo / config["adapter_name_or_path"]).resolve()
prior_manifest = load(run / "quote80_sft_adapter_probe_manifest_v1.json")
assert parent == Path(prior_manifest["selected_adapter"])
assert config["stage"] == "kto"
adapter_config = load(adapter / "adapter_config.json")
assert (repo / adapter_config["base_model_name_or_path"]).resolve() == Path(prior_manifest["base_model"])
state = load(adapter / "trainer_state.json")
assert state["global_step"] == state["max_steps"] == 370 and state["epoch"] == 2.0
dataset_dir = repo / config["dataset_dir"]
registry_path = dataset_dir / "dataset_info.json"
registry = load(registry_path)
strict_questions = {qkey(r["question"]) for r in load(run / "after.matrix.json")["rows"]}
fixed_questions = {qkey(r["question"]) for r in load(run / "fixed79.current_quote.matrix.json")["rows"]}
splits, question_sets = {}, {}
for split, registry_key in (("train", "dataset"), ("val", "eval_dataset")):
    path = dataset_dir / registry[config[registry_key]]["file_name"]
    records = load(path)
    counts = defaultdict(Counter)
    questions = set()
    signatures = defaultdict(list)
    for record in records:
        tag = record.get("kto_tag")
        counts[str(tag)]["rows"] += 1
        found = False
        for message in record["conversations"]:
            if message["from"] == "human":
                match = re.search(r"(?m)^(?:question|用户原问题|问题)[:：]\s*(.+)$", message["value"])
                if match:
                    questions.add(qkey(match.group(1)))
                    found = True
            if message["from"] != "gpt":
                continue
            try:
                payload = json.loads(message["value"])
            except json.JSONDecodeError:
                counts[str(tag)]["invalid_json_outputs"] += 1
                continue
            if not isinstance(payload, dict):
                counts[str(tag)]["non_object_outputs"] += 1
                continue
            counts[str(tag)]["json_objects"] += 1
            if payload.get("next_action") == "answer_directly":
                counts[str(tag)]["answer_directly"] += 1
                facts = payload.get("supported_facts", [])
                if isinstance(facts, list) and facts:
                    counts[str(tag)]["answer_with_supported_facts"] += 1
                if isinstance(facts, list) and any(
                    isinstance(fact, dict) and any(isinstance(ref, dict) and ref.get("quote")
                    for ref in fact.get("evidence_refs", [])) for fact in facts
                ):
                    counts[str(tag)]["answer_with_fact_quotes"] += 1
                inferred = payload.get("inferred_facts")
                if isinstance(inferred, list) and inferred:
                    counts[str(tag)]["answer_with_inferred_facts"] += 1
                    if any(not isinstance(fact, dict) or not fact.get("evidence_refs") for fact in inferred):
                        counts[str(tag)]["answer_with_uncited_inferred_fact"] += 1
        if not found:
            counts[str(tag)]["question_parse_failed"] += 1
        sample = {key: record.get(key) for key in ("system", "tools", "conversations")}
        signature = hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        signatures[signature].append({"id": record["id"], "kto_tag": tag})
    assert all(c["question_parse_failed"] == 0 for c in counts.values())
    conflicts = [{"sha256": key, "records": group} for key, group in signatures.items()
                 if {r["kto_tag"] for r in group} == {False, True}]
    question_sets[split] = questions
    splits[split] = {
        "path": str(path), "sha256": sha(path), "rows": len(records), "distinct_questions": len(questions),
        "counts_by_tag": {tag: dict(c) for tag, c in counts.items()},
        "strict9_question_overlap": sorted(questions & strict_questions),
        "fixed79_question_overlap": sorted(questions & fixed_questions),
        "identical_input_output_label_conflicts": conflicts,
    }
report = {
    "evaluation_only": True, "training_ready": False, "production_changed": False,
    "selected_adapter": str(adapter), "parent_adapter": str(parent), "base_model": prior_manifest["base_model"],
    "adapter_weights_sha256": sha(adapter / "adapter_model.safetensors"),
    "adapter_config_sha256": sha(adapter / "adapter_config.json"), "trainer_state_sha256": sha(adapter / "trainer_state.json"),
    "training_config_sha256": sha(config_path), "registry_sha256": sha(registry_path),
    "splits": splits, "train_val_question_overlap": sorted(question_sets["train"] & question_sets["val"]),
    "limitations": [
        "Preflight only: this KTO checkpoint has not been evaluated here.",
        "Stored configuration/data provenance, not authenticated historical training-cache contents.",
        "KTO negatives may deliberately be invalid; they are not positive-label corruption.",
        "Quote-field presence is not citation or semantic correctness.",
        "No model pointer, dataset or label is modified; no training is started.",
    ],
}
with (run / "quote80_kto_candidate_audit_v1.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "adapter": str(adapter), "train_val_overlap_count": len(report["train_val_question_overlap"]),
    "splits": {name: {k: v for k, v in info.items()
                     if k in ("rows", "distinct_questions", "counts_by_tag", "strict9_question_overlap",
                              "identical_input_output_label_conflicts")} for name, info in splits.items()},
}, ensure_ascii=False))
