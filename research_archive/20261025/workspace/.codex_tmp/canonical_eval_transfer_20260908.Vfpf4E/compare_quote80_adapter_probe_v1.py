"""Compare only hash-verified paired inputs; model-judge labels remain provisional."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from answer_semantics_content_v2 import messages, validate
from judge_visible_answers import digest


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(records):
    result = {r["id"]: r for r in records}
    assert len(result) == len(records)
    return result


def checked_verdict(analysis_row, audit, verdict):
    if verdict["status"] != "ok":
        return verdict["status"]
    payload = {"model": audit["requested_model"], "messages": messages(analysis_row),
               "temperature": 0, "max_tokens": audit["max_tokens"]}
    expected = digest({"payload": payload, "endpoint": audit["endpoint"],
                       "id": analysis_row["id"], "grounded": analysis_row["grounded"]})
    assert verdict["input_key"] == expected
    assert verdict["response_model"] == audit["requested_model"] and verdict["finish_reason"] == "stop"
    return validate(verdict["judgement"], analysis_row)["verdict"]


run = Path(__file__).resolve().parent
candidate_folder = run / "quote80_sft_adapter_probe9_ctx12288_v1"
control_folder = run / "question_focus_paired18_ctx12288_v1"
manifest = load(run / "quote80_sft_adapter_probe_manifest_v1.json")
matrix_path = run / "quote80_sft_adapter_probe9_v1.matrix.json"
matrix = load(matrix_path)
candidate_meta = load(candidate_folder / "metadata.json")
control_meta = load(control_folder / "metadata.json")
assert candidate_meta["status"] == control_meta["status"] == "completed"
for field in ("base_model_assets", "vllm", "torch", "runtime_runner_sha256"):
    assert candidate_meta[field] == control_meta[field], field
assert candidate_meta["runtime"]["max_model_len"] == control_meta["runtime"]["max_model_len"] == 12288
assert candidate_meta["adapter"] == manifest["selected_adapter"]
assert sha(control_folder / "predictions.json") == manifest["control_predictions_sha256"]
control_preds = indexed(load(control_folder / "predictions.json"))
candidate_preds = indexed(load(candidate_folder / "predictions.json"))
control_seeds = indexed(load(control_folder / "sampling_seed_override.json")["rows"])
candidate_seeds = indexed(candidate_meta["seed_calls"])
control_analysis_path = control_folder / "validation_v1.json"
candidate_analysis_path = candidate_folder / "validation_v1.json"
control_analysis = indexed(load(control_analysis_path)["results"])
candidate_analysis = indexed(load(candidate_analysis_path)["results"])
control_audit = load(control_folder / "semantic18_content_v2.json")
candidate_audit = load(candidate_folder / "semantic9_content_v2.json")
assert control_audit["source_analysis_sha256"] == sha(control_analysis_path)
assert candidate_audit["source_analysis_sha256"] == sha(candidate_analysis_path)
assert control_audit["protocol"] == candidate_audit["protocol"] == "asa_visible_answer_semantics_v2"
assert control_audit["rubric_script_sha256"] == candidate_audit["rubric_script_sha256"] == sha(run / "answer_semantics_content_v2.py")
old_verdicts, new_verdicts = indexed(control_audit["results"]), indexed(candidate_audit["results"])
assert len(matrix["rows"]) == len(candidate_preds) == len(candidate_analysis) == len(new_verdicts) == 9
rows = []
for row in matrix["rows"]:
    old_id, new_id = row["control_prediction_id"], row["id"]
    before, after = control_preds[old_id], candidate_preds[new_id]
    assert before["actual_prompt"] == after["actual_prompt"] == row["runtime_prompt"]
    assert before["prompt_token_ids_sha256"] == after["prompt_token_ids_sha256"] == row["control_prompt_token_ids_sha256"]
    assert control_seeds[old_id]["seed"] == candidate_seeds[new_id]["seed"] == after["sampling_seed"] == row["sampling_seed"]
    old, new = control_analysis[old_id], candidate_analysis[new_id]
    for field in ("question", "evidence_brief", "evidence_id_to_corpus_id", "current_round", "max_retrieval_rounds"):
        assert old[field] == new[field], field
    rows.append({
        "id": new_id, "control_id": old_id, "question": row["question"],
        "control_label": checked_verdict(old, control_audit, old_verdicts[old_id]),
        "candidate_label": checked_verdict(new, candidate_audit, new_verdicts[new_id]),
        "control_action": old.get("grounded", {}).get("next_action", "pipeline_error"),
        "candidate_action": new.get("grounded", {}).get("next_action", "pipeline_error"),
        "control_answer": old.get("grounded", {}).get("answer", ""),
        "candidate_answer": new.get("grounded", {}).get("answer", ""),
        "control_issues": old.get("grounded", {}).get("grounding_warnings", []),
        "candidate_issues": new.get("grounded", {}).get("grounding_warnings", []),
        "candidate_finish_reason": after["finish_reason"],
        "candidate_error": new.get("error") or new_verdicts[new_id].get("error"),
        "input_tokens_and_seed_match_control": True,
    })
summary = {
    "records_per_model": 9,
    "control_labels": dict(Counter(r["control_label"] for r in rows)),
    "candidate_labels": dict(Counter(r["candidate_label"] for r in rows)),
    "control_actions": dict(Counter(r["control_action"] for r in rows)),
    "candidate_actions": dict(Counter(r["candidate_action"] for r in rows)),
    "candidate_length_outputs": sum(r["candidate_finish_reason"] == "length" for r in rows),
}
report = {
    "evaluation_only": True, "training_ready": False, "production_changed": False,
    "summary": summary, "rows": rows,
    "matrix_sha256": sha(matrix_path),
    "candidate_metadata_sha256": sha(candidate_folder / "metadata.json"),
    "control_metadata_sha256": sha(control_folder / "metadata.json"),
    "candidate_audit_sha256": sha(candidate_folder / "semantic9_content_v2.json"),
    "control_audit_sha256": sha(control_folder / "semantic18_content_v2.json"),
    "comparator_sha256": sha(Path(__file__)),
    "limitations": [
        "These nine cases have been repeatedly inspected; some overlap recorded training data.",
        "Existing checkpoints differ in training data/objective/history; not a causal estimate of formatting alone.",
        "Same frozen prompt, token IDs, seeds, runtime implementation and base model; controls were generated earlier.",
        "One sample per model/case; kernel and sampling variance remain possible.",
        "Matching model-judge protocol does not make its labels human gold.",
        "Not fresh retrieval, production multi-round execution or independent final E2E accuracy.",
    ],
}
with (run / "quote80_adapter_vs_seeded_control9_v1.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False))
