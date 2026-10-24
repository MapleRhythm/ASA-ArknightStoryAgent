"""Verify the completed experiment, without rerunning inference or altering labels."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SOURCE = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
OUT = HERE / "slot_feedback_paired8_ctx12288_v1"
sys.path.insert(0, str(RUN))
from generate_exx_predictions_transformers import model_asset_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    destination = HERE / "slot_feedback_postflight_v1.json"
    comparison_path = HERE / "slot_feedback_comparison_v1.json"
    assert not destination.exists() and not comparison_path.exists()
    metadata = load(OUT / "metadata.json")
    seed = load(OUT / "sampling_seed_override.json")
    preflight = load(HERE / "slot_feedback_preflight_v1.json")
    launch = load(HERE / "slot_feedback_launch_v1.json")
    matrix = load(HERE / "slot_feedback_paired8_v1.matrix.json")
    analysis = load(HERE / "slot_feedback_validation_v1.json")
    audit = load(HERE / "slot_feedback_content_audit_v1.json")
    assert metadata["status"] == "completed" and metadata["completed_rows"] == 8
    assert metadata["length_outputs"] == 0 and metadata["input_truncated"] is False
    assert metadata["predictions_sha256"] == seed["predictions_sha256"] == sha(OUT / "predictions.json")
    assert launch["returncode"] == 0 and launch["source_unchanged"]
    assert seed["completed"] and seed["matrix_sha256"] == sha(HERE / "slot_feedback_paired8_v1.matrix.json")
    rows = {r["id"]: r for r in matrix["rows"]}
    assert len(rows) == len(analysis["results"]) == len(audit["results"]) == len(seed["rows"]) == 8
    assert set(rows) == {r["id"] for r in analysis["results"]} == {r["id"] for r in audit["results"]}
    pair_seeds = defaultdict(set)
    for receipt in seed["rows"]:
        row = rows[receipt["id"]]
        assert row["sampling_seed"] == receipt["seed"]
        assert row["runtime_prompt_sha256"] == receipt["prompt_sha256"]
        assert hashlib.sha256(row["runtime_prompt"].encode()).hexdigest() == receipt["prompt_sha256"]
        pair_seeds[row["parent_record_id"]].add(receipt["seed"])
    assert len(pair_seeds) == 4 and all(len(s) == 1 for s in pair_seeds.values())
    expected_lengths = {r["id"]: r["input_tokens"] for r in preflight["rows"]}
    assert all(r["input_tokens"] == expected_lengths[r["id"]] for r in analysis["results"])
    assert all(r["fits_without_truncation"] for r in preflight["rows"])
    current_source = {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}
    assert current_source == preflight["source_hashes"]
    manifests = {}
    for key, location in (("base_model_assets", "base_model"), ("adapter_assets", "adapter")):
        manifests[key] = model_asset_manifest(Path(metadata[location]))
        assert manifests[key] == metadata[key], f"Changed model assets: {location}"
    compute = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, timeout=20)
    # Other users may reuse an idle GPU after completion; only record current use.
    currently_in_use = any(line.split(",")[0].strip() == launch["gpu_uuid"] for line in compute.splitlines())
    inputs = [
        OUT / "metadata.json", OUT / "sampling_seed_override.json", OUT / "predictions.json",
        HERE / "slot_feedback_preflight_v1.json", HERE / "slot_feedback_launch_v1.json",
        HERE / "slot_feedback_paired8_v1.matrix.json", HERE / "slot_feedback_validation_v1.json",
        HERE / "slot_feedback_content_audit_v1.json",
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "new_generation_in_this_check": False,
        "rows": 8, "pairs": 4, "paired_seed_receipts_match": True,
        "token_lengths_match_preflight": True, "input_truncated": False, "length_outputs": 0,
        "source_unchanged": True, "base_and_adapter_assets_unchanged": True,
        "post_model_asset_manifests": manifests,
        "gpu_uuid": launch["gpu_uuid"], "gpu_currently_in_use": currently_in_use,
        "compute_processes_at_check": compute,
        "artifacts_sha256": {str(p): sha(p) for p in inputs},
        "checker_sha256": sha(Path(__file__)),
        "limitations": [
            "metadata.decoding.seed is inherited boilerplate; sampling_seed_override.json records the actual explicit per-request seeds.",
            "All four feedback variants happened to run before their controls; order is not balanced.",
            "No new retrieval; four selected development questions, not production accuracy.",
        ],
    }
    save(destination, report)
    audits = {r["id"]: r for r in audit["results"]}
    summary = defaultdict(lambda: {"mechanical_status": Counter(), "actions": Counter(), "auto_verdicts": Counter()})
    paired = defaultdict(list)
    for row in analysis["results"]:
        result = audits[row["id"]]
        variant = row["renderer"]
        verdict = result.get("judgement", {}).get("verdict", result["status"])
        action = row.get("grounded", {}).get("next_action", "pipeline_error")
        summary[variant]["mechanical_status"][row["status"]] += 1
        summary[variant]["actions"][action] += 1
        summary[variant]["auto_verdicts"][verdict] += 1
        paired[rows[row["id"]]["parent_record_id"]].append({
            "id": row["id"], "variant": variant, "question": row["question"],
            "pipeline_status": row["status"], "action": action,
            "shown_answer": row.get("grounded", {}).get("answer"),
            "auto_status": result["status"], "auto_verdict": verdict,
        })
    comparison = {
        "created_at_utc": report["created_at_utc"], "evaluation_only": True,
        "training_ready": False, "new_generation": True, "states": 8, "questions": 4,
        "summary": summary, "pairs": dict(paired), "postflight_sha256": sha(destination),
        "promotion": "not_promoted",
        "limitations": report["limitations"] + [
            "Night-sword feedback still does not name the crisis despite an automatic complete verdict.",
            "Question-slot checks and semantic model labels are not human gold or reliable correctness counts.",
        ],
    }
    save(comparison_path, comparison)
    print(json.dumps({"postflight": str(destination), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
