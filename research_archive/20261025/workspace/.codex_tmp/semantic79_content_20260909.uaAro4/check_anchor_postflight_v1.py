"""Postflight for old-answer ablation, without altering its frozen matrix."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SEM = RUN / "semantic79_content_v2.fPkkhM"
OUT = HERE / "anchor_ablation_paired8_ctx12288_v1"
SOURCE = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
sys.path.insert(0, str(RUN))
from generate_exx_predictions_transformers import model_asset_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    destination = HERE / "anchor_ablation_postflight_v1.json"
    assert not destination.exists()
    metadata = load(OUT / "metadata.json")
    seeds = load(OUT / "sampling_seed_override.json")
    launch = load(HERE / "anchor_ablation_launch_v1.json")
    preflight = load(HERE / "slot_feedback_preflight_v1.json")
    matrix_path = HERE / "anchor_ablation_paired8_v1.matrix.json"
    matrix = load(matrix_path)
    analysis_path = HERE / "anchor_ablation_validation_v1.json"
    analysis = load(analysis_path)
    source = {r["id"]: r for r in load(RUN / "fixed79.current_quote.matrix.json")["rows"]}
    assert metadata["completed_rows"] == 8 and metadata["status"] == "completed"
    assert metadata["length_outputs"] == 0 and metadata["input_truncated"] is False
    assert launch["returncode"] == 0 and launch["source_unchanged"]
    assert sha(OUT / "predictions.json") == metadata["predictions_sha256"] == seeds["predictions_sha256"]
    assert seeds["completed"] and seeds["matrix_sha256"] == preflight["matrix_sha256"] == sha(matrix_path)
    rows = {r["id"]: r for r in matrix["rows"]}
    assert len(rows) == len(seeds["rows"]) == len(analysis["results"]) == 8
    assert set(rows) == {r["id"] for r in analysis["results"]}
    pairs = defaultdict(list)
    for row in matrix["rows"]:
        parent = source[row["parent_record_id"]]
        assert row["evidence_brief"] == parent["evidence_brief"]
        assert row["evidence_sha256"] == parent["evidence_sha256"]
        assert row["system"] == parent["system"] and row["user"].startswith(parent["user"])
        expected_seed = int(hashlib.sha256(
            ("old-answer-context-ablation-v1:" + row["parent_record_id"]).encode(),
        ).hexdigest()[:8], 16) % (2**31)
        assert expected_seed == row["sampling_seed"]
        pairs[row["parent_record_id"]].append(row)
    for receipt in seeds["rows"]:
        row = rows[receipt["id"]]
        assert receipt["seed"] == row["sampling_seed"]
        assert receipt["prompt_sha256"] == row["runtime_prompt_sha256"]
        assert receipt["prompt_sha256"] == hashlib.sha256(row["runtime_prompt"].encode()).hexdigest()
    assert len(pairs) == 4 and all(len(p) == 2 for p in pairs.values())
    assert sum(p[0]["renderer"] == "no_previous_answer_v1" for p in pairs.values()) == 2
    old_answers = {r["id"]: r["grounded"]["answer"] for r in load(SEM / "canonical_fixed79_validation_v1.json")["results"] if r["status"] == "ok"}
    for parent_id, pair in pairs.items():
        by_variant = {r["renderer"]: r for r in pair}
        empty = by_variant["no_previous_answer_v1"]["user"]
        anchored = by_variant["with_previous_answer_v1"]["user"]
        old_json = json.dumps({"previous_answer": old_answers[parent_id]}, ensure_ascii=False)
        assert empty.count('{"previous_answer": ""}') == 1
        assert anchored == empty.replace('{"previous_answer": ""}', old_json)
    lengths = {r["id"]: r["input_tokens"] for r in preflight["rows"]}
    assert all(r["input_tokens"] == lengths[r["id"]] for r in analysis["results"])
    assert all(r["fits_without_truncation"] for r in preflight["rows"])
    source_hashes = {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}
    assert source_hashes == preflight["source_hashes"]
    assets = {}
    for name, path_key in (("base_model_assets", "base_model"), ("adapter_assets", "adapter")):
        assets[name] = model_asset_manifest(Path(metadata[path_key]))
        assert assets[name] == metadata[name]
    # Preserve, rather than rewrite, a copied provenance field with old seeds.
    old_selection = load(SEM / "slot_feedback_paired8_v1.matrix.json")["selection"]
    assert matrix["selection"] == old_selection
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "rows": 8, "pairs": 4, "input_truncated": False,
        "length_outputs": 0, "runtime_source_unchanged": True,
        "base_and_adapter_assets_unchanged": True, "assets": assets,
        "actual_seeds_and_prompts_verified": True, "balanced_order_verified": True,
        "within_pair_only_previous_answer_changed": True,
        "matrix_sha256": sha(matrix_path), "analysis_sha256": sha(analysis_path),
        "predictions_sha256": sha(OUT / "predictions.json"),
        "seed_receipt_sha256": sha(OUT / "sampling_seed_override.json"),
        "post_compute_processes": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ], text=True, timeout=20),
        "mechanical_summary": analysis["summary"],
        "provenance_clarifications": [
            "matrix.selection is the copied previous experiment selection, including its old seed field.",
            "Actual new per-request seeds are row.sampling_seed and sampling_seed_override.json; both checked against the predeclared anchor-specific seed formula.",
            "slot_feedback_preflight_v1.json is an inherited filename in this isolated directory; its matrix hash and rows are this anchor ablation.",
            "Frozen source matrix was not edited to correct these naming/provenance ambiguities.",
        ],
        "limitations": [
            "Four selected development questions, no new retrieval and no production-accuracy estimate.",
            "No human-gold labels or source corrections were supplied to the generator.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"postflight": str(destination), "assets_unchanged": True, "summary": analysis["summary"]}))


if __name__ == "__main__":
    main()
