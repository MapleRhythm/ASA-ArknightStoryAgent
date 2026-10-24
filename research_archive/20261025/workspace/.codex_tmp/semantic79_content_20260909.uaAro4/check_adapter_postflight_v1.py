"""Audit actual paired adapter requests; never change frozen inputs."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SOURCE = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
OUT = HERE / "adapter_pairs12_ctx12288_v1"
sys.path.insert(0, str(RUN))
from generate_exx_predictions_transformers import model_asset_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    destination = HERE / "adapter_pairs_postflight_v1.json"
    assert not destination.exists()
    metadata = load(OUT / "metadata.json")
    receipts = load(OUT / "adapter_seed_overrides_v1.json")
    preflight = load(HERE / "adapter_pairs_preflight_v1.json")
    launch = load(HERE / "adapter_pairs_launch_v1.json")
    matrix_path = HERE / "adapter_pairs12_v1.matrix.json"
    matrix = load(matrix_path)
    analysis_path = HERE / "adapter_pairs_validation_v1.json"
    analysis = load(analysis_path)
    source = {r["id"]: r for r in load(RUN / "fixed79.current_quote.matrix.json")["rows"]}
    assert metadata["completed_rows"] == 12 and metadata["status"] == "completed"
    assert metadata["length_outputs"] == 0 and metadata["input_truncated"] is False
    assert launch["returncode"] == 0 and launch["source_unchanged"]
    assert receipts["completed"] and len(receipts["rows"]) == 12
    assert sha(OUT / "predictions.json") == metadata["predictions_sha256"] == receipts["predictions_sha256"] == analysis["predictions_sha256"]
    assert sha(matrix_path) == preflight["matrix_sha256"] == receipts["matrix_sha256"] == analysis["matrix_sha256"]
    assert sha(HERE / "run_adapter_pairs_v1.py") == receipts["wrapper_sha256"]
    assert sha(HERE / "build_adapter_pairs_v1.py") == matrix["builder_sha256"]
    for path, digest in preflight["script_sha256"].items():
        assert sha(Path(path)) == digest
    assert {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")} == preflight["source_hashes"]
    rows = matrix["rows"]
    assert [r["id"] for r in rows] == [r["id"] for r in receipts["rows"]]
    assert {r["id"] for r in rows} == {r["id"] for r in analysis["results"]}
    assets = {}
    for name, path_key in (("base_model_assets", "base_model"), ("adapter_assets", "adapter")):
        assets[name] = model_asset_manifest(Path(metadata[path_key]))
        assert assets[name] == metadata[name]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(preflight["tokenizer_path"], trust_remote_code=True, local_files_only=True)
    lengths = {r["id"]: r for r in preflight["rows"]}
    results = {r["id"]: r for r in analysis["results"]}
    pairs = defaultdict(list)
    verified = []
    for row, receipt in zip(rows, receipts["rows"]):
        parent = source[row["parent_record_id"]]
        for key in ("runtime_prompt", "runtime_prompt_sha256", "evidence_brief", "evidence_sha256", "system", "user"):
            assert row[key] == parent[key]
        expected_seed = int(hashlib.sha256(("adapter-weight-ablation-v1:" + row["parent_record_id"]).encode()).hexdigest()[:8], 16) % (2**31)
        assert receipt["seed"] == row["sampling_seed"] == expected_seed
        assert receipt["status"] == "returned"
        assert receipt["adapter_active_at_call"] is row["use_adapter"]
        assert receipt["configured_lora_id"] == 1
        expected_id = 1 if row["use_adapter"] else None
        assert receipt["effective_lora_id"] == expected_id
        assert receipt["returned_lora_echo"] == [{"available": True, "lora_id": expected_id}]
        assert receipt["prompt_sha256"] == row["runtime_prompt_sha256"] == hashlib.sha256(row["runtime_prompt"].encode()).hexdigest()
        token_ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        token_sha = hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()
        assert token_sha == lengths[row["id"]]["prompt_token_ids_sha256"]
        assert len(token_ids) == lengths[row["id"]]["input_tokens"] == results[row["id"]]["input_tokens"]
        assert len(token_ids) + 2048 <= 12288
        pairs[row["parent_record_id"]].append(row)
        verified.append({"id": row["id"], "seed": expected_seed, "adapter_active": row["use_adapter"], "returned_lora_id": expected_id, "token_ids_sha256": token_sha})
    assert len(pairs) == 6 and all(len(pair) == 2 for pair in pairs.values())
    assert sum(not pair[0]["use_adapter"] for pair in pairs.values()) == 3
    for pair in pairs.values():
        assert {row["use_adapter"] for row in pair} == {False, True}
        assert pair[0]["runtime_prompt"] == pair[1]["runtime_prompt"]
        assert pair[0]["sampling_seed"] == pair[1]["sampling_seed"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "rows": 12, "pairs": 6,
        "source_unchanged": True, "base_and_adapter_assets_unchanged": True,
        "token_ids_recomputed": True, "balanced_order_verified": True,
        "actual_adapter_calls_and_returned_echo_verified": True,
        "within_pair_original_prompt_and_seed_identical": True,
        "assets": assets, "verified_rows": verified, "mechanical_summary": analysis["summary"],
        "matrix_sha256": sha(matrix_path), "analysis_sha256": sha(analysis_path),
        "predictions_sha256": sha(OUT / "predictions.json"),
        "receipt_sha256": sha(OUT / "adapter_seed_overrides_v1.json"),
        "limitations": [
            "Six selected development questions, not holdout or fresh end-to-end retrieval.",
            "Base condition uses the same LoRA-enabled engine and tokenizer with request adapter disabled.",
            "Metadata decoding seed/runtime mode describes the shared engine; verified per-request overrides are authoritative.",
            "Mechanical release counts are not semantic accuracy or permission to promote a model.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"postflight": str(destination), "verified_pairs": 6, "assets_unchanged": True}))


if __name__ == "__main__":
    main()
