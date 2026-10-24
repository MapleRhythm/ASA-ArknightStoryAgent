"""Verify the exact treatment, actual seeds/LoRA, asset hashes and token IDs."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import (
    HERE, ROOT, RUN, SOURCE, build_pairs, digest, load, normalized_question, select_states, sha,
)
from verify_empty_quote_candidate_v1 import save, snapshot


def main():
    destination = HERE / "generation_postflight_v1.json"
    assert not destination.exists()
    output = HERE / "paired116_ctx12288_v1"
    metadata = load(output / "metadata.json")
    dispatch = load(output / "actual_dispatch_v1.json")
    preflight = load(HERE / "preflight_v1.json")
    launch = load(HERE / "launch_receipt_v1.json")
    matrix_path = HERE / "fact_coverage_pairs116_v1.matrix.json"
    matrix = load(matrix_path)
    source_path = RUN / "fixed79.current_quote.matrix.json"
    source = load(source_path)
    analysis_path = HERE / "validation_v1.json"
    analysis = load(analysis_path)
    predictions = load(output / "predictions.json")
    assert metadata["completed_rows"] == 116 and metadata["status"] == "completed"
    assert launch["returncode"] == 0 and launch["source_unchanged"] and launch["config_unchanged"]
    assert dispatch["completed"] and len(dispatch["rows"]) == 116
    assert sha(output / "predictions.json") == metadata["predictions_sha256"] == dispatch["predictions_sha256"] == analysis["predictions_sha256"]
    assert sha(matrix_path) == preflight["matrix_sha256"] == dispatch["matrix_sha256"] == analysis["matrix_sha256"]
    assert sha(source_path) == matrix["source_matrix_sha256"]
    assert sha(HERE / "build_fact_coverage_pairs_v1.py") == matrix["builder_sha256"]
    assert sha(HERE / "run_fact_coverage_pairs_v1.py") == dispatch["wrapper_sha256"]
    assert all(sha(Path(p)) == h for p, h in preflight["script_sha256"].items())
    assert snapshot(SOURCE) == preflight["source_hashes"]
    config = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    assert sha(config) == preflight["canonical_config_sha256"] == matrix["canonical_config_sha256"]
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    six_path = RUN / "adapter_pair_v1.p0RvMj/adapter_pairs12_v1.matrix.json"
    assert sha(six_path) == matrix["development_membership_source_sha256"]
    six = {normalized_question(row["question"]) for row in load(six_path)["selection"]}
    rebuilt, selection = build_pairs(select_states(source["rows"]), render_qwen_chat_prompt, six)
    assert rebuilt == matrix["rows"] and selection == matrix["selection"]
    assert Counter(row["first_variant"] for row in selection) == {
        "original_prompt_v1": 29, "fact_coverage_prompt_v1": 29,
    }
    sys.path.insert(0, str(RUN))
    from generate_exx_predictions_transformers import model_asset_manifest
    assets = {}
    for key, path_key in (("base_model_assets", "base_model"), ("adapter_assets", "adapter")):
        assets[key] = model_asset_manifest(Path(metadata[path_key]))
        assert assets[key] == metadata[key]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        preflight["tokenizer_path"], trust_remote_code=True, local_files_only=True,
    )
    assert len(predictions) == len(analysis["results"]) == len(rebuilt) == 116
    lengths = {row["id"]: row for row in preflight["rows"]}
    ids = [r["id"] for r in rebuilt]
    assert [r["id"] for r in predictions] == [r["id"] for r in dispatch["rows"]] == ids
    assert [r["id"] for r in analysis["results"]] == ids
    verified = []
    for row, receipt, pred in zip(rebuilt, dispatch["rows"], predictions):
        assert receipt["status"] == "returned" and receipt["lora_id"] == 1
        assert receipt["returned_lora_echo"] == [1]
        assert receipt["seed"] == row["sampling_seed"]
        assert receipt["prompt_sha256"] == row["runtime_prompt_sha256"] == pred["actual_prompt_sha256"]
        assert pred["actual_prompt"] == row["runtime_prompt"]
        assert digest(json.dumps(row, ensure_ascii=False, sort_keys=True)) == pred["input_row_sha256"]
        token_ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=True)
        assert digest(json.dumps(token_ids)) == lengths[row["id"]]["prompt_token_ids_sha256"] == pred["prompt_token_ids_sha256"]
        assert len(token_ids) == lengths[row["id"]]["input_tokens"] == pred["prompt_token_count"]
        assert len(token_ids) + 2048 <= 12288
        assert pred["input_token_ids_match_untruncated_prompt"] and not pred["reserved_output_over_context"]
        verified.append({
            "id": row["id"], "seed": receipt["seed"], "actual_lora_id": 1,
            "input_tokens": len(token_ids), "finish_reason": pred["finish_reason"],
        })
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "protocol_verified": True,
        "states": 116, "questions": 58, "broader_development_questions": 52,
        "all_actual_seed_and_adapter_calls_verified": True,
        "all_input_token_ids_recomputed": True, "input_truncated": False,
        "source_config_base_and_adapter_unchanged": True,
        "prompt_treatment_rebuilt_exactly": True, "balanced_order_verified": True,
        "assets": assets, "verified_rows": verified,
        "mechanical_summary": analysis["summary"],
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
        "matrix_sha256": sha(matrix_path), "analysis_sha256": sha(analysis_path),
        "predictions_sha256": sha(output / "predictions.json"),
        "dispatch_sha256": sha(output / "actual_dispatch_v1.json"),
        "limitations": [
            "Length-ended or malformed outputs, if any, are retained; protocol verification is not answer correctness.",
            "52 broader development questions were already part of the old79 audit, not unseen holdout.",
            "No retrieval or production orchestration recovery was executed.",
        ],
    })
    print(json.dumps({
        "protocol_verified": True, "states": 116, "assets_unchanged": True,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
    }), flush=True)


if __name__ == "__main__":
    main()
