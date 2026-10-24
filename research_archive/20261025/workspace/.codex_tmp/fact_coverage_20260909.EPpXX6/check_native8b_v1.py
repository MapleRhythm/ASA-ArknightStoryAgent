"""Check actual58 dispatches, unchanged assets/source and all-denominator replay."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from run_native8b_v1 import (
    HERE, RUN, MATRIX, PREFLIGHT, OUTPUT, RECEIPT, VALIDATION,
    ensure_frozen, load, sha, digest, save,
)


def main():
    destination = HERE / "qwen3_8b_generation_postflight_v1.json"
    assert not destination.exists()
    pre, metadata, receipt = load(PREFLIGHT), load(OUTPUT / "metadata.final.json"), load(RECEIPT)
    ensure_frozen(pre)
    assert metadata["completed"] and receipt["returncode"] == 0
    assert metadata["preflight_sha256"] == receipt["preflight_sha256"] == sha(PREFLIGHT)
    assert receipt["idle_observations"][-1]["idle_seconds"] >= 60
    assert metadata["model_assets_after_equal"] and metadata["canonical_source_after_equal"]
    engine = metadata["actual_engine_constructor"]
    assert engine["enable_lora"] is False and engine["model"] == engine["tokenizer"] == pre["runner_kwargs"]["base_model_path"]
    assert engine["max_model_len"] == 12288 and engine["gpu_memory_utilization"] == 0.90
    predictions = load(OUTPUT / "predictions.json")
    progress = [json.loads(line) for line in (OUTPUT / "predictions.progress.jsonl").read_text().splitlines()]
    assert predictions == progress and metadata["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    matrix = load(MATRIX)
    assert len(matrix["rows"]) == len(predictions) == len(metadata["requests"]) == 58
    sys.path.insert(0, str(RUN))
    from analyze_canonical_grounding_v4 import matched_rows
    checks = []
    for (row, prediction), request, item in zip(matched_rows(matrix, predictions), metadata["requests"], pre["rows"]):
        assert row["id"] == prediction["id"] == request["id"] == item["id"]
        assert request["seed"] == row["sampling_seed"] and request["status"] == "returned"
        assert request["lora_request"] is request["returned_lora_echo"] is None
        assert request["prompt_sha256"] == prediction["actual_prompt_sha256"] == row["runtime_prompt_sha256"]
        assert request["prompt_token_ids_sha256"] == item["prompt_token_ids_sha256"] == prediction["prompt_token_ids_sha256"]
        assert prediction["prompt_token_count"] == item["input_tokens"]
        assert prediction["input_token_ids_match_untruncated_prompt"]
        assert not prediction["reserved_output_over_context"]
        diagnostics = prediction["runtime_generation_diagnostics"]
        assert diagnostics["input_truncated"] is diagnostics["output_budget_reduced"] is False
        assert diagnostics["requested_max_tokens"] == 2048 and diagnostics["max_model_len"] == 12288
        assert diagnostics["finish_reason"] == prediction["finish_reason"]
        assert diagnostics["generated_tokens"] == prediction["generated_token_count"]
        assert diagnostics["prompt_tokens"] + 2048 <= 12288
        checks.append({"id": row["id"], "actual_seed": request["seed"], "no_lora": True,
                       "untruncated_input_and_budget": True, "finish_reason": prediction["finish_reason"]})
    analysis = load(VALIDATION)
    assert analysis["matrix_sha256"] == sha(MATRIX)
    assert analysis["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    assert [r["id"] for r in analysis["results"]] == [r["id"] for r in matrix["rows"]]
    assert len(analysis["results"]) == 58
    for row, result, prediction in zip(matrix["rows"], analysis["results"], predictions):
        assert row["evidence_brief"] == result["evidence_brief"]
        assert row["current_round"] == result["current_round"]
        assert row["max_retrieval_rounds"] == result["max_retrieval_rounds"]
        assert result["raw_output"] == prediction["raw_output"]
    baseline = load(HERE / "token_rejection_new116_replay_v1.json")
    old = [r for r in baseline["results"] if r["renderer"] == "original_prompt_v1"]
    assert len(old) == 58
    pairs = []
    for row, original, candidate in zip(matrix["rows"], old, analysis["results"]):
        assert original["id"] == row["parent_record_id"] + "::original_prompt_v1"
        assert original["question"] == candidate["question"]
        assert original["evidence_brief"] == candidate["evidence_brief"]
        pairs.append({
            "parent_record_id": row["parent_record_id"], "question": row["question"],
            "original_id": original["id"], "native8b_id": candidate["id"],
            "original_action": original.get("grounded", {}).get("next_action", "pipeline_error"),
            "native8b_action": candidate.get("grounded", {}).get("next_action", "pipeline_error"),
        })
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "protocol_verified": True, "states": 58,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
        "mechanical_summary": analysis["summary"], "paired_actions": pairs, "dispatch_checks": checks,
        "native_template_enable_thinking": False, "actual_lora_enabled": False,
        "assets_and_source_unchanged": True, "certified_answer_accuracy": None,
        "artifacts_sha256": {str(p): sha(p) for p in [
            PREFLIGHT, MATRIX, OUTPUT / "metadata.final.json", OUTPUT / "predictions.json",
            VALIDATION, RECEIPT, HERE / "token_rejection_new116_replay_v1.json",
        ]},
        "checker_sha256": sha(Path(__file__)),
        "limitations": pre["limitations"] + ["JSON validity or direct acceptance is not semantic correctness."],
    }
    save(destination, report)
    print(json.dumps({"protocol_verified": True, "finish_reasons": report["finish_reasons"],
                      "mechanical_summary": report["mechanical_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
