"""Verify all58 actual4B/LoRA requests after removing only reasoning metadata."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from launch_reasoning_context_ablation_v1 import (
    HERE, RUN, MATRIX, PREFLIGHT, OUTPUT, RECEIPT, VALIDATION, unchanged,
)
from build_fact_coverage_pairs_v1 import load, sha, digest
from build_reasoning_context_ablation_v1 import remove_reasoning_context
from verify_empty_quote_candidate_v1 import save


def main():
    destination = HERE / "without_reasoning_hints58_generation_postflight_v1.json"
    assert not destination.exists()
    pre, receipt = load(PREFLIGHT), load(RECEIPT)
    unchanged(pre)
    assert receipt["returncode"] == 0 and receipt["preflight_sha256"] == sha(PREFLIGHT)
    assert receipt["idle_observations"][-1]["idle_seconds"] >= 60
    assert receipt["base_assets_after_equal"] and receipt["adapter_assets_after_equal"]
    metadata = load(OUTPUT / "metadata.json")
    assert metadata["base_model_assets"] == pre["base_assets"]
    assert metadata["adapter_assets"] == pre["adapter_assets"]
    assert metadata["base_model"] == pre["base_path"] and metadata["adapter"] == pre["adapter_path"]
    dispatch = load(OUTPUT / "actual_dispatch_v1.json")
    assert dispatch["completed"] and dispatch["matrix_sha256"] == sha(MATRIX)
    predictions = load(OUTPUT / "predictions.json")
    assert dispatch["predictions_sha256"] == metadata["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    assert predictions == [json.loads(line) for line in (OUTPUT / "predictions.progress.jsonl").read_text().splitlines()]
    matrix = load(MATRIX)
    originals = [r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                 if r["renderer"] == "original_prompt_v1"]
    assert len(matrix["rows"]) == len(predictions) == len(dispatch["rows"]) == len(originals) == 58
    sys.path.insert(0, str(RUN))
    from analyze_canonical_grounding_v4 import matched_rows
    for (row, prediction), request, item, original in zip(
            matched_rows(matrix, predictions), dispatch["rows"], pre["rows"], originals):
        assert row["id"] == request["id"] == item["id"]
        assert request["status"] == "returned" and request["lora_id"] == 1 and request["returned_lora_echo"] == [1]
        assert row["sampling_seed"] == request["seed"] == original["sampling_seed"] == item["seed"]
        assert row["user"] == remove_reasoning_context(original)[0]
        for key in ("question", "hypothesis", "prompt_evidence", "evidence_brief", "current_round",
                    "max_retrieval_rounds", "system", "assistant_prefix"):
            assert row[key] == original[key]
        assert request["prompt_sha256"] == prediction["actual_prompt_sha256"] == row["runtime_prompt_sha256"]
        assert prediction["prompt_token_ids_sha256"] == item["prompt_token_ids_sha256"]
        assert prediction["prompt_token_count"] == item["input_tokens"]
        assert prediction["input_token_ids_match_untruncated_prompt"] and not prediction["reserved_output_over_context"]
        diagnostics = prediction["runtime_generation_diagnostics"]
        assert diagnostics["input_truncated"] is diagnostics["output_budget_reduced"] is False
        assert diagnostics["requested_max_tokens"] == 2048 and diagnostics["max_model_len"] == 12288
    analysis = load(VALIDATION)
    assert analysis["matrix_sha256"] == sha(MATRIX) and analysis["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    assert [r["id"] for r in analysis["results"]] == [r["id"] for r in matrix["rows"]]
    baseline = {r["id"]: r for r in load(HERE / "token_rejection_new116_replay_v1.json")["results"]}
    pairs = []
    for row, candidate in zip(matrix["rows"], analysis["results"]):
        original = baseline[row["parent_record_id"] + "::original_prompt_v1"]
        assert original["question"] == candidate["question"] and original["evidence_brief"] == candidate["evidence_brief"]
        pairs.append({
            "parent_id": row["parent_record_id"], "question": row["question"],
            "original_action": original.get("grounded", {}).get("next_action", "pipeline_error"),
            "candidate_action": candidate.get("grounded", {}).get("next_action", "pipeline_error"),
            "shown_answers_equal": original.get("grounded", {}).get("answer") == candidate.get("grounded", {}).get("answer"),
        })
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_verified": True,
        "evaluation_only": True, "training_ready": False, "states": 58,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
        "mechanical_summary": analysis["summary"], "paired_actions": pairs,
        "assets_and_source_unchanged": True, "downstream_hypotheses_unchanged": True,
        "certified_answer_accuracy": None,
        "artifacts_sha256": {str(p): sha(p) for p in [
            MATRIX, PREFLIGHT, RECEIPT, OUTPUT / "metadata.json", OUTPUT / "actual_dispatch_v1.json",
            OUTPUT / "predictions.json", VALIDATION, HERE / "token_rejection_new116_replay_v1.json",
        ]},
        "checker_sha256": sha(Path(__file__)), "limitations": matrix["predeclared_protocol"],
    }
    save(destination, result)
    print(json.dumps({"protocol_verified": True, "mechanical_summary": result["mechanical_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
