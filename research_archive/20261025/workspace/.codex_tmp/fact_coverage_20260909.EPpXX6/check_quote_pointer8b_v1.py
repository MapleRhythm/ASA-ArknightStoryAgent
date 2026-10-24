"""Postflight validates dispatch, lossless annotation and fresh deterministic replay."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, load, sha
from verify_empty_quote_candidate_v1 import save
from run_quote_pointer8b_v1 import (
    MATRIX, PREFLIGHT, OUTPUT, RECEIPT, VALIDATION, CONTROL_REPLAY,
    configure_base, ensure_frozen,
)
from analyze_quote_pointer8b_v1 import analyze


def main():
    destination = HERE / "quote_pointer8b58_generation_postflight_v1.json"
    assert not destination.exists()
    configure_base()
    pre, metadata, receipt = load(PREFLIGHT), load(OUTPUT / "metadata.final.json"), load(RECEIPT)
    ensure_frozen(pre)
    assert metadata["completed"] and receipt["returncode"] == 0
    assert metadata["preflight_sha256"] == receipt["preflight_sha256"] == sha(PREFLIGHT)
    assert receipt["idle_observations"][-1]["idle_seconds"] >= 60
    engine = metadata["actual_engine_constructor"]
    assert engine["enable_lora"] is False
    assert engine["model"] == engine["tokenizer"] == pre["runner_kwargs"]["base_model_path"]
    assert engine["max_model_len"] == 12288 and engine["gpu_memory_utilization"] == 0.90
    assert metadata["model_assets_after_equal"] and metadata["canonical_source_after_equal"]
    predictions = load(OUTPUT / "predictions.json")
    progress = [json.loads(line) for line in (OUTPUT / "predictions.progress.jsonl").read_text().splitlines()]
    assert predictions == progress and metadata["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    rows = load(MATRIX)["rows"]
    assert len(rows) == len(predictions) == len(metadata["requests"]) == len(pre["rows"]) == 58
    for row, pred, request, prepared in zip(rows, predictions, metadata["requests"], pre["rows"]):
        assert row["id"] == pred["id"] == request["id"] == prepared["id"]
        assert request["status"] == "returned" and request["seed"] == row["sampling_seed"]
        assert request["lora_request"] is request["returned_lora_echo"] is None
        assert pred["actual_prompt"] == row["runtime_prompt"]
        assert request["prompt_sha256"] == pred["actual_prompt_sha256"] == row["runtime_prompt_sha256"]
        assert request["prompt_token_ids_sha256"] == pred["prompt_token_ids_sha256"] == prepared["prompt_token_ids_sha256"]
        assert pred["prompt_token_count"] == prepared["input_tokens"]
        diagnostics = pred["runtime_generation_diagnostics"]
        assert diagnostics["input_truncated"] is diagnostics["output_budget_reduced"] is False
        assert diagnostics["prompt_tokens"] + 2048 <= 12288
        assert diagnostics["requested_max_tokens"] == 2048 and diagnostics["max_model_len"] == 12288
        assert diagnostics["finish_reason"] == pred["finish_reason"]
        assert diagnostics["generated_tokens"] == pred["generated_token_count"]
    report = load(VALIDATION)
    assert report["matrix_sha256"] == sha(MATRIX) and report["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    results, summary = analyze(load(MATRIX), predictions)
    assert results == report["results"] and summary == report["summary"]
    controls = load(CONTROL_REPLAY)["results"]
    assert all(a["question"] == b["question"] and a["evidence_brief"] == b["evidence_brief"]
               for a, b in zip(controls, results))
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "protocol_verified": True, "states": 58,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
        "mechanical_summary": summary,
        "binding_failures": [{"id": r["id"], "error_stage": r["error_stage"], "error": r["error"]}
                             for r in results if r["status"] != "ok"],
        "native_control_source_evidence_identical": True,
        "original_quote_gates_retained": True, "deterministic_analysis_reproduced": True,
        "certified_answer_accuracy": None, "default_model_changed": False,
        "artifacts_sha256": {str(p): sha(p) for p in [
            MATRIX, PREFLIGHT, OUTPUT / "metadata.final.json", OUTPUT / "predictions.json",
            RECEIPT, VALIDATION, CONTROL_REPLAY,
        ]},
        "checker_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"protocol_verified": True, "mechanical_summary": summary},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
