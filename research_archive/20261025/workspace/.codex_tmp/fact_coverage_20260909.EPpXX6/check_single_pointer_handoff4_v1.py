"""Verify actual4B/LoRA dispatch, exact source annotation and deterministic binding."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save
from run_single_pointer_handoff4_v1 import (
    MATRIX, PREFLIGHT, OUTPUT, RECEIPT, VALIDATION, CONTROL, configure, base, build_row)
from analyze_single_pointer_handoff4_v1 import analyze


def main():
    destination = HERE / "single_pointer_handoff4_postflight_v1.json"
    replaypath = HERE / "single_pointer_handoff4_grounding_replay_v1.json"
    assert not destination.exists() and not replaypath.exists()
    configure()
    pre, launch = load(PREFLIGHT), load(RECEIPT)
    base.unchanged(pre, assets=True)
    assert launch["returncode"] == 0 and launch["preflight_sha256"] == sha(PREFLIGHT)
    assert launch["idle_observations"][-1]["idle_seconds"] >= 60
    assert launch["base_assets_after_equal"] and launch["adapter_assets_after_equal"]
    dispatch = load(OUTPUT / "actual_dispatch_v1.json")
    metadata = load(OUTPUT / "metadata.json")
    predictions = load(OUTPUT / "predictions.json")
    progress = [json.loads(line) for line in (OUTPUT / "predictions.progress.jsonl").read_text().splitlines()]
    assert progress == predictions
    assert dispatch["completed"] and dispatch["preflight_sha256"] == sha(PREFLIGHT)
    assert dispatch["predictions_sha256"] == metadata["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    assert metadata["runtime_source_root"] == str(SOURCE) and metadata["requested_context_override"] == 12288
    assert metadata["base_model_assets"] == pre["base_assets"] and metadata["adapter_assets"] == pre["adapter_assets"]
    assert metadata["decoding"]["max_new_tokens"] == 2048
    assert metadata["decoding"]["temperature"] == 0.1 and metadata["decoding"]["top_p"] == 0.8
    assert len(dispatch["engine_constructor"]) == 1
    constructor = dispatch["engine_constructor"][0]
    assert constructor["enable_lora"] is True and constructor["max_model_len"] == 12288
    assert Path(constructor["model"]).resolve() == Path(pre["base_path"]).resolve()
    matrix = load(MATRIX)
    rows, controls = matrix["rows"], load(CONTROL)["rows"]
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    assert len(rows) == len(predictions) == len(dispatch["rows"]) == len(pre["rows"]) == len(controls) == 4
    for row, pred, call, check, control in zip(rows, predictions, dispatch["rows"], pre["rows"], controls):
        assert row == build_row(control, render_qwen_chat_prompt)
        assert row["id"] == pred["id"] == call["id"] == check["id"]
        assert pred["actual_prompt"] == row["runtime_prompt"]
        assert pred["actual_prompt_sha256"] == row["runtime_prompt_sha256"] == call["prompt_sha256"]
        assert pred["prompt_token_ids_sha256"] == check["input_token_ids_sha256"]
        assert pred["prompt_token_count"] == check["input_tokens"]
        assert call["seed"] == row["sampling_seed"] == check["seed"] == control["sampling_seed"]
        assert call["lora_id"] == 1 and call["returned_lora_echo"] == [1] and call["status"] == "returned"
        diag = pred["runtime_generation_diagnostics"]
        assert diag["input_truncated"] is diag["output_budget_reduced"] is False
        assert diag["max_model_len"] == 12288 and diag["requested_max_tokens"] == 2048
        assert diag["prompt_tokens"] == check["input_tokens"]
        assert diag["finish_reason"] == pred["finish_reason"]
        assert diag["generated_tokens"] == pred["generated_token_count"]
        assert check["input_tokens"] + 2048 <= 12288
    results, summary = analyze(matrix, predictions)
    analysis = load(VALIDATION)
    assert analysis["results"] == results and analysis["summary"] == summary
    save(replaypath, {"results": results, "summary": summary, "new_model_calls": 0,
                      "matrix_sha256": sha(MATRIX), "predictions_sha256": sha(OUTPUT / "predictions.json")})
    base.unchanged(pre)
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "protocol_verified": True,
        "states": 4, "independent_questions": 1, "actual_seed_and_lora_verified": True,
        "all_inputs_untruncated": True, "all_output_budgets_unreduced": True,
        "original_evidence_question_seed_retained": True, "deterministic_binding_and_grounding_reproduced": True,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in predictions)),
        "summary": summary, "new_certified_answer_accuracy": None,
        "artifacts_sha256": {str(p): sha(p) for p in [
            MATRIX, PREFLIGHT, RECEIPT, VALIDATION, replaypath, OUTPUT / "predictions.json",
            OUTPUT / "metadata.json", OUTPUT / "actual_dispatch_v1.json", CONTROL,
        ]},
        "checker_sha256": sha(Path(__file__)), "models_or_source_changed": False,
    })
    print(json.dumps({"protocol_verified": True, "states": 4, "questions": 1,
                      "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
