"""Assemble known prefill, resolve a single Q, retain all original grounding gates."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import analyze_quote_pointer8b_v1 as shared
from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save
from single_quote_pointer_protocol_v1 import bind
from run_single_pointer_handoff4_v1 import MATRIX, PREFLIGHT, OUTPUT, VALIDATION, configure, base


def assemble(row, prediction):
    if row["assistant_prefix"] != "{":
        raise ValueError("unexpected known prefill")
    return row["assistant_prefix"] + prediction["raw_output"]


def analyze(matrix, predictions):
    rows = matrix["rows"]
    assert len(rows) == len(predictions) == 4
    converted = deepcopy(predictions)
    for row, pred in zip(rows, converted):
        assert row["id"] == pred["id"]
        pred["raw_output"] = assemble(row, pred)
    original_bind = shared.bind
    shared.bind = bind
    try:
        results, summary = shared.analyze(matrix, converted)
    finally:
        shared.bind = original_bind
    for row, pred, result in zip(rows, predictions, results):
        result["strict_json_with_known_prefill"] = assemble(row, pred)
        result["assistant_prefix"] = row["assistant_prefix"]
        result["raw_output"] = pred["raw_output"]
    return results, summary


def main():
    assert not VALIDATION.exists()
    configure()
    base.unchanged(load(PREFLIGHT))
    results, summary = analyze(load(MATRIX), load(OUTPUT / "predictions.json"))
    save(VALIDATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "grounding_mode": "quote", "runtime_source_root": str(SOURCE),
        "matrix_sha256": sha(MATRIX), "predictions_sha256": sha(OUTPUT / "predictions.json"),
        "analyzer_sha256": sha(Path(__file__)), "results": results, "summary": summary,
        "limitations": [
            "Real continuation kept under raw_output; only the already-visible assistant prefill is prepended for strict JSON.",
            "expanded_payload is deterministic catalog lookup, not generated text and not evidence-ID repair.",
            "Dual-pointer outputs are rejected, not stripped or corrected; no failed row/fact is removed.",
            "No fresh semantic judge or certified answer accuracy in this mechanical analysis.",
        ],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
