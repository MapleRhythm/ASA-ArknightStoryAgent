"""Preserve real model output; separately bind pointers before normal grounding."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, RUN, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save
from quote_pointer_protocol_v1 import bind, strict_loads
from run_quote_pointer8b_v1 import MATRIX, PREFLIGHT, OUTPUT, VALIDATION, configure_base, ensure_frozen


def analyze(matrix, predictions):
    sys.path.insert(0, str(RUN))
    sys.path.insert(0, str(SOURCE))
    from analyze_canonical_grounding_v4 import matched_rows, round_state, summarize
    from asa_arknight_story_agent.inference.generation.conclusion_generation import parse_conclusion_output
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    results = []
    for row, pred in matched_rows(matrix, predictions):
        current, maximum = round_state(row)
        item = {k: row[k] for k in (
            "id", "source_id", "question", "renderer", "template", "evidence_brief",
            "evidence_id_to_corpus_id", "packing_trace",
        )}
        item.update(
            current_round=current, max_retrieval_rounds=maximum,
            raw_output=pred["raw_output"], finish_reason=pred["finish_reason"],
            input_tokens=pred["prompt_token_count"], generated_tokens=pred["generated_token_count"],
            actual_total_tokens=pred["prompt_token_count"] + pred["generated_token_count"],
            generation_seconds=pred["generation_seconds"], normalized_strict_json=False,
            source_catalog_sha256=digest(json.dumps(row["quote_catalog"], ensure_ascii=False, sort_keys=True)),
        )
        stage = "strict_pointer_json"
        try:
            value = strict_loads(pred["raw_output"])
            item["normalized_strict_json"] = isinstance(value, dict)
            stage = "source_pointer_resolution"
            expanded = bind(value, row["evidence_brief"], row["quote_catalog"])
            item["pointer_payload"] = value
            item["expanded_payload"] = expanded
            normalized = json.dumps(expanded, ensure_ascii=False)
            item["normalized_output"] = normalized
            hypothesis = HypothesisDocument(**row["hypothesis"])
            stage = "parsing"
            parsed = parse_conclusion_output(normalized, question=row["question"],
                                            current_hypothesis=hypothesis, max_round_reached=current >= maximum)
            item["parsed"] = asdict(parsed)
            stage = "grounding"
            grounded = validate_conclusion_grounding(
                question=row["question"], hypothesis=hypothesis,
                evidence=deepcopy(row["prompt_evidence"]), conclusion=deepcopy(parsed),
                max_round_reached=current >= maximum, mode="quote",
                evidence_prompt_text=row["evidence_brief"])
            item.update(grounded=asdict(grounded), status="ok")
        except Exception as exc:
            item.update(status="parse_or_validation_error", error_stage=stage,
                        error=f"{type(exc).__name__}: {exc}")
        results.append(item)
    return results, summarize(results)


def main():
    assert not VALIDATION.exists()
    configure_base()
    ensure_frozen(load(PREFLIGHT))
    metadata = load(OUTPUT / "metadata.final.json")
    assert metadata["completed"] and metadata["predictions_sha256"] == sha(OUTPUT / "predictions.json")
    results, summary = analyze(load(MATRIX), load(OUTPUT / "predictions.json"))
    assert len(results) == 58
    save(VALIDATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "grounding_mode": "quote",
        "runtime_source_root": str(SOURCE), "matrix_sha256": sha(MATRIX),
        "predictions_sha256": sha(OUTPUT / "predictions.json"),
        "analyzer_sha256": sha(Path(__file__)),
        "summary": summary, "results": results,
        "limits": [
            "raw_output is the real model output; expanded_payload is programmatically bound, not generated.",
            "No recovery for malformed JSON, bad IDs, cross-E pointers or missing refs; full58 denominator.",
            "All original visible characters were annotated, not truncated.",
            "Original quote length/matching/token/echo gates unchanged; semantic correctness not certified.",
        ],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
