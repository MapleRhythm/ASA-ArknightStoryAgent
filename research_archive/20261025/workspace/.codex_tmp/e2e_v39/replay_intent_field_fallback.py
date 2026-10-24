"""All 58 captured planner outputs; no new model calls or production changes."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V37 = HERE.parent / "planner_entity_provenance_20260910.v37_attempt02"
sys.path.insert(0, str(V27))
import run_reader_router as shared
sys.path.insert(0, str(SRC))
from intent_field_fallback import normalize_with_intent_fallback, preserve_fallback_routing


def main():
    from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
    from asa_arknight_story_agent.inference.pipeline.constants import HYPOTHESIS_INTENTS, QUERY_TYPES
    from asa_arknight_story_agent.inference.pipeline.types import ModelOutputError, HypothesisDocument
    from asa_arknight_story_agent.inference.planning.query_understanding import detect_intent, build_hypothesis, build_retrieval_query
    assert not (HERE / "preflight.json").exists()
    pre37 = shared.load(V37 / "preflight.json")
    gen37 = shared.load(V37 / "generations.json")
    assert gen37["actual_call_count"] == 58
    assert gen37["preflight_sha256"] == shared.sha(V37 / "preflight.json")
    files = [
        *sorted(SRC.rglob("*.py")), ROOT / "indexes/arknights_story/operator_aliases.json",
        V37 / "preflight.json", V37 / "generations.json", V37 / "postflight_v1.json",
        Path(__file__), HERE / "intent_field_fallback.py", HERE / "test_intent_field_fallback.py",
    ]
    manifest = {str(path): shared.sha(path) for path in files}
    shared.save(HERE / "preflight.json", {
        "version": "initial_planner_intent_field_fallback_cpu_v39",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "production_changed": False, "training_ready": False,
        "sha256": manifest,
        "protocol": [
            "Replay all 58 fresh v37 initial-planner payloads; never regenerate or relabel their raw output.",
            "Valid canonical normalization returns unchanged, with no recovery diagnostic.",
            "Recover only initial-planner invalid-intent errors whose literal is a known query_type enum and not a valid intent.",
            "Use the exact existing question-based fallback intent, not an invented intent map or answer label.",
            "Change only that copied raw field, then rerun all canonical payload validation; reject extra/missing/bad remaining fields normally.",
            "Store original error, raw enum and replacement provenance in a separate diagnostic sidecar.",
            "Other model fields survive canonical normalization; downstream query_type/answer-type/entities/keywords may differ from FULL fallback.",
            "This is field-scoped recovery, NOT a claim that only final entities change.",
            "A second, more conservative variant retains full-fallback intent/query_type/answer-type/question/context and every previous cue.",
            "Conservative variant merges recovered cues first, but preserves the original fallback unchanged if union would exceed existing limits; no cue truncation.",
            "No follow-up recovery, new model generation, retrieval, answer, source edit, enum relaxation, training or promotion.",
            "Inspect preserved names literally present in questions; this is not source-backed answer truth or a retrieval gain measurement.",
        ],
    })
    results, summary = [], Counter()
    for source, item in zip(pre37["rows"], gen37["rows"], strict=True):
        assert source["id"] == item["id"] and item["status"] == "ok"
        saved = item["processing"]
        payload = deepcopy(saved["parsed_model_payload"])
        original_payload = deepcopy(payload)
        try:
            candidate, diagnostic = normalize_with_intent_fallback(
                payload, normalize=normalize_hypothesis_payload, error_type=ModelOutputError,
                detect_intent=detect_intent, hypothesis_intents=HYPOTHESIS_INTENTS,
                query_types=QUERY_TYPES, question=source["question"], dialogue_context="",
            )
            record = {"status": "normalized", "diagnostic": diagnostic}
        except ModelOutputError as exc:
            candidate = build_hypothesis(source["question"], "")
            diagnostic = None
            record = {"status": "full_fallback_preserved", "error": str(exc), "diagnostic": None}
        assert payload == original_payload
        control = saved["control"]
        candidate_data = asdict(candidate)
        changed = [key for key in control if control[key] != candidate_data[key]]
        conservative, conservative_diagnostic = candidate, None
        if diagnostic is not None:
            conservative, conservative_diagnostic = preserve_fallback_routing(
                HypothesisDocument(**control), candidate,
            )
        conservative_data = asdict(conservative)
        for key in control.keys() - {"entities", "keywords"}:
            assert conservative_data[key] == control[key]
        assert set(control["entities"]) <= set(conservative.entities)
        assert set(control["keywords"]) <= set(conservative.keywords)
        if saved["normalization_status"] == "valid_model_payload":
            assert not changed and diagnostic is None
            summary["valid_payloads_identical"] += 1
        elif diagnostic is not None:
            assert diagnostic["replacement_intent"] == control["intent"]
            assert diagnostic["original_error"] == saved["normalization_error"]
            summary["invalid_intents_recovered"] += 1
        summary[record["status"]] += 1
        model_entities = payload.get("entities", [])
        preserved = [
            term for term in model_entities
            if isinstance(term, str) and term in source["question"]
            and term in candidate.entities and term not in control["entities"]
        ]
        record.update(
            id=source["id"], question=source["question"],
            original_payload=payload, original_status=saved["normalization_status"],
            original_error=saved.get("normalization_error"),
            control=control, candidate=candidate_data, changed_fields=changed,
            routing_frozen_candidate=conservative_data,
            routing_frozen_diagnostic=conservative_diagnostic,
            routing_frozen_query=build_retrieval_query(conservative),
            restored_model_entity_spellings_in_question=preserved,
            control_query=saved["control_query"], candidate_query=build_retrieval_query(candidate),
        )
        results.append(record)
    assert len(results) == 58
    assert all(shared.sha(Path(p)) == h for p, h in manifest.items())
    shared.save(HERE / "replay_v1.json", {
        "version": "initial_planner_intent_field_fallback_replay_v39",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "production_changed": False, "training_ready": False,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "summary_not_accuracy": summary, "rows": results,
        "source_and_input_hashes_unchanged": True,
        "promotion_decision": "experimental_only_not_promoted",
        "limitations": [
            "Replayed the same 58 development planner outputs, not independent evaluation.",
            "No initial-planner output was generated anew; no final answer or evidence quality measured.",
            "Remaining fixed-window heuristics are still appended after the partial recovery.",
            "Model-supplied terms and metadata are unverified retrieval cues, not factual claims.",
            "Production integration would require persistent recovery diagnostics and downstream/full-pipeline validation.",
        ],
    })
    print(dict(summary))
    for row in results:
        if row["diagnostic"] is not None:
            print({"id": row["id"], "changed_fields": row["changed_fields"],
                   "restored_question_names": row["restored_model_entity_spellings_in_question"],
                   "routing_frozen_diagnostic": row["routing_frozen_diagnostic"]})


if __name__ == "__main__":
    main()
