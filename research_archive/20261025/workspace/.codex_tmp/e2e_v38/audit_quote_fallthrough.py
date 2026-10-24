"""CPU-only audit of quote -> legacy-token fallthrough; no validator edit."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V35 = HERE.parent / "reader_alias_groups_20260910.v35"
RESULTS = Path("${REMOTE_ASA_ROOT}/rank_v8_coverage/runs/e2e_no_minirag_fixed58_20260909.v23v3/results.jsonl")
sys.path.insert(0, str(V27))
import run_reader_router as shared
from selector_protocol import blocks
sys.path.insert(0, str(SRC))


def main():
    import json
    from asa_arknight_story_agent.inference.generation.conclusion_generation import normalize_minimal_conclusion_output, parse_conclusion_output
    from asa_arknight_story_agent.inference.grounding import validation
    from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
    from asa_arknight_story_agent.inference.grounding.identity import primary_entity_anchor_required
    from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
    from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument

    assert not (HERE / "preflight.json").exists()
    assert not (HERE / "audit_v1.json").exists()
    files = [*sorted(SRC.rglob("*.py")), Path(__file__), RESULTS,
             V35 / "preflight.json", V35 / "generations.json",
             ROOT / "indexes/arknights_story/operator_aliases.json"]
    manifest = {str(p): shared.sha(p) for p in files}
    shared.save(HERE / "preflight.json", {
        "version": "quote_legacy_fallthrough_cpu_audit_v38",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "sha256": manifest,
        "protocol": [
            "32 completed v35 reader outputs, 8 questions, replayed without new generation.",
            "Same raw output, parsed conclusion, hypothesis, quote policy and visible evidence for every evidence-object condition.",
            "Compare empty objects (old harness), original retrieved objects, and objects containing only visible E blocks.",
            "Instrument legacy token-flow calls in this isolated CPU process, restore module function immediately after each call.",
            "Use CURRENT canonical source, including v33 layout; do not attribute old/new display punctuation differences to this audit.",
            "Separate synthetic contract fixtures are NOT story evidence, model outputs, accuracy labels or retrieval measurements.",
            "Synthetic fixtures test valid quote, augmented vs projected entities, and a warning-triggered answer reconstruction branch.",
            "No policy bypass patch, promotion, retraining, evidence mutation in old artifacts, or claim that more accepts are correct.",
        ],
    })
    token_validator = validation.validate_answer_token_grounding

    def run(question, hypothesis, conclusion, visible, evidence, mode="quote"):
        calls = []

        def record(**kwargs):
            result = token_validator(**kwargs)
            calls.append({
                "grounding_mode": kwargs["grounding_mode"],
                "evidence_objects": len(kwargs["evidence"]),
                "result_action": result.next_action if result is not None else None,
                "result_missing_slots": result.missing_slots if result is not None else [],
            })
            return result

        validation.validate_answer_token_grounding = record
        try:
            result = validation.validate_conclusion_grounding(
                question=question, hypothesis=deepcopy(hypothesis), evidence=deepcopy(evidence),
                conclusion=deepcopy(conclusion), max_round_reached=True,
                mode=mode, evidence_prompt_text=visible,
            )
        finally:
            validation.validate_answer_token_grounding = token_validator
        return {"result": asdict(result), "legacy_token_calls": calls}

    pre35 = shared.load(V35 / "preflight.json")
    gen35 = shared.load(V35 / "generations.json")
    original_rows = [json.loads(line) for line in RESULTS.read_text().splitlines() if line.strip()]
    original = {row["question"]: row for row in original_rows}
    replay, summary = [], Counter()
    for source, generated in zip(pre35["rows"], gen35["rows"], strict=True):
        assert source["pair_id"] == generated["pair_id"]
        visible = source["evidence_brief"]
        hypothesis = HypothesisDocument(**source["hypothesis"])
        original_evidence = original[source["question"]]["evidence"]
        assert isinstance(original_evidence, list) and original_evidence
        visible_evidence = [
            {"document": {"id": f"visible-fixture-{eid}", "clean_text": text}}
            for eid, text in blocks(visible)
        ]
        for condition, item in generated["conditions"].items():
            assert item["status"] == "ok"
            normalized = normalize_minimal_conclusion_output(item["raw_output"], "minimal")
            parsed = parse_conclusion_output(
                normalized, question=source["question"], current_hypothesis=hypothesis, max_round_reached=True,
            )
            record = {"pair_id": source["pair_id"], "condition": condition,
                      "raw_output_sha256": shared.digest(item["raw_output"]), "conditions": {}}
            for kind, evidence in (("empty_objects", []), ("original_objects", original_evidence),
                                   ("visible_objects", visible_evidence)):
                result = run(source["question"], hypothesis, parsed, visible, evidence)
                record["conditions"][kind] = result
                summary[f"{kind}:legacy_token_entries"] += len(result["legacy_token_calls"])
                summary[f"{kind}:action:{result['result']['next_action']}"] += 1
            empty = record["conditions"]["empty_objects"]["result"]
            for kind in ("original_objects", "visible_objects"):
                actual = record["conditions"][kind]["result"]
                summary[f"{kind}:changed_action_vs_empty"] += actual["next_action"] != empty["next_action"]
                summary[f"{kind}:changed_display_vs_empty"] += actual["answer"] != empty["answer"]
            replay.append(record)
    assert len(replay) == 32

    # Entirely invented names/text to isolate the interface, not Arknights source.
    question = "甲乙代号是什么？"
    payload = {
        "entities": ["甲乙"], "keywords": ["代号"],
        "intent": "plot_fact", "query_type": "fact", "expected_answer_type": "人物代号",
    }
    augmented = normalize_hypothesis_payload(payload, question=question, dialogue_context="")
    projected = replace(augmented, entities=["甲乙"])
    fact = "代号是丙丁，同时负责持续监测异常信号。"
    text = "甲乙：" + fact
    visible = "[E1] " + text
    objects = [{"document": {"id": "synthetic-contract-only", "clean_text": text}}]
    synthetic = {
        "synthetic_only_not_story_or_model_output": True,
        "question": question, "payload": payload, "visible_evidence": visible,
        "augmented_hypothesis": asdict(augmented), "projected_hypothesis": asdict(projected),
        "required_anchors": {
            "augmented": primary_entity_anchor_required(question, augmented),
            "projected": primary_entity_anchor_required(question, projected),
        }, "cases": [],
    }
    for label, answer in (("exact_fact", fact), ("extra_final_answer_text", "另外补充无依据细节。" + fact)):
        conclusion = ConclusionResult(
            next_action="answer_directly", answer=answer, missing_slots=[],
            clarification_question="", follow_up_hypothesis=None,
            supported_facts=[{"fact": fact, "evidence_refs": [{"evidence_id": "E1", "quote": fact}]}],
        )
        issues, warnings = validate_grounded_quotes(
            conclusion=deepcopy(conclusion), evidence=objects, question=question, evidence_prompt_text=visible,
        )
        case = {"label": label, "conclusion": asdict(conclusion),
                "quote_issues": issues, "quote_warnings": warnings, "conditions": {}}
        for mode in ("quote", "grounded", "strict", "evidence_id"):
            for entity_kind, hypothesis in (("augmented", augmented), ("projected", projected)):
                for object_kind, evidence in (("empty", []), ("complete", objects)):
                    key = f"{mode}:{entity_kind}:{object_kind}"
                    case["conditions"][key] = run(question, hypothesis, conclusion, visible, evidence, mode)
        synthetic["cases"].append(case)
    assert validation.validate_answer_token_grounding is token_validator
    assert all(shared.sha(Path(p)) == h for p, h in manifest.items())
    report = {
        "version": "quote_legacy_fallthrough_cpu_results_v38",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "gpu_used": False, "training_ready": False,
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "actual_replay_summary_not_accuracy": summary,
        "actual_replays": replay, "synthetic_contract_probe": synthetic,
        "production_source_and_inputs_unchanged": True,
        "promotion_decision": "no_validation_change_implemented",
    }
    shared.save(HERE / "audit_v1.json", report)
    print(dict(summary))
    print({"synthetic_required_anchors": synthetic["required_anchors"]})
    for case in synthetic["cases"]:
        print({"synthetic_label": case["label"], "quote_issues": case["quote_issues"],
               "quote_warnings": case["quote_warnings"],
               "actions": {k: v["result"]["next_action"] for k, v in case["conditions"].items()}})


if __name__ == "__main__":
    main()
