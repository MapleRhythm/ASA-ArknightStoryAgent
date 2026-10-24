"""Post-hoc assistant source review and completion receipt; never model input."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

import run_shadow_v2 as wrapper

base, protocol, shared, HERE = wrapper.base, wrapper.protocol, wrapper.shared, wrapper.HERE


def note(item):
    family, kind, tid = item["family"], item["kind"], item["task_id"]
    if kind == "verify_frozen":
        if family == "parent_direction":
            if "projection0" in tid:
                return (
                    "The relation is source-consistent in the full packet, but the rationale misstates "
                    "who is being thanked: Nastie thanks Fernan, not Fulalia. T22/T71 alone do not name "
                    "Fulalia. This is not certified relation-entailment or a complete answer to who is sought."
                )
            return (
                "Mother/daughter direction is reversed. T22/T71 say Nastie thanks Fernan for caring for "
                "HER mother. Full context T112-T117 identifies Fulalia as mother. Raw supported is erroneous; "
                "historical output additionally has schema extras, not semantic rejection."
            )
        if family == "first_person_owner":
            if "projection0" in tid:
                return (
                    "Juanna's father's hunting/treatment statement is source-consistent. T11-T14 identify "
                    "Juanna as speaker; it does not answer the Pramanix question. The model's answers "
                    "relevance label is wrong, and T12 alone omits the speaker label."
                )
            return (
                "The owner is substituted: T11-T14 (also T67-T70) are Juanna's speech, with Enya as "
                "interlocutor. Model still calls it Pramanix's father's story and returns supported/answers. "
                "Historic compound also misattributes the station/disaster episode."
            )
        if "projection0" in tid:
            return (
                "The narrow self-description is supported by T47-T48: not the original Friston, a "
                "manufactured tool. Source speaker label is in T47, preceding selected T48."
            )
        return (
            "T48 manufactured tool and T56 copy identity do not establish Friston as manufacturer. "
            "Raw supported is an unsupported creator inference, not proof the opposite creator is true. "
            "Actual schema rejection is caused by extra fields, not detection of this semantic defect."
        )
    if family == "parent_direction":
        return (
            "Both raw drafts have the sought-person Nastie core plus a reversed mother relation. "
            "Both lack unresolved and choose 3 references for their first fact (>2). They are rejected "
            "before canonical citation construction or any child verifier; no displayed gain."
        )
    if family == "first_person_owner":
        return (
            "Raw draft combines a train-accident core matching the available family dialogue with Juanna's "
            "station/disaster story misattributed to Pramanix. T24-T28 explicitly speak of Enciodes' parents; "
            "T25/T26 selected without T24 do not independently certify the Pramanix family identity. "
            "Seed0 repeats the wrong second fact until 2048-token length stop; seed1 has trailing comma "
            "invalid JSON. Neither is repaired, displayed, or sent to a child verifier."
        )
    if kind == "draft":
        if "seed0" in tid:
            return (
                "Fresh span citations are exact, but first claim invents creator. Existing canonical "
                "quote gate rejects missing manufacture term in T6. Second fact has a supported "
                "personality-simulation/core from T5, but the whole draft is withheld; it is not salvaged."
            )
        return (
            "Fresh span citations are exact, but first claim invents creator and T47/T137 do not "
            "support manufacture. Existing quote gate rejects manufacture. Second claim omits T6's "
            "'perhaps to you' self-description framing; it is not certified as an unqualified fact. "
            "Whole draft withheld, no citation addition or rebind."
        )
    if "fact0" in tid:
        return (
            "Verifier falsely supports creator attribution from T48/T56. Seed0 has extra-schema fields; "
            "seed1 is schema-valid but its T48/T56 were not cited by draft T47/T137, so the fixed "
            "citation-subset gate rejects it. Neither outcome is a correct semantic verdict."
        )
    if "seed0" in tid:
        return (
            "Personality-simulation/long watch claim has a supported core at T5, with speaker context "
            "in adjacent full evidence. Child verifier passes using already cited T5. Whole draft still "
            "fails due to its separate unsupported creator claim; no final answer delivered."
        )
    return (
        "Raw verifier supports helper/mastermind wording at T6 but ignores its perspectival framing. "
        "Output has extra fields and is rejected by schema, not a semantic correction."
    )


def compute_review():
    pre = shared.load(HERE / "preflight.json")
    base.verify(pre)
    saved = shared.load(HERE / "generations.json")
    assert saved["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(x) for x in (HERE / "generations.progress.jsonl").read_text().splitlines()]
    assert journal == saved["records"]
    post = shared.load(HERE / "postflight_v1.json")
    replay = base.recompute(pre, saved)
    assert all(post[k] == v for k, v in replay.items())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    rows = {r["family"]: r for r in shared.load(HERE / "inputs.json")}
    reviewed, bindings = [], []
    for index, result in enumerate(saved["records"]):
        item, call = result["task"], result["calls"][0]
        assert len(result["calls"]) == 1
        token_ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        assert shared.digest(item["prompt"]) == call["prompt_sha256"] == item["prompt_sha256"]
        assert len(token_ids) == call["prompt_tokens"]
        assert shared.digest(json.dumps(token_ids)) == call["token_ids_sha256"]
        assert len(token_ids) + 2048 <= pre["runtime"]["max_model_len"]
        assert call["lora_id"] == call["returned_lora_id"] == 1
        assert call["seed"] == item["seed"] and call["params"] == base.PARAMS
        raw = result["sanitized_raw_output"]
        packet = protocol.from_dict(rows[item["family"]]["packet"])
        if item["kind"] == "draft":
            expected_prompt = protocol.draft_prompt(packet, rows[item["family"]]["question"])
        else:
            expected_prompt = protocol.verify_prompt(
                packet, rows[item["family"]]["question"], item["claim"], item["claim_id"])
        assert expected_prompt == item["prompt"]
        try:
            diagnostic_payload, mode = protocol.parse_prefilled(raw)
            syntax_error = None
        except ValueError as exc:
            diagnostic_payload, mode = None, None
            syntax_error = f"{type(exc).__name__}: {exc}"
        # Diagnostic inspection of rejected output NEVER changes actual schema
        # status, canonical result, citations, subset gate or final display.
        selected_ids = list(dict.fromkeys(re.findall(r'"(T\d+)"', raw)))
        available = {s.span_id: s for s in packet.spans}
        missing = [s for s in selected_ids if s not in available]
        assert not missing
        refs = []
        for sid in selected_ids:
            span = available[sid]
            block = dict(packet.blocks)[span.evidence_id]
            assert shared.digest(block) == span.block_sha256
            assert block[span.start:span.end] == span.text
            start, end = max(0, span.start - 200), min(len(block), span.end + 200)
            binding = {
                "task_id": item["task_id"], "packet_id": packet.packet_id, "span_id": sid,
                "evidence_id": span.evidence_id, "block_sha256": span.block_sha256,
                "start": span.start, "end": span.end, "text": span.text,
                "context_start": start, "context_end": end, "context": block[start:end],
            }
            refs.append(binding)
            bindings.append(binding)
        reviewed.append({
            "index": index, "task_id": item["task_id"], "kind": item["kind"],
            "claim": item["claim"], "status": result["status"], "error": result.get("error"),
            "raw_sha256": shared.digest(raw), "record_sha256": shared.digest(json.dumps(result, ensure_ascii=False, sort_keys=True)),
            "raw_payload_diagnostic_only_not_validation": diagnostic_payload,
            "raw_json_parse_mode": mode, "raw_json_error": syntax_error,
            "finish": call["finish_reason"], "generated_tokens": call["generated_tokens"],
            "seconds_including_validation_not_engine_startup": result["seconds"],
            "source_bindings": refs, "assistant_observation_not_human_gold": note(item),
        })
    assert len(reviewed) == 25
    frozen = [r for r in saved["records"] if r["task"]["kind"] == "verify_frozen"]
    def is_bad_reference(result):
        label = result["task"]["assistant_reference_label"]
        return label in {
            "direction_reversed", "owner_substituted", "creator_added_without_support",
            "known_erroneous_or_unsupported_composite",
        }
    bad = [r for r in frozen if is_bad_reference(r)]
    good = [r for r in frozen if not is_bad_reference(r)]
    raw_frozen = [r for r in reviewed if r["kind"] == "verify_frozen"]
    assert len(bad) == 9 and len(good) == 6
    assert all(r["raw_payload_diagnostic_only_not_validation"]["verdict"] == "supported" for r in raw_frozen)
    assert all(r["raw_payload_diagnostic_only_not_validation"]["relevance"] == "answers" for r in raw_frozen)
    draft_results = [r for r in saved["records"] if r["task"]["kind"] == "draft"]
    seconds = [r["seconds"] for r in saved["records"]]
    total_draft_seconds = [
        sum(r["seconds"] for r in saved["records"]
            if r["task"]["task_id"] == d["task"]["task_id"]
            or r["task"].get("parent_task_id") == d["task"]["task_id"])
        for d in draft_results
    ]
    for name in ("progress_snapshot01.jsonl", "progress_snapshot02.jsonl"):
        snapshot = [json.loads(s) for s in (HERE / name).read_text().splitlines()]
        assert snapshot == saved["records"][:len(snapshot)]
    return {
        "assistant_review_not_human_gold": True, "reviewed_rows": reviewed,
        "all_prompt_token_ids_recomputed": 25, "all_fresh_source_bindings": bindings,
        "source_binding_count": len(bindings),
        "metrics_development_only_not_accuracy": {
            "actual_calls": 25, "frozen_checker_calls": 15, "fresh_drafts": 6, "fresh_fact_checkers": 4,
            "protocol_success": 15, "protocol_errors": 10,
            "finish": dict(Counter(r["finish"] for r in reviewed)),
            "raw_frozen_supported_verdicts_including_schema_errors": 15,
            "bad_frozen_reference_items": 9,
            "bad_frozen_raw_supported": 9,
            "bad_frozen_shadow_pass": sum(r.get("analysis", {}).get("shadow_semantic_pass", False) for r in bad),
            "bad_frozen_schema_rejected_not_semantic_correction": sum(r["status"] != "ok" for r in bad),
            "source_consistent_frozen_reference_items": 6,
            "source_consistent_frozen_supported": sum(r["status"] == "ok" for r in good),
            "irrelevant_juanna_claim_incorrectly_marked_answers": 2,
            "draft_protocol_success": sum(d["status"] == "ok" for d in draft_results),
            "draft_combined_pass": sum(d["shadow_combined_pass"] for d in replay["drafts"]),
            "call_seconds_total": sum(seconds), "call_seconds_median": statistics.median(seconds),
            "call_seconds_range": [min(seconds), max(seconds)],
            "draft_plus_child_seconds": total_draft_seconds,
            "no_matched_baseline_latency_or_accuracy_claimed": True,
        },
        "decision": "reject_this_same_model_semantic_checker_and_fresh_draft_candidate_for_production",
        "production_changed": False, "training_ready": False,
        "general_fact_verification_or_span_protocol_not_disproven": True,
    }


def review():
    assert not (HERE / "source_review_v1.json").exists()
    result = compute_review()
    shared.save(HERE / "source_review_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "script_sha256": shared.sha(Path(__file__)), **result,
    })
    print(result["metrics_development_only_not_accuracy"], flush=True)


def close():
    assert not (HERE / "checkpoint_v53.json").exists()
    saved = shared.load(HERE / "source_review_v1.json")
    computed = compute_review()
    assert all(saved[k] == v for k, v in computed.items())
    idle = shared.load(HERE / "idle_samples.json")
    assert idle["elapsed_seconds"] >= 60 and len(idle["samples"]) >= 13
    assert all(s["memory_mib"] < 256 and s["utilization"] < 10 and not s["compute_pids"] for s in idle["samples"])
    log = (HERE / "run_v1.log").read_text()
    assert "Shutdown complete" in log and "Explicit engine shutdown requested." in log
    assert "Traceback" not in log and re.search(r"\bERROR\b", log) is None
    gpu = shared.gpu_idle()
    assert base.reader.retrieval.numeric_mapping() == shared.load(HERE / "preflight.json")["verified_nvidia_mapping"]
    files = [
        Path(__file__), HERE / "preflight.json", HERE / "inputs.json", HERE / "generations.json",
        HERE / "generations.progress.jsonl", HERE / "postflight_v1.json",
        HERE / "source_review_v1.json", HERE / "idle_samples.json", HERE / "run_v1.log",
        HERE / "check_v1.log", HERE / "source_review_v1.log",
        HERE / "progress_snapshot01.jsonl", HERE / "progress_snapshot02.jsonl",
        HERE / "checkpoint_cpu_v53.json", HERE / "cpu_integration_v1.json",
        HERE / "CANONICAL_SPAN_SEMANTIC_REJECTION_20260910_v53.md",
    ]
    shared.save(HERE / "checkpoint_v53.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "v53_model_development_experiment_complete_candidate_not_promoted",
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "fresh_replay_of_review_and_checks_equal": True, "actual_model_calls": 25,
        "protocol_success_replays": 15, "recorded_failures_preserved": 10,
        "prompt_token_ids_recomputed": 25, "source_binding_count": computed["source_binding_count"],
        "assistant_source_review_not_human_gold": True,
        "metrics_development_only_not_accuracy": computed["metrics_development_only_not_accuracy"],
        "gpu2_post_shutdown_idle": gpu, "process_group_shutdown_warning_preserved": True,
        "run_error_or_traceback": False,
        "production_changed": False, "training_ready": False,
        "new_retrieval_calls": 0, "old_evidence_ids_rebound": False,
        "decision": computed["decision"],
        "historical_cpu_only_checkpoint_and_attempt01_failures_preserved": True,
    })
    print({"status": "complete_not_promoted", "actual_calls": 25,
           "source_bindings": computed["source_binding_count"], "gpu2_idle": gpu}, flush=True)


if __name__ == "__main__":
    {"review": review, "close": close}[sys.argv[1]]()
