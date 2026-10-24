"""Bind request/endpoint observations without assigning human-gold labels."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import run_second_round_reader as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def blocks(text):
    parts = re.split(r"(?m)^\[(E\d+)\][ \t]*", text)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def main():
    assert not (HERE / "checkpoint_v45.json").exists()
    pre, gen, post, audit, review = [
        shared.load(HERE / name) for name in (
            "preflight.json", "generations.json", "postflight_v1.json",
            "primary_fallback_audit_v1.json", "source_review_v45.json",
        )
    ]
    probe.reader.verify(pre)
    assert shared.sha(HERE / "preflight.json") == gen["preflight_sha256"] == post["preflight_sha256"]
    assert shared.sha(HERE / "generations.json") == post["generations_sha256"] == audit["generations_sha256"]
    assert post["round2_canonical_replays"] == audit["canonical_replays"] == 14
    assert post["terminal_carry_unchanged"] == 10 and post["endpoints"] == 24
    inputs = {(r["pair_id"], r["condition"]): r for r in pre["records"]}
    outputs = {(r["pair_id"], r["condition"]): r for r in gen["records"]}
    assert len(inputs) == len(outputs) == 24
    assert Counter(r["status"] for r in outputs.values()) == {"ok": 14, "terminal_carried_unchanged": 10}
    assert review["reviewer"] == "assistant_source_review_not_human_gold"
    reviewed, output_bindings = set(), []
    for case in review["cases"]:
        for item in case["endpoints"]:
            key = item["pair_id"], item["condition"]
            assert key not in reviewed and key in outputs
            reviewed.add(key)
            saved = outputs[key]
            assert saved["conclusion"]["next_action"] == item["expected_action"]
            for quote in item.get("answer_contains", []):
                assert quote in saved["final_answer"], (key, quote)
            for request_binding in item.get("request_contains", []):
                request = next(r for r in saved["requests"] if r["kind"] == request_binding["kind"])
                assert request_binding["text"] in request["sanitized_raw_output"]
            output_bindings.append({
                **item, "case_observation": case["observation"],
                "final_answer_sha256": shared.digest(saved["final_answer"]),
                "conclusion_sha256": probe.staged.object_digest(saved["conclusion"]),
                "requests_sha256": probe.staged.object_digest(saved["requests"]),
            })
    assert reviewed == outputs.keys()
    source_bindings = []
    for witness in review["source_bindings"]:
        for pair_id in witness["pair_ids"]:
            for condition in witness["conditions"]:
                key = pair_id, condition
                item, saved = inputs[key], outputs[key]
                assert not item["terminal_after_round1"]
                request = next(r for r in item["planned_requests"] if r["kind"] == witness["request_kind"])
                assert any(r["kind"] == witness["request_kind"] for r in saved["requests"])
                source_blocks = blocks(request["visible_evidence"])
                assert witness["quote"] in source_blocks[witness["eid"]]
                for absent_eid in witness.get("absent_from_eids", []):
                    assert witness["quote"] not in source_blocks[absent_eid]
                source_bindings.append({
                    **witness, "pair_id": pair_id, "condition": condition,
                    "visible_sha256": shared.digest(request["visible_evidence"]),
                    "block_sha256": shared.digest(source_blocks[witness["eid"]]),
                })
    carried = [r for r in pre["records"] if r["terminal_after_round1"]]
    assert len(carried) == 10
    pairs = []
    for pair_id in sorted({r["pair_id"] for r in pre["records"]}):
        a, b = [outputs[(pair_id, k)] for k in ("control", "candidate")]
        ia, ib = [inputs[(pair_id, k)] for k in ("control", "candidate")]
        both_new = not ia["terminal_after_round1"] and not ib["terminal_after_round1"]
        pairs.append({
            "pair_id": pair_id, "both_reached_round2": both_new,
            "round2_primary_prompt_identical": (
                ia["planned_requests"][0]["prompt"] == ib["planned_requests"][0]["prompt"] if both_new else None
            ),
            "same_final_action": a["conclusion"]["next_action"] == b["conclusion"]["next_action"],
            "same_final_answer": a["final_answer"] == b["final_answer"],
            "same_new_raw_sequence": (
                [r.get("sanitized_raw_output") for r in a["requests"]] ==
                [r.get("sanitized_raw_output") for r in b["requests"]] if both_new else None
            ),
        })
    log = (HERE / "run_v1.log").read_text()
    released = shared.gpu_idle()
    basis = [HERE / name for name in (
        "preflight.json", "prepare.log", "idle_samples.json", "run_v1.log",
        "generations.json", "generations.progress.jsonl", "postflight_v1.json", "check_v1.log",
        "primary_fallback_audit_v1.json", "primary_audit_v1.log",
        "source_review_v45.json", "run_second_round_reader.py",
        "audit_primary_fallback_v1.py", "close_staged_reader.py",
    )]
    shared.save(HERE / "checkpoint_v45.json", {
        "version": "staged_round2_canonical_reader_closed_v45",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoints": 24, "terminal_carry_unchanged": 10, "round2_canonical_replays": 14,
        "actual_new_llm_calls": gen["actual_new_llm_calls"],
        "summary_not_accuracy": post["summary_not_accuracy"], "finish_counts": post["finish_counts"],
        "primary_final_transitions_not_accuracy": audit["transitions_not_accuracy"],
        "paired_endpoint_comparisons_not_accuracy": pairs,
        "reviewer": review["reviewer"], "assistant_output_review_complete": True,
        "source_bindings": source_bindings, "output_bindings": output_bindings,
        "carried_source_review": {
            "checkpoint_v41_sha256": shared.sha(probe.V41 / "checkpoint_v41.json"),
            "carried_count": 10, "prior_errors_not_repaired_or_retried": True,
        },
        "gpu2_released": released,
        "gpu_processes_at_close": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader",
        ], text=True),
        "log_error_lines": [line for line in log.splitlines() if "ERROR" in line or "Traceback" in line],
        "log_shutdown_lines": [line for line in log.splitlines() if "shutdown" in line.lower() or "destroy_process_group" in line],
        "basis_sha256": {str(p): shared.sha(p) for p in basis},
        "production_changed": False, "training_ready": False,
        "promotion_decision": review["promotion_decision"],
        "limitations": review["limitations"],
    })
    print({
        "endpoints": 24, "new_calls": gen["actual_new_llm_calls"], "source_bindings": len(source_bindings),
        "output_bindings": len(output_bindings), "gpu2_released": True,
        "promotion": review["promotion_decision"],
    }, flush=True)


if __name__ == "__main__":
    main()
