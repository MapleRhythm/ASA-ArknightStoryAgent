from datetime import datetime, timezone
from pathlib import Path
import re
import run_canonical_reader_v2 as probe

HERE = Path(__file__).resolve().parent
shared = probe.shared


def blocks(text):
    pieces = re.split(r"(?m)^\[(E\d+)\]\s*", text)
    return dict(zip(pieces[1::2], pieces[2::2], strict=True))


def main():
    assert not (HERE / "checkpoint_v41.json").exists()
    pre, gen, post, review, protocol, witnesses = [
        shared.load(HERE / name) for name in (
            "preflight.json", "generations.json", "postflight_v1.json", "source_review_v41.json",
            "protocol_check_v1.json", "evidence_trace_v1.json",
        )
    ]
    probe.verify(pre)
    assert post["preflight_sha256"] == gen["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert post["generations_sha256"] == shared.sha(HERE / "generations.json")
    assert post["attempts"] == post["actual_calls"] == post["canonical_method_replays"] == 24
    assert len(protocol["checks"]) == 12 and protocol["model_calls"] == 0
    assert protocol["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in witnesses["basis_sha256"].items())
    inputs = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    outputs = {(r["pair_id"], r["condition"]): r for r in gen["attempts"]}
    bound_sources, bound_outputs, reviewed = [], [], set()
    for case in review["cases"]:
        for binding in case["sources"]:
            for kind in binding["conditions"]:
                item = inputs[(case["id"], kind)]
                text = blocks(item["visible_evidence"])[binding["eid"]]
                assert binding["quote"] in text, binding
                for other_eid in binding.get("absent_from_eids", []):
                    assert binding["quote"] not in blocks(item["visible_evidence"])[other_eid]
                bound_sources.append({
                    "id": case["id"], "condition": kind, **binding,
                    "prompt_sha256": item["prompt_sha256"], "block_sha256": shared.digest(text),
                })
            for kind in binding.get("absent_from_visible", []):
                assert binding["quote"] not in inputs[(case["id"], kind)]["visible_evidence"]
        for binding in case["outputs"]:
            for repeat in binding["repeats"]:
                for kind in binding["conditions"]:
                    key = (f"{case['id']}:seed{repeat}", kind)
                    record = outputs[key]
                    conclusion = record["conclusion"]
                    assert record["status"] == "ok" and record["calls"][0]["finish_reason"] == "stop"
                    assert conclusion["next_action"] == binding["action"]
                    if "answer_contains" in binding:
                        assert binding["answer_contains"] in conclusion["answer"]
                    if "answer_exact" in binding:
                        assert binding["answer_exact"] == conclusion["answer"]
                    if "raw_contains" in binding:
                        assert binding["raw_contains"] in record["sanitized_raw_output"]
                    if "warnings_exact" in binding:
                        assert conclusion["grounding_warnings"] == binding["warnings_exact"]
                    if "warning_contains" in binding:
                        assert any(binding["warning_contains"] in w for w in conclusion["grounding_warnings"])
                    assert key not in reviewed
                    reviewed.add(key)
                    bound_outputs.append({
                        "pair_id": key[0], "condition": kind, **binding,
                        "raw_sha256": shared.digest(record["sanitized_raw_output"]),
                        "answer_sha256": shared.digest(conclusion["answer"]),
                    })
    assert reviewed == set(outputs) and len(reviewed) == 24
    noise = [r for r in post["paired_comparisons_not_accuracy"] if r["same_input_prompt"]]
    assert len(noise) == 4 and sum(r["same_raw"] for r in noise) == 2
    assert all(r["same_action"] and r["same_answer"] for r in noise)
    log = (HERE / "run_v1.log").read_text()
    assert "Shutdown complete" in log and "destroy_process_group() was not called" in log
    assert not re.search(r"\bERROR\b", log)
    first_failure = shared.load(HERE / "preparation_attempt01_failure.json")
    assert first_failure["actual_generation_calls"] == 0
    files = [HERE / name for name in (
        "preflight.json", "generations.json", "postflight_v1.json", "source_review_v41.json",
        "run_v1.log", "idle_samples.json", "protocol_check_v1.json", "check_protocol_v1.py",
        "evidence_trace_v1.json", "audit_evidence_trace_v1.py", "close_canonical_reader.py",
        "preparation_attempt01_failure.json", "run_canonical_reader.py", "run_canonical_reader_v2.py",
    )]
    shared.save(HERE / "checkpoint_v41.json", {
        "version": "canonical_first_round_reader_closed_v41",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False,
        "summary_not_accuracy": post["summary_not_accuracy"],
        "canonical_method_replays": 24, "cpu_path_checks": 12,
        "source_bindings": bound_sources, "output_bindings": bound_outputs,
        "no_intervention_pairs": noise, "reviewer": review["reviewer"],
        "assistant_output_source_review_complete": True,
        "promotion_decision": review["promotion_decision"],
        "basis_sha256": {str(p): shared.sha(p) for p in files},
        "limitations": review["limitations"],
    })
    print({"reviewed_outputs": 24, "source_bindings": len(bound_sources),
           "promotion": review["promotion_decision"]})


if __name__ == "__main__":
    main()
