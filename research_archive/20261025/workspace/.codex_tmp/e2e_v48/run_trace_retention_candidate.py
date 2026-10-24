"""CPU-only trace-sidecar candidate with strict behavioral projection checks."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
V45 = HERE.parent / "staged_round2_reader_20260910.v45"
V47 = HERE.parent / "fallback_trace_completeness_20260910.v47"
sys.path[:0] = [str(V45), str(V47)]
import run_second_round_reader as probe
import run_fallback_trace_audit as audit
shared = probe.shared
FIELD = "final_direct_fallback_attempts"


def helpers_and_candidate():
    helpers = probe.staged.imports()
    import answer_conclusion_generation_original as original
    import answer_conclusion_generation_candidate as candidate
    assert candidate.sample_grounded_conclusions is original.sample_grounded_conclusions
    assert candidate.should_try_final_direct_answer is original.should_try_final_direct_answer
    return helpers, candidate


def verify(pre):
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    probe.reader.verify(shared.load(V45 / "preflight.json"))


def prepare():
    assert not (HERE / "preflight.json").exists()
    old = shared.load(V45 / "preflight.json")
    probe.reader.verify(old)
    closed = shared.load(V47 / "checkpoint_v47.json")
    assert closed["actual_state_replays"] == 14
    source = probe.reader.SRC / "asa_arknight_story_agent/inference/generation/answer_conclusion_generation.py"
    assert shared.sha(source) == shared.sha(HERE / "answer_conclusion_generation_original.py")
    helpers_and_candidate()
    paths = [
        Path(__file__), HERE / "answer_conclusion_generation_original.py",
        HERE / "answer_conclusion_generation_candidate.py", HERE / "test_fallback_trace_candidate.py",
        V45 / "preflight.json", V45 / "generations.json", V45 / "checkpoint_v45.json",
        V45 / "run_second_round_reader.py", V45 / "primary_fallback_audit_v1.json",
        V47 / "run_fallback_trace_audit.py", V47 / "preflight.json", V47 / "audit_v1.json",
        V47 / "checkpoint_v47.json", probe.V44 / "retrieval.json",
        *sorted(probe.reader.SRC.rglob("*.py")),
    ]
    shared.save(HERE / "preflight.json", {
        "version": "isolated_final_direct_trace_retention_candidate_v48",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {str(p): shared.sha(p) for p in paths},
        "canonical_source_sha256": shared.sha(source), "new_trace_field": FIELD,
        "protocol": [
            "Isolated candidate function imports all unchanged sampling/parsing/grounding/prompt/eligibility helpers from an exact frozen source copy.",
            "Only add an append-only final-step fallback-attempt sidecar after the original direct fallback call succeeds or fails.",
            "Retain primary conclusions/failure snapshots and fallback success/error; do not overwrite previous sidecar entries.",
            "Twenty synthetic unit cases cover empty trace, fallback success/failure, rejected/empty/accepted primary, no eligible fallback, deep-copy and append.",
            "Replay all14 actual v45 second-round states against baseline and candidate, using all saved raw requests with exact prompt/kwargs assertions.",
            "Projection removes ONLY the new sidecar. Conclusion, action, final answer, every legacy trace field, state and evidence must match.",
            "Ten already-terminal v41 endpoints are only checked for unchanged carry and never regenerated.",
            "Replay v47's two explicit synthetic exception fixtures against both implementations; stdout and legacy results must remain unchanged.",
            "No production edit, new model/retrieval/GPU call, source labels, human-gold claim, semantic validator change or accuracy claim.",
        ],
        "production_changed": False, "training_ready": False, "gpu_used": False,
    })
    print({"baseline_source_sha256": shared.sha(source), "planned_unit_tests": 20,
           "planned_actual_state_replays": 14, "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def candidate_invoke(pre, helpers, candidate, item, source, generator):
    pipeline = probe.reader.pipeline_for(pre, helpers, generator)
    state = helpers["PipelineRunState"](**deepcopy(source["state_after_retrieval"]))
    hypothesis = helpers["HypothesisDocument"](**item["hypothesis"])
    conclusion = candidate.generate_conclusion_from_model(
        pipeline=pipeline, question=item["question"], current_hypothesis=hypothesis,
        evidence=state.evidence, retrieval_trace=state.retrieval_trace, current_round=2,
    )
    helpers["append_conclusion_to_step"](state.retrieval_trace[-1], conclusion)
    assert helpers["apply_terminal_conclusion"](
        state=state, conclusion=conclusion, round_index=2, max_retrieval_rounds=2,
    )
    assert state.evidence == source["state_after_retrieval"]["evidence"]
    complete = asdict(state)
    sidecars = deepcopy(complete["retrieval_trace"][-1].get(FIELD, []))
    projected = deepcopy(complete)
    assert all(FIELD not in step for step in projected["retrieval_trace"][:-1])
    projected["retrieval_trace"][-1].pop(FIELD, None)
    return {
        "conclusion": asdict(conclusion), "retrieval_trace": projected["retrieval_trace"],
        "final_answer": projected["final_answer"],
        "state_after_reader_sha256": probe.staged.object_digest(projected),
    }, sidecars, complete


def run():
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    tree = ET.parse(HERE / "unit_tests_v1.xml")
    suites = list(tree.getroot().iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 20
    assert all(int(s.attrib.get(k, "0")) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    old = shared.load(V45 / "preflight.json")
    saved = shared.load(V45 / "generations.json")
    primary = {(r["pair_id"], r["condition"]): r for r in shared.load(V45 / "primary_fallback_audit_v1.json")["rows"]}
    helpers, candidate = helpers_and_candidate()
    frozen, rows, carry = probe.sources(), [], 0
    for item, result in zip(old["records"], saved["records"], strict=True):
        key = item["pair_id"], item["condition"]
        if item["terminal_after_round1"]:
            assert result["final_answer"] == item["carried_terminal_answer"]
            assert result["conclusion"] == item["carried_conclusion"] and not result["requests"]
            carry += 1
            continue
        baseline_gen = probe.ReplayGenerator(old["generator"], item, result)
        baseline = probe.invoke(old, helpers, item, frozen[key], baseline_gen)
        assert all(result[k] == v for k, v in baseline.items())
        candidate_gen = probe.ReplayGenerator(old["generator"], item, result)
        projected, sidecars, complete = candidate_invoke(
            old, helpers, candidate, item, frozen[key], candidate_gen,
        )
        assert projected == baseline
        assert baseline_gen.calls == candidate_gen.calls == len(result["requests"])
        if len(result["requests"]) == 2:
            assert len(sidecars) == 1
            record = sidecars[0]
            assert record["primary_conclusions"] == [primary[key]["primary_conclusion"]]
            assert record["fallback"] == {"status": "ok", "conclusion": result["conclusion"]}
            assert record["primary_generation_failures"] == []
        else:
            assert sidecars == []
        rows.append({
            "pair_id": key[0], "condition": key[1], "calls_replayed": candidate_gen.calls,
            "legacy_behavior_identical": True, "sidecars": sidecars,
            "projected_state_sha256": projected["state_after_reader_sha256"],
            "candidate_full_state_sha256": probe.staged.object_digest(complete),
        })
    assert len(rows) == 14 and carry == 10
    assert sum(bool(r["sidecars"]) for r in rows) == 9
    item = next(r for r in old["records"] if (r["pair_id"], r["condition"]) == ("fixed58-010:seed0", "control"))
    result = next(r for r in saved["records"] if (r["pair_id"], r["condition"]) == ("fixed58-010:seed0", "control"))
    source = frozen[("fixed58-010:seed0", "control")]
    synthetic = []
    for fail_primary in (False, True):
        baseline_gen = audit.FailingFallback(old["generator"], item, result, fail_primary)
        candidate_gen = audit.FailingFallback(old["generator"], item, result, fail_primary)
        baseline_log, candidate_log = io.StringIO(), io.StringIO()
        with redirect_stdout(baseline_log):
            baseline = probe.invoke(old, helpers, item, source, baseline_gen)
        with redirect_stdout(candidate_log):
            projected, sidecars, complete = candidate_invoke(old, helpers, candidate, item, source, candidate_gen)
        assert projected == baseline and baseline_log.getvalue() == candidate_log.getvalue()
        assert baseline_gen.requests == candidate_gen.requests and candidate_gen.calls == 2
        assert len(sidecars) == 1
        assert sidecars[0]["fallback"] == {
            "status": "error", "error": "ValueError: synthetic_final_direct_fallback_error_v47",
        }
        assert bool(sidecars[0]["primary_generation_failures"]) == fail_primary
        synthetic.append({
            "fixture": "both_calls_raise" if fail_primary else "rejected_primary_fallback_raises",
            "synthetic_only": True, "legacy_behavior_and_stdout_identical": True,
            "fallback_error_retained": True, "sidecars": sidecars,
        })
    verify(pre)
    shared.save(HERE / "replay_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "unit_tests": 20, "unit_tests_sha256": shared.sha(HERE / "unit_tests_v1.xml"),
        "actual_state_pairs": 14, "terminal_carry_unchanged": 10,
        "all_legacy_behavior_identical_after_sidecar_projection": True,
        "fallback_primary_records_retained": 9, "rows": rows,
        "synthetic_failure_protocol_cases": synthetic,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "production_changed": False, "training_ready": False,
    })
    print({"unit_tests": 20, "actual_state_pairs": 14, "terminal_carry": 10,
           "fallback_primary_records_retained": 9, "legacy_behavior_changed": 0,
           "synthetic_failure_pairs": 2, "production_changed": False}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_trace_retention_candidate.py prepare|run")
    operations[sys.argv[1]]()
