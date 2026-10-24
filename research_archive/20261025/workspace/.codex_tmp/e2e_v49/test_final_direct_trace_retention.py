from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from asa_arknight_story_agent.inference.generation import answer_conclusion_generation as candidate
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def conclusion(action, raw="primary raw"):
    return ConclusionResult(
        next_action=action, answer="answer" if action == "answer_directly" else "not enough",
        missing_slots=[], clarification_question="", follow_up_hypothesis=None,
        grounding_warnings=["retained_primary_warning"],
        generation_diagnostics={"raw_output": raw, "nested": {"value": ["original"]}},
    )


def fixture(monkeypatch, primary, fallback_error, allowed=True):
    monkeypatch.setattr(candidate, "build_conclusion_prompt", lambda *args, **kwargs: ("prompt", "visible"))
    monkeypatch.setattr(candidate, "sample_grounded_conclusions", lambda **kwargs: primary)
    monkeypatch.setattr(candidate, "select_self_consistent_conclusion", lambda values: values[0])
    monkeypatch.setattr(candidate, "should_try_final_direct_answer", lambda **kw: allowed and kw["current_round"] >= 2)
    requests = []
    direct = conclusion("answer_directly", raw="fallback raw")

    def generate(*args):
        requests.append(args)
        if fallback_error:
            raise ValueError("synthetic fallback error")
        return direct

    pipeline = SimpleNamespace(
        prepare_prompt_evidence=lambda q, h, e: deepcopy(e),
        max_retrieval_rounds=2, prompt_evidence_top_k=10,
        prompt_evidence_max_chars_per_doc=1800, prompt_conclusion_evidence_max_total_chars=12000,
        conclusion_prompt_mode="minimal", answer_grounding_mode="quote", generate_direct_answer=generate,
    )
    hypothesis = HypothesisDocument(
        question="测试问题", intent="plot_fact", query_type="fact",
        entities=[], keywords=[], expected_answer_type="事实",
    )
    return pipeline, hypothesis, direct, requests


@pytest.mark.parametrize("action", ["abstain", "retrieve_more", "none"])
@pytest.mark.parametrize("fallback_error", [False, True])
@pytest.mark.parametrize("has_trace", [False, True])
def test_success_and_failure_preserve_primary_without_changing_result(
    monkeypatch, action, fallback_error, has_trace,
):
    primary = [] if action == "none" else [conclusion(action)]
    pipeline, hypothesis, direct, requests = fixture(monkeypatch, primary, fallback_error)
    trace = [{"round": 2, "generation_failures": [{"error": "synthetic earlier failure"}]}] if has_trace else []
    result = candidate.generate_conclusion_from_model(
        pipeline=pipeline, question=hypothesis.question, current_hypothesis=hypothesis,
        evidence=[], retrieval_trace=trace, current_round=2,
    )
    assert len(requests) == 1
    if not fallback_error:
        assert result is direct
    elif primary:
        assert result is primary[0]
    else:
        assert asdict(result) == asdict(candidate.default_failed_conclusion(max_round_reached=True))
    if has_trace:
        record = trace[-1]["final_direct_fallback_attempts"][0]
        assert record["primary_conclusions"] == [asdict(c) for c in primary]
        assert record["primary_generation_failures"] == [{"error": "synthetic earlier failure"}]
        if fallback_error:
            assert record["fallback"] == {"status": "error", "error": "ValueError: synthetic fallback error"}
        else:
            assert record["fallback"] == {"status": "ok", "conclusion": asdict(direct)}
        trace[-1]["generation_failures"][0]["error"] = "changed later"
        assert record["primary_generation_failures"][0]["error"] == "synthetic earlier failure"
        if primary:
            primary[0].generation_diagnostics["nested"]["value"].append("changed later")
            assert record["primary_conclusions"][0]["generation_diagnostics"]["nested"]["value"] == ["original"]
    else:
        assert trace == []


@pytest.mark.parametrize("action", ["answer_directly", "abstain", "none"])
@pytest.mark.parametrize("reason", ["round1", "disabled"])
def test_ineligible_fallback_does_not_add_trace_or_calls(monkeypatch, action, reason):
    primary = [] if action == "none" else [conclusion(action)]
    pipeline, hypothesis, direct, requests = fixture(
        monkeypatch, primary, False, allowed=reason != "disabled",
    )
    round_index = 1 if reason == "round1" else 2
    trace = [{"round": round_index}]
    result = candidate.generate_conclusion_from_model(
        pipeline=pipeline, question=hypothesis.question, current_hypothesis=hypothesis,
        evidence=[], retrieval_trace=trace, current_round=round_index,
    )
    assert requests == [] and "final_direct_fallback_attempts" not in trace[0]
    if primary:
        assert result is primary[0]


def test_accepted_primary_is_not_replaced_or_recorded(monkeypatch):
    primary = [conclusion("answer_directly")]
    pipeline, hypothesis, direct, requests = fixture(monkeypatch, primary, False)
    trace = [{"round": 2}]
    result = candidate.generate_conclusion_from_model(
        pipeline=pipeline, question=hypothesis.question, current_hypothesis=hypothesis,
        evidence=[], retrieval_trace=trace, current_round=2,
    )
    assert result is primary[0] and requests == []
    assert "final_direct_fallback_attempts" not in trace[0]


def test_append_preserves_existing_records_and_all_primary_samples():
    primary = [conclusion("abstain", "raw1"), conclusion("retrieve_more", "raw2")]
    trace = [{"final_direct_fallback_attempts": [{"existing": True}]}]
    candidate.record_final_direct_attempt(trace, primary, conclusion=conclusion("answer_directly", "raw3"))
    assert trace[0]["final_direct_fallback_attempts"][0] == {"existing": True}
    assert len(trace[0]["final_direct_fallback_attempts"]) == 2
    assert trace[0]["final_direct_fallback_attempts"][1]["primary_conclusions"] == [asdict(c) for c in primary]
