from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.inference.generation import answer_conclusion_generation as generation
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument


def hypothesis():
    return HypothesisDocument(
        question="问题？", intent="plot_fact", query_type="fact",
        entities=[], keywords=[], expected_answer_type="事实",
    )


def answer():
    return ConclusionResult(
        next_action="answer_directly", answer="答案", missing_slots=[],
        clarification_question="", follow_up_hypothesis=None,
    )


def pipeline(generate, samples=1):
    return SimpleNamespace(
        generator=SimpleNamespace(generate=generate, max_tokens=512, repeat_penalty=1.0),
        self_consistency_samples=samples, self_consistency_temperature=0.1,
        answer_grounding_mode="evidence_id", conclusion_prompt_mode="minimal",
        max_retrieval_rounds=2,
    )


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("visible", ["", None])
def test_sampling_preserves_explicit_empty_evidence_in_normal_and_recovery_paths(
    monkeypatch, recovered, visible,
):
    captured = []

    def parse(*args, **kwargs):
        if recovered:
            raise ValueError("truncated")
        return answer()

    def validate(**kwargs):
        captured.append(kwargs["evidence_prompt_text"])
        return kwargs["conclusion"]

    monkeypatch.setattr(generation, "parse_conclusion_output", parse)
    monkeypatch.setattr(generation, "recover_truncated_grounded_answer", lambda *a, **kw: answer())
    monkeypatch.setattr(generation, "validate_conclusion_grounding", validate)
    result = generation.sample_grounded_conclusions(
        pipeline=pipeline(lambda *a, **kw: "{}"), question="问题？",
        current_hypothesis=hypothesis(), prompt_evidence=[],
        prompt="FULL PROMPT WITH SCHEMA", evidence_prompt_text=visible, current_round=1,
    )
    assert len(result) == 1
    assert captured == ["FULL PROMPT WITH SCHEMA" if visible is None else ""]
    assert result[0].generation_diagnostics["parse_status"] == (
        "recovered_truncated" if recovered else "strict_json"
    )


@pytest.mark.parametrize("with_trace", [False, True])
def test_failed_sample_does_not_crash_remaining_samples_and_records_trace(monkeypatch, with_trace):
    calls = []

    def generate(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("transient generation error")
        return "{}"

    monkeypatch.setattr(generation, "recover_truncated_grounded_answer", lambda *a, **kw: None)
    monkeypatch.setattr(generation, "parse_conclusion_output", lambda *a, **kw: answer())
    monkeypatch.setattr(generation, "validate_conclusion_grounding", lambda **kw: kw["conclusion"])
    trace = [{"round": 1}] if with_trace else None
    result = generation.sample_grounded_conclusions(
        pipeline=pipeline(generate, samples=2), question="问题？",
        current_hypothesis=hypothesis(), prompt_evidence=[],
        prompt="prompt", evidence_prompt_text="", current_round=1, retrieval_trace=trace,
    )
    assert len(calls) == 2
    assert len(result) == 1
    if with_trace:
        assert trace[0]["generation_failures"][0]["sample_index"] == 0
        assert "transient generation error" in trace[0]["generation_failures"][0]["error"]


def test_single_sample_unrecoverable_error_is_still_raised(monkeypatch):
    def generate(*args, **kwargs):
        raise ValueError("transient generation error")

    monkeypatch.setattr(generation, "recover_truncated_grounded_answer", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="transient generation error"):
        generation.sample_grounded_conclusions(
            pipeline=pipeline(generate), question="问题？",
            current_hypothesis=hypothesis(), prompt_evidence=[],
            prompt="prompt", current_round=1,
        )
