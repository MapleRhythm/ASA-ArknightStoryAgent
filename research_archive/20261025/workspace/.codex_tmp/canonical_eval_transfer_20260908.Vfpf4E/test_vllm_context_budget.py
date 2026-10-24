from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from asa_arknight_story_agent.inference.model_runtime.vllm_runner import VllmRunner


class Tokenizer:
    def __init__(self, length):
        self.length = length
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return list(range(self.length))


class Engine:
    def __init__(self, length, finish="stop"):
        self.tokenizer = Tokenizer(length)
        self.calls = []
        self.finish = finish

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, params, **kwargs):
        self.calls.append((prompts, params, kwargs))
        return [SimpleNamespace(outputs=[
            SimpleNamespace(text='{"ok":true}', token_ids=[1, 2], finish_reason=self.finish, stop_reason=None)
        ])]


def make_runner(monkeypatch, length=6, capacity=10, tokens=4, finish="stop"):
    class Sampling:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Sampling))
    runner = VllmRunner(base_model_path=Path("/nonexistent-test-model"),
                        max_model_len=capacity, max_tokens=tokens)
    engine = Engine(length, finish)
    runner._llm = engine
    return runner, engine


def test_exact_boundary_passes_with_unchanged_input_and_budget(monkeypatch):
    runner, engine = make_runner(monkeypatch)
    assert runner.generate("完整原文") == '{"ok":true}'
    assert engine.calls[0][0] == ["完整原文"]
    assert engine.calls[0][1].max_tokens == 4
    assert engine.tokenizer.calls == [("完整原文", {"add_special_tokens": True})]
    assert runner.last_generation_diagnostics["available_output_tokens"] == 4
    assert runner.last_generation_diagnostics["input_truncated"] is False


@pytest.mark.parametrize("length,tokens,capacity", [(7, 4, 10), (10, 1, 10), (11, 1, 10)])
def test_overflow_rejected_before_engine_generate(monkeypatch, length, tokens, capacity):
    runner, engine = make_runner(monkeypatch, length, capacity, tokens)
    with pytest.raises(ValueError, match="No prompt or output budget was truncated"):
        runner.generate("全部证据")
    assert not engine.calls
    assert runner.last_generation_diagnostics["status"] == "context_budget_exceeded"


def test_explicit_output_override_is_reserved(monkeypatch):
    runner, engine = make_runner(monkeypatch, tokens=9)
    assert runner.generate("P", max_tokens=4)
    assert engine.calls[0][1].max_tokens == 4
    with pytest.raises(ValueError, match="context budget"):
        runner.generate("P", max_tokens=5)


def test_generation_finish_reason_and_token_count_are_not_discarded(monkeypatch):
    runner, _ = make_runner(monkeypatch, finish="length")
    runner.generate("P")
    assert runner.last_generation_diagnostics["finish_reason"] == "length"
    assert runner.last_generation_diagnostics["generated_tokens"] == 2


def test_diagnostics_reset_between_requests(monkeypatch):
    runner, _ = make_runner(monkeypatch)
    runner.generate("P")
    with pytest.raises(ValueError):
        runner.generate("P", max_tokens=5)
    assert "finish_reason" not in runner.last_generation_diagnostics


def test_rejects_zero_output_budget(monkeypatch):
    runner, engine = make_runner(monkeypatch)
    with pytest.raises(ValueError, match="positive"):
        runner.generate("P", max_tokens=0)
    assert not engine.calls
