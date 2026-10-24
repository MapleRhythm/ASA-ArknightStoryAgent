"""No GPU tests for the request-level adapter switch and provenance."""
from types import SimpleNamespace

import pytest

from run_adapter_pairs_v1 import paired_dispatch


def rows():
    return [
        {"id": "off", "runtime_prompt": "same prompt", "sampling_seed": 42, "use_adapter": False},
        {"id": "on", "runtime_prompt": "same prompt", "sampling_seed": 42, "use_adapter": True},
    ]


def test_same_prompt_seed_only_adapter_differs():
    calls, seen = [], []
    adapter = SimpleNamespace(lora_int_id=1)

    def original(prompts, params, **kwargs):
        seen.append((list(prompts), params.seed, dict(kwargs)))
        return [SimpleNamespace(lora_request=kwargs["lora_request"])]

    generate = paired_dispatch(original, rows(), calls)
    for _ in range(2):
        generate(["same prompt"], SimpleNamespace(seed=None), lora_request=adapter, use_tqdm=False)
    assert seen[0][:2] == seen[1][:2]
    assert seen[0][2]["lora_request"] is None and seen[1][2]["lora_request"] is adapter
    assert seen[0][2]["use_tqdm"] is seen[1][2]["use_tqdm"] is False
    assert [c["adapter_active_at_call"] for c in calls] == [False, True]
    assert [c["seed"] for c in calls] == [42, 42]
    assert all(c["status"] == "returned" for c in calls)


@pytest.mark.parametrize("prompts", [[], ["wrong"], ["same prompt", "extra"]])
def test_changed_or_batched_prompt_rejected(prompts):
    called = []
    dispatch = paired_dispatch(lambda *a, **k: called.append(True), rows(), [])
    with pytest.raises(AssertionError):
        dispatch(prompts, SimpleNamespace(), lora_request=object())
    assert called == []


def test_absent_configured_adapter_is_not_silently_accepted():
    dispatch = paired_dispatch(lambda *a, **k: [], rows(), [])
    with pytest.raises(AssertionError):
        dispatch(["same prompt"], SimpleNamespace(), lora_request=None)


def test_extra_generation_is_rejected():
    calls = []
    dispatch = paired_dispatch(lambda *a, **k: [], rows(), calls)
    for _ in range(2):
        dispatch(["same prompt"], SimpleNamespace(), lora_request=object())
    with pytest.raises(AssertionError):
        dispatch(["same prompt"], SimpleNamespace(), lora_request=object())


def test_failure_is_recorded_and_propagated():
    calls = []

    def fails(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        paired_dispatch(fails, rows(), calls)(
            ["same prompt"], SimpleNamespace(), lora_request=object(),
        )
    assert calls[0]["status"] == "error"
    assert "synthetic failure" in calls[0]["error"]
