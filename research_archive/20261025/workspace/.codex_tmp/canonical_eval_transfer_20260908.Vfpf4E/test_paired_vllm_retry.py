from types import SimpleNamespace

import pytest

from run_paired_vllm_retry_v1 import seeded_generate


def test_seed_override_is_actually_passed_and_duplicate_prompts_keep_their_rows():
    rows = [
        {"id": "a", "runtime_prompt": "same", "sampling_seed": 13},
        {"id": "b", "runtime_prompt": "same", "sampling_seed": 29},
    ]
    actual, calls = [], []
    def original(prompts, params, **kwargs):
        actual.append((prompts, params.seed, kwargs))
        return "output"
    generate = seeded_generate(original, rows, calls)
    for seed in (13, 29):
        assert generate(["same"], SimpleNamespace(seed=None), use_tqdm=False) == "output"
        assert actual[-1][1] == seed
    assert [r["id"] for r in calls] == ["a", "b"]
    assert [r["seed"] for r in calls] == [13, 29]
    with pytest.raises(AssertionError, match="extra"):
        generate(["same"], SimpleNamespace(seed=None))


def test_changed_prompt_is_refused_before_generation():
    actual, calls = [], []
    generate = seeded_generate(
        lambda *a, **kw: actual.append(a),
        [{"id": "a", "runtime_prompt": "frozen", "sampling_seed": 1}], calls,
    )
    with pytest.raises(AssertionError, match="input changed"):
        generate(["modified"], SimpleNamespace(seed=None))
    assert actual == calls == []
