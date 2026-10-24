import copy
import hashlib

import pytest

from build_question_focus_paired18_v1 import FOCUS, build_rows
from run_paired_vllm_retry_v1 import seeded_generate


def render(system, user, *, assistant_prefix):
    return f"{system}\n{user}\n{assistant_prefix}"


def example():
    prompt = render("SYSTEM", "QUESTION AND COMPLETE EVIDENCE", assistant_prefix="{")
    return {
        "id": "example", "question": "who?", "hypothesis": {"entities": ["someone"]},
        "system": "SYSTEM", "user": "QUESTION AND COMPLETE EVIDENCE", "assistant_prefix": "{",
        "runtime_prompt": prompt, "runtime_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "evidence_brief": "Full visible text.", "prompt_evidence": [{"id": "E1", "text": "Full visible text."}],
        "current_round": 1, "max_retrieval_rounds": 2,
    }


def test_control_unchanged_and_focus_only_appends_generic_rules():
    source = [example()]
    before = copy.deepcopy(source)
    by_recipe = {r["renderer"]: r for r in build_rows(source, render)}
    assert source == before
    control, focus = by_recipe["focus_control_v1"], by_recipe["question_focus_v1"]
    assert control["runtime_prompt"] == source[0]["runtime_prompt"]
    assert focus["user"] == source[0]["user"] + "\n" + FOCUS
    for key in ("question", "hypothesis", "system", "assistant_prefix", "evidence_brief",
                "prompt_evidence", "current_round", "max_retrieval_rounds"):
        assert control[key] == focus[key] == source[0][key]
    assert control["sampling_seed"] == focus["sampling_seed"]
    assert len({control["id"], focus["id"]}) == 2


def test_no_previous_failure_or_candidate_or_gold_is_injected():
    assert "上一次" not in FOCUS
    assert "芙蕾达" not in FOCUS and "娜斯提" not in FOCUS
    assert build_rows([example()], render) == build_rows([example()], render)


def test_duplicate_source_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        build_rows([example(), example()], render)


def test_seeded_wrapper_rejects_different_prompt():
    from types import SimpleNamespace
    rows = build_rows([example()], render)
    calls = []
    wrapped = seeded_generate(lambda *args, **kwargs: "result", rows, calls)
    with pytest.raises(AssertionError, match="input changed"):
        wrapped(["changed"], SimpleNamespace())
    assert calls == []
