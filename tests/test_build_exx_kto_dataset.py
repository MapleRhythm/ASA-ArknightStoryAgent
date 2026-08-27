import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_exx_kto_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_exx_kto_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def source(tag: bool, payload: dict) -> dict:
    return {
        "kto_tag": tag,
        "conversations": [
            {"from": "human", "value": "prompt"},
            {"from": "gpt", "value": json.dumps(payload, ensure_ascii=False)},
        ],
    }


def task(round_value: str = "1/2") -> dict:
    return {
        "task_id": "t1",
        "split": "train",
        "question": "测试问题",
        "hypothesis": "{}",
        "round": round_value,
        "evidence": [{"label": "E1", "doc_id": "d1", "text": "证据"}],
        "source_refs": [],
    }


def test_converts_explicit_negative_retrieve_more() -> None:
    payload = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {"question": "错误的续检索问题"},
    }
    assert MODULE.convert_legacy_negative(source(False, payload), task()) == payload


def test_rejects_positive_legacy_row_as_negative() -> None:
    with pytest.raises(ValueError, match="polarity_not_explicitly_negative"):
        MODULE.convert_legacy_negative(
            source(True, {"next_action": "abstain"}), task()
        )


def test_quarantines_unbound_legacy_direct_answer() -> None:
    with pytest.raises(ValueError, match="no_deterministic_exx_binding"):
        MODULE.convert_legacy_negative(
            source(False, {"next_action": "answer_directly", "answer": "猜测"}), task()
        )


def test_rejects_retrieve_more_at_max_round() -> None:
    with pytest.raises(MODULE.RELABEL.LabelError, match="retrieve_more_at_max_round"):
        MODULE.convert_legacy_negative(
            source(
                False,
                {
                    "next_action": "retrieve_more",
                    "follow_up_hypothesis": {"question": "继续查"},
                },
            ),
            task("2/2"),
        )


def test_uses_explicitly_rejected_first_pass_as_negative() -> None:
    task_object = MODULE.RELABEL.Task(
        task_id="t1",
        split="train",
        question="测试问题",
        hypothesis="{}",
        round_value="1/2",
        evidence=[MODULE.RELABEL.CONVERTER.Evidence("E1", "d1", "证据")],
        source_refs=[],
    )
    positive = {"next_action": "abstain", "reason": "现有证据不足以确认。"}
    first_pass = {
        "next_action": "retrieve_more",
        "follow_up_hypothesis": {"question": "继续查"},
    }
    result = {
        "first_pass_label": first_pass,
        "semantic_verifier": {"valid": False, "issues": ["OVER_RETRIEVE"]},
    }

    assert MODULE.semantic_verifier_negative(result, task_object, positive) == first_pass


def test_does_not_infer_negative_without_explicit_verifier_rejection() -> None:
    assert MODULE.semantic_verifier_negative(
        {"first_pass_label": {"next_action": "abstain"}}, object(), {"next_action": "abstain"}
    ) is None


def test_remove_unpaired_prompts_drops_orphan_after_split_isolation() -> None:
    stats = MODULE.Counter()
    base = {
        "system": "s",
        "conversations": [{"from": "human", "value": "p"}, {"from": "gpt", "value": "x"}],
    }
    paired = [{**base, "id": "p", "kto_tag": True}, {**base, "id": "n", "kto_tag": False}]
    orphan = {
        "system": "s",
        "conversations": [{"from": "human", "value": "orphan"}, {"from": "gpt", "value": "x"}],
        "id": "o",
        "kto_tag": True,
    }

    assert [row["id"] for row in MODULE.remove_unpaired_prompts(paired + [orphan], stats, "train")] == [
        "p",
        "n",
    ]
    assert stats["quarantine:train:unpaired_after_isolation"] == 1
