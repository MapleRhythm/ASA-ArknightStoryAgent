import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_reader_natural_language_control.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_reader_natural_language_control", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_row(*, sufficiency: str = "sufficient") -> dict:
    return {
        "id": "q1::minimal_sufficient",
        "task_type": "grounded_action_generation",
        "system": "只输出 JSON。",
        "conversations": [
            {
                "from": "human",
                "value": (
                    "task: grounded_action_generation\n"
                    "question: 谁做了什么？\n"
                    "round: 1/2\n"
                    "evidence:\n"
                    "[E1]\n甲完成了任务。\n"
                    "output_schema: grounded_action_exx_v1\n"
                    "rules: 只输出 JSON。"
                ),
            }
        ],
        "diagnostic": {
            "source_id": "q1",
            "question_family": "family-1",
            "condition": "minimal_sufficient",
            "reference_sufficiency": sufficiency,
        },
    }


def sample_annotations() -> dict[str, dict]:
    return {
        "q1": {
            "id": "q1",
            "review_status": "approved",
            "reviewer": "human",
            "answer_facts": [{"fact": "隐藏答案要点"}],
        }
    }


def test_build_control_preserves_evidence_without_leaking_rubric() -> None:
    rows = MODULE.build_control_rows(
        [sample_row()],
        sample_annotations(),
        source_condition="minimal_sufficient",
    )

    assert len(rows) == 1
    row = rows[0]
    prompt = row["conversations"][0]["value"]
    assert "谁做了什么？" in prompt
    assert "[E1]\n甲完成了任务。" in prompt
    assert "output_schema" not in prompt
    assert "rules:" not in prompt
    assert "隐藏答案要点" not in prompt
    assert row["diagnostic"]["reference_answer_facts"] == ["隐藏答案要点"]
    assert row["diagnostic"]["condition"] == (
        "minimal_sufficient_natural_language"
    )


def test_build_control_requires_reviewed_sufficient_evidence() -> None:
    with pytest.raises(ValueError, match="requires sufficient"):
        MODULE.build_control_rows(
            [sample_row(sufficiency="insufficient")],
            sample_annotations(),
            source_condition="minimal_sufficient",
        )


def test_extract_question_and_evidence_rejects_non_exx_prompt() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        MODULE.extract_question_and_evidence("普通问题")
