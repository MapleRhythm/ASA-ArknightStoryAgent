import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "relabel_conclusion_to_exx.py"
SPEC = importlib.util.spec_from_file_location("relabel_conclusion_to_exx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def task(task_id: str, split: str = "train") -> object:
    return MODULE.Task(
        task_id=task_id,
        split=split,
        question="测试问题",
        hypothesis="{}",
        round_value="1/2",
        evidence=[],
        source_refs=[],
    )


def test_explicit_task_ids_do_not_bypass_completed_or_excluded_ids() -> None:
    selected = MODULE.select_pending_tasks(
        [task("done"), task("excluded"), task("keep"), task("val", "val")],
        completed_ids={"done"},
        excluded_ids={"excluded"},
        include_splits={"train"},
        requested_ids={"done", "excluded", "keep", "val"},
    )

    assert [item.task_id for item in selected] == ["keep"]


def test_load_task_ids_reads_checkpoint_without_importing_labels(tmp_path: Path) -> None:
    checkpoint = tmp_path / "labels.jsonl"
    checkpoint.write_text(
        "\n".join(
            (
                json.dumps({"task_id": "a", "label": {"next_action": "abstain"}}),
                json.dumps({"task_id": "b", "label": {"next_action": "answer_directly"}}),
                json.dumps({"task_id": "a", "label": {"next_action": "retrieve_more"}}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert MODULE.load_task_ids(checkpoint) == {"a", "b"}


def test_load_task_ids_rejects_malformed_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "labels.jsonl"
    checkpoint.write_text('{"task_id":"a"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(MODULE.LabelError, match="invalid_exclusion_jsonl"):
        MODULE.load_task_ids(checkpoint)


def test_teacher_prompt_exposes_supported_fact_limit() -> None:
    prompt = MODULE.teacher_prompt(task("prompt"))

    assert "1至8条" in prompt
    assert "不能输出第9条" in prompt
