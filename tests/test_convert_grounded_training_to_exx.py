import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_grounded_training_to_exx.py"
SPEC = importlib.util.spec_from_file_location("convert_grounded_training_to_exx", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def grounded_record(*, evidence_id: str, quote: str) -> dict:
    prompt = """任务：根据证据决定RAG下一步动作，只输出JSON。
问题：测试问题？
轮次：1/2
当前检索假设：{"query_type":"fact"}
证据：
E1 doc/one#chunk-0001
第一条完整证据包含唯一短语甲。

E2 doc/two#chunk-0002
第二条完整证据包含唯一短语乙。
输出格式：
legacy schema
"""
    output = {
        "next_action": "answer_directly",
        "supported_facts": [
            {
                "id": "sf1",
                "fact": "这是一个可核验事实。",
                "evidence_refs": [{"evidence_id": evidence_id, "quote": quote}],
            }
        ],
        "inferred_facts": [],
        "final_answer": "旧自由答案",
    }
    return {
        "id": "fixture-1",
        "task_type": "grounded_action_generation",
        "system": "legacy",
        "tools": "",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json.dumps(output, ensure_ascii=False)},
        ],
        "meta": {"question_key": "q1", "gold": "must not enter prompt"},
    }


def test_maps_visible_doc_id_and_removes_quote_fields() -> None:
    converted = MODULE.convert_grounded_record(
        grounded_record(evidence_id="doc/two#chunk-0002", quote="唯一短语乙"), "fixture"
    ).record
    prompt = converted["conversations"][0]["value"]
    output = json.loads(converted["conversations"][-1]["value"])

    assert "[E1]" in prompt and "[E2]" in prompt
    assert "第一条完整证据" in prompt and "第二条完整证据" in prompt
    assert "must not enter prompt" not in prompt
    assert output == {
        "next_action": "answer_directly",
        "supported_facts": [{"fact": "这是一个可核验事实。", "evidence_ids": ["E2"]}],
    }
    serialized = json.dumps(output, ensure_ascii=False)
    assert "quote" not in serialized
    assert "final_answer" not in serialized
    assert "inferred_facts" not in serialized


def test_maps_unique_quote_when_legacy_id_is_not_usable() -> None:
    converted = MODULE.convert_grounded_record(
        grounded_record(evidence_id="legacy-unknown", quote="唯一短语甲"), "fixture"
    ).record
    output = json.loads(converted["conversations"][-1]["value"])
    assert output["supported_facts"][0]["evidence_ids"] == ["E1"]


def test_rejects_reference_not_provable_from_visible_prompt() -> None:
    with pytest.raises(MODULE.ConversionError, match="unresolved_evidence_reference"):
        MODULE.convert_grounded_record(
            grounded_record(evidence_id="doc/missing", quote="模型不可见的金标引文"), "fixture"
        )


def test_rejects_ambiguous_quote_instead_of_guessing() -> None:
    record = grounded_record(evidence_id="legacy-unknown", quote="完整证据")
    with pytest.raises(MODULE.ConversionError, match="ambiguous_quote_mapping"):
        MODULE.convert_grounded_record(record, "fixture")


def test_preserves_legacy_inferred_answer_as_evidence_bound_fact() -> None:
    record = grounded_record(evidence_id="doc/two#chunk-0002", quote="唯一短语乙")
    output = json.loads(record["conversations"][-1]["value"])
    output["inferred_facts"] = [
        {
            "id": "inf1",
            "fact": "这是由已引用事实直接归纳出的答案。",
            "premise_fact_ids": ["sf1"],
            "inference_type": "direct_summary",
        }
    ]
    record["conversations"][-1]["value"] = json.dumps(output, ensure_ascii=False)

    converted = MODULE.convert_grounded_record(record, "fixture").record
    converted_output = json.loads(converted["conversations"][-1]["value"])
    assert converted_output["supported_facts"][-1] == {
        "fact": "这是由已引用事实直接归纳出的答案。",
        "evidence_ids": ["E2"],
    }
