from copy import deepcopy

import pytest

from run_single_pointer_handoff4_v1 import build_row
from single_quote_pointer_protocol_v1 import make_user, strict_loads
from analyze_single_pointer_handoff4_v1 import assemble
from test_single_quote_pointer_v1 import original


def render(system, user, assistant_prefix):
    return "SYSTEM:" + system + "\nUSER:" + user + "\nASSISTANT:" + assistant_prefix


def control():
    return {**original(), "id": "old-state", "renderer": "old-handoff", "system": "original system",
            "assistant_prefix": "{", "question": "原问题", "sampling_seed": 71,
            "hypothesis": {"question": "原问题"}, "prompt_evidence": [{"document": {"id": "source"}}],
            "packing_trace": [{"status": "included"}], "current_round": 2, "max_retrieval_rounds": 2}


def test_row_only_changes_protocol_not_evidence_seed_or_question():
    old = control()
    frozen = deepcopy(old)
    row = build_row(old, render)
    assert old == frozen
    for field in ("question", "system", "assistant_prefix", "sampling_seed", "hypothesis",
                  "prompt_evidence", "packing_trace", "current_round", "max_retrieval_rounds", "evidence_brief"):
        assert row[field] == old[field]
    assert row["control_record_id"] == old["id"] and row["id"] != old["id"]


def test_row_renders_the_exact_single_pointer_protocol():
    old = control()
    row = build_row(old, render)
    user, annotated, catalog = make_user(old)
    assert row["user"] == user and row["quote_catalog"] == catalog and row["annotated_evidence"] == annotated
    assert row["runtime_prompt"] == render(old["system"], user, "{")


def test_assembler_only_prepends_known_prefill_without_json_repair():
    raw = '"next_action":"abstain"}'
    assembled = assemble({"assistant_prefix": "{"}, {"raw_output": raw})
    assert assembled == "{" + raw
    assert strict_loads(assembled) == {"next_action": "abstain"}


@pytest.mark.parametrize("raw", ['{"next_action":"abstain"}', '"next_action":"abstain"', 'NOT JSON'])
def test_assembler_does_not_repair_malformed_continuation(raw):
    with pytest.raises(ValueError):
        strict_loads(assemble({"assistant_prefix": "{"}, {"raw_output": raw}))


def test_unexpected_prefill_rejected():
    with pytest.raises(ValueError, match="unexpected known prefill"):
        assemble({"assistant_prefix": ""}, {"raw_output": "{}"})
