from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_answer_slots_v2 as v2
import probe_answer_slots_v3 as v3

HERE = Path(__file__).resolve().parent


def test_exactly_one_cause_scope_clarification():
    assert v3.RUBRIC.replace(v3.CLARIFICATION, "") == v2.RUBRIC
    assert v3.VERSION == v2.VERSION and v3.validate is v2.validate


def test_independent_fixture_is_complete_and_unambiguous_in_shape():
    data = json.loads((HERE / "slot_cause_scope_synthetic20_v1.json").read_text())
    assert data["evaluation_only"] and data["training_ready"] is False
    assert len(data["cases"]) == len({c["id"] for c in data["cases"]}) == 20
    assert Counter(c["expected_coverage"] for c in data["cases"]) == {"complete": 11, "none": 7, "partial": 2}
    assert len({(c["question"], c["shown_answer"]) for c in data["cases"]}) == 20


def test_fixture_oracles_and_ids_never_enter_requests():
    data = json.loads((HERE / "slot_cause_scope_synthetic20_v1.json").read_text())
    for case in data["cases"]:
        before = deepcopy(case)
        sent = json.loads(v3.messages(case)[1]["content"])
        assert set(sent) == {"question", "shown_answer"}
        assert v3.messages(case)[1] == v2.messages(case)[1]
        assert case == before


def test_empty_slot_field_fix_is_retained():
    assert "unfilled的answer_spans必须为[]" in v3.RUBRIC
    assert "除非问题明确问到" in v3.RUBRIC
    assert "仅复述被问主体的同一行为，不构成原因" in v3.RUBRIC
