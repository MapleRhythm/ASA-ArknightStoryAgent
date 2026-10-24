import copy
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_rendered_evidence_visibility.py"
SPEC = importlib.util.spec_from_file_location("audit_rendered_evidence_visibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(doc_id, text, score, **extras):
    return {"id": doc_id, "clean_text": text, "fusion_score": score, **extras}


def case(items, gold=("gold",), minimal=None, groups=None):
    row = {
        "id": "q", "question": "申请是否获准？", "gold_evidence_ids": list(gold),
        "stages": {"rerank": items},
    }
    review = {
        "id": "q", "minimal_sufficient_evidence_ids": list(minimal or gold),
        "answer_facts": [{
            "fact": "仅用于事后计算覆盖，不可加入选择器。",
            "support_groups": groups or [list(minimal or gold)],
        }],
    }
    return row, review


def audit(row, review, **overrides):
    kwargs = dict(
        mode="evidence_id", strategy="production", packing="break",
        top_k=10, per_doc=1000, total=8000,
    )
    kwargs.update(overrides)
    return MODULE.audit_case(row, review, **kwargs)


def test_chain_metadata_is_not_rendered_direct_text():
    chain = "甲：今天晴朗。乙：申请未获准，阿米娅没有同意。"
    items = [
        candidate("outer", "甲：今天晴朗。", 10, evidence_chain_text=chain,
                  evidence_chain_member_ids=["outer", "gold"]),
        candidate("gold", "乙：申请未获准，阿米娅没有同意。", 9,
                  evidence_chain_text=chain, evidence_chain_member_ids=["outer", "gold"]),
    ]
    row, review = case(items)
    before = copy.deepcopy(row)
    result = audit(row, review)
    assert result["legacy_gold"]["chain_metadata_ids"]["all"]
    assert not result["legacy_gold"]["rendered_full_text"]["all"]
    assert "申请未获准" not in result["evidence_brief"]
    assert row == before
    aligned = audit(row, review, strategy="aligned_similarity")
    assert aligned["legacy_gold"]["rendered_full_text"]["all"]
    assert row == before
    quote = audit(row, review, mode="quote")
    assert quote["legacy_gold"]["rendered_full_text"]["all"]


def test_quote_preserves_chain_member_text_despite_old_per_doc_limit():
    chain = "甲：" + "现在只有背景信息。" * 20 + "乙：申请未获准。"
    row, review = case([
        candidate("outer", "甲：只有背景。", 10, evidence_chain_text=chain,
                  evidence_chain_member_ids=["outer", "gold"]),
        candidate("gold", "乙：申请未获准。", 9, evidence_chain_text=chain),
    ])
    result = audit(row, review, mode="quote", per_doc=50)
    assert result["legacy_gold"]["chain_metadata_ids"]["all"]
    assert result["legacy_gold"]["rendered_full_text"]["all"]
    assert "…" not in result["evidence_brief"]


def test_budget_break_and_offline_skip_are_distinct():
    row, review = case([
        candidate("first", "甲", 10),
        candidate("large", "乙" * 70, 9),
        candidate("gold", "丙", 8),
    ])
    result = audit(row, review, strategy="identity_only", total=20)
    assert result["rendered_ids"] == ["first"]
    assert result["packing_trace"][1]["status"] == "budget_break"
    assert result["packing_trace"][2]["status"] == "after_budget_break"
    assert result["packing_trace"][2]["fits_remaining"]
    alternative = audit(row, review, strategy="identity_only", total=20, packing="skip")
    assert alternative["rendered_ids"] == ["first", "gold"]
    assert alternative["blocks"][1]["evidence_id"] == "E2"
    assert alternative["legacy_gold"]["rendered_full_text"]["all"]


@pytest.mark.parametrize("mode,packing", [("quote", "skip"), ("evidence_id", "break")])
def test_runtime_packing_is_verified_without_prepacking(monkeypatch, mode, packing):
    row, review = case([
        candidate("first", "甲", 10),
        candidate("large", "乙" * 70, 9),
        candidate("gold", "丙", 8),
    ])
    actual_builder = MODULE.build_minimal_conclusion_prompt
    seen_ids = []

    def capture_builder(**kwargs):
        seen_ids.extend(MODULE.hit_id(item) for item in kwargs["prompt_evidence"])
        return actual_builder(**kwargs)

    monkeypatch.setattr(MODULE, "build_minimal_conclusion_prompt", capture_builder)
    result = audit(
        row, review, mode=mode, packing=packing, strategy="identity_only", total=20,
    )
    assert seen_ids == ["first", "large", "gold"]
    assert result["runtime_packing_verified"]
    assert result["rendered_ids"] == (["first", "gold"] if mode == "quote" else ["first"])


def test_oversized_first_block_yields_empty_visible_evidence():
    row, review = case([candidate("gold", "甲" * 50, 10)])
    result = audit(row, review, total=20)
    assert result["block_count"] == result["counted_chars"] == 0
    assert result["evidence_brief"] == ""
    assert not result["legacy_gold"]["rendered_full_text"]["all"]


def test_text_can_survive_under_another_outer_id():
    row, review = case([
        candidate("outer", "甲：申请获准。", 10),
        candidate("gold", "甲：申请获准。", 9),
    ])
    result = audit(row, review)
    assert not result["legacy_gold"]["outer_ids"]["all"]
    assert result["legacy_gold"]["rendered_full_text"]["all"]


def test_reviewed_alternative_support_not_all_legacy_ids_required():
    row, review = case([
        candidate("current", "甲的申请获准。", 10),
        candidate("gold", "另一段相同事件的长篇叙述。", 1),
    ], minimal=("current",), groups=[["gold"], ["current"]])
    result = audit(row, review, top_k=1)
    assert not result["legacy_gold"]["rendered_full_text"]["all"]
    assert result["reviewed_minimal"]["rendered_full_text"]["all"]
    assert result["all_reviewed_fact_groups_full_text_visible"]


def test_exact_containment_does_not_treat_negation_as_duplicate():
    row, review = case([
        candidate("negative", "阿米娅没有同意申请。", 10),
        candidate("gold", "阿米娅同意申请。", 9),
    ])
    result = audit(row, review, strategy="exact_containment")
    assert result["selected_ids"] == ["negative", "gold"]


def test_gold_changes_do_not_change_selected_or_rendered_text():
    row, review = case([
        candidate("gold", "甲在学校。", 10),
        candidate("other", "乙在城外。", 9),
    ])
    changed = copy.deepcopy(row)
    changed["gold_evidence_ids"] = ["other"]
    for strategy in MODULE.STRATEGIES:
        original = audit(row, review, top_k=1, strategy=strategy)
        alternative = audit(changed, review, top_k=1, strategy=strategy)
        assert original["selected_ids"] == alternative["selected_ids"]
        assert original["prompt_sha256"] == alternative["prompt_sha256"]


def test_missing_reference_text_fails_closed():
    row, review = case([candidate("unrelated", "甲在学校。", 10)])
    with pytest.raises(ValueError, match="missing source text"):
        audit(row, review)


def test_conflicting_text_for_same_id_is_rejected():
    row, review = case([candidate("gold", "甲在学校。", 10)])
    row["stages"]["dense"] = [candidate("gold", "甲不在学校。", 1)]
    with pytest.raises(ValueError, match="conflicting source text"):
        audit(row, review)


def test_actual_newline_chars_are_not_hidden_in_budget_count():
    row, review = case([
        candidate("gold", "甲", 10), candidate("second", "乙", 9),
    ])
    result = audit(row, review, strategy="identity_only", total=12)
    assert result["block_count"] == 2
    assert result["counted_chars"] == 12
    assert result["evidence_chars"] == 13


def test_empty_document_does_not_consume_number():
    row, review = case([
        candidate("empty", "", 10), candidate("gold", "甲", 9),
    ])
    result = audit(row, review, strategy="identity_only")
    assert result["blocks"][0]["source_id"] == "gold"
    assert result["blocks"][0]["evidence_id"] == "E1"


def test_real_configs_are_supported():
    for name, total in (("evidence_id", 8000), ("quote", 12000)):
        suffix = "_evidence_id" if name == "evidence_id" else ""
        path = SCRIPT.parents[1] / f"configs/runtime_gpu_reranker_qwen35_4b{suffix}.json"
        profile = MODULE.profile_from_config(path)
        assert profile["mode"] == name
        assert profile["total"] == total


def test_direct_render_ablation_keeps_production_selection_and_quote_protocol():
    chain = "甲：" + "这是一段无关叙述。" * 20 + "乙：申请未获准。"
    row, review = case([
        candidate("gold", "乙：申请未获准。", 10, evidence_chain_text=chain),
        candidate("other", "甲：今天晴朗。", 9, evidence_chain_text=chain),
    ])
    baseline = audit(row, review, mode="quote", per_doc=50)
    direct = audit(row, review, mode="quote", per_doc=50, rendering="direct_only")
    assert baseline["selected_ids"] == direct["selected_ids"]
    assert baseline["legacy_gold"]["rendered_full_text"]["all"]
    assert direct["legacy_gold"]["rendered_full_text"]["all"]
    assert direct["evidence_brief"] == "[E1] 乙：申请未获准。"
