import pytest

from chain_owner_protocol import with_explicit_owner


def test_nested_numbers_replaced_by_outer_owner():
    source = "[E2] [CHAIN_MEMBER_1] 甲：原话。 [CHAIN_MEMBER_2] 乙：不是甲。\n[E7] 丙。"
    result, receipt = with_explicit_owner(source)
    assert result == "[E2] 〔引用归属:E2〕 甲：原话。 〔引用归属:E2〕 乙：不是甲。\n[E7] 丙。"
    assert [r["member_count"] for r in receipt] == [2, 0]


def test_whole_counterevidence_and_source_headers_retained():
    source = ("[E1] [CHAIN_MEMBER_1] 活动 / 章节 / ST-1 甲说乙在背后。\n"
              "[E5] 乙其实在台下；那只手不是乙的。")
    result, _ = with_explicit_owner(source)
    assert "活动 / 章节 / ST-1 甲说乙在背后。" in result
    assert result.endswith("[E5] 乙其实在台下；那只手不是乙的。")


def test_identity_without_member_metadata():
    source = "[E1] 无链条原文。\n[E9] 另一完整段落。"
    assert with_explicit_owner(source)[0] == source


def test_whitespace_and_internal_newlines_preserved():
    source = "[E3] [CHAIN_MEMBER_1]  甲\n乙\t丙  [CHAIN_MEMBER_2] 丁"
    assert with_explicit_owner(source)[0] == "[E3] 〔引用归属:E3〕  甲\n乙\t丙  〔引用归属:E3〕 丁"


@pytest.mark.parametrize("source", [
    "[E1] [CHAIN_MEMBER_0] 甲",
    "[E1] [CHAIN_MEMBER_2] 甲",
    "[E1] [CHAIN_MEMBER_1] 甲 [CHAIN_MEMBER_1] 乙",
    "[E1] [CHAIN_MEMBER_1] 甲 [CHAIN_MEMBER_3] 乙",
    "[E1] 〔引用归属:E1〕 甲",
    "[E1] [CHAIN_MEMBER_bad] 甲",
    "[E1] 甲\n[E1] 乙",
    "缺少编号",
])
def test_invalid_or_ambiguous_metadata_is_not_silently_repaired(source):
    with pytest.raises(ValueError):
        with_explicit_owner(source)
