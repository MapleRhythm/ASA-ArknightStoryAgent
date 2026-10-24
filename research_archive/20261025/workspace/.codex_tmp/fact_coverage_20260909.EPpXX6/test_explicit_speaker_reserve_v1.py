from copy import deepcopy
import pytest
from explicit_speaker_reserve_v1 import explicit_speakers, reserve_ranked


def choose(rows, anchors=("甲甲", "乙乙"), top_k=3, reserve=2):
    return reserve_ranked(rows, list(anchors), top_k=top_k, reserve=reserve,
                          id_of=lambda r: r["id"], text_of=lambda r: r["text"])


def rows():
    return [{"id": i, "text": text} for i, text in enumerate([
        "甲甲做其他事", "其他故事", "没有名字", "甲甲遇到乙乙", "乙乙向甲甲说话", "乙乙",
    ])]


def test_literal_question_only():
    assert explicit_speakers("甲甲如何救乙乙？", {"甲甲", "乙乙", "未出现", "甲甲？"}) == ["甲甲", "乙乙"]


def test_longest_overlapping_name():
    assert explicit_speakers("阿甲甲和乙乙", {"阿甲甲", "甲甲", "乙乙"}) == ["阿甲甲", "乙乙"]


def test_question_order_and_repeated_name():
    assert explicit_speakers("乙乙问甲甲，乙乙离开", {"甲甲", "乙乙"}) == ["乙乙", "甲甲"]


def test_no_guessed_alias():
    assert explicit_speakers("绰号甲去哪里", {"本名甲"}) == []


def test_reserve_preserves_every_row_and_text():
    source = rows()
    original = deepcopy(source)
    chosen = choose(source)
    assert [r["id"] for r in chosen[:3]] == [0, 3, 4]
    assert sorted(r["id"] for r in chosen) == list(range(6))
    assert source == original and all(r is source[r["id"]] for r in chosen)


@pytest.mark.parametrize("anchors", [(), ("甲甲",)])
def test_insufficient_anchors_identity(anchors):
    source = rows()
    assert choose(source, anchors=anchors) == source


def test_zero_reserve_identity():
    source = rows()
    assert choose(source, reserve=0) == source


def test_no_joint_document_identity():
    source = rows()[:3] + [{"id": 8, "text": "其他"}]
    assert choose(source) == source


def test_already_selected_identity():
    source = rows()[3:] + rows()[:3]
    assert choose(source) == source


@pytest.mark.parametrize("reserve", [-1, 4, True])
def test_invalid_reserve(reserve):
    with pytest.raises(ValueError, match="invalid top_k/reserve"):
        choose(rows(), reserve=reserve)


def test_duplicate_document_rejected():
    with pytest.raises(ValueError, match="duplicate document id"):
        choose(rows() + rows()[:1])


def test_duplicate_anchor_rejected():
    with pytest.raises(ValueError, match="invalid anchors"):
        choose(rows(), anchors=("甲甲", "甲甲"))


def test_corpus_vocabulary_permutation_invariant():
    assert explicit_speakers("阿甲甲问乙乙", ["甲甲", "乙乙", "阿甲甲"]) == explicit_speakers(
        "阿甲甲问乙乙", ["阿甲甲", "甲甲", "乙乙"])
