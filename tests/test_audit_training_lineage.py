import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_training_lineage.py"
SPEC = importlib.util.spec_from_file_location("lineage_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(name, question, **meta):
    return {"id": name, "conversations": [{"value": "question: " + question}], "meta": meta}


def test_transitive_source_components_and_no_rowwise_split():
    first = row("train1", "甲做了什么？", source_refs=[{"id": "bridge"}])
    second = row("bridge", "完全不同问法？")
    heldout = row("test", "完全不同问法？")
    report = MODULE.build_report({"train": [first, second], "test": [heldout]}, [])
    assert len(report["cross_split_components"]) == 1
    train = [r for r in report["records"] if r["split"] == "train"]
    assert all("cross_split_question_or_lineage" in r["quarantine_reasons"] for r in train)
    assert len({r["component_id"] for r in report["records"]}) == 1


def test_family_sets_use_intersection_not_whole_joined_string():
    first = row("train", "不同的问题？", story_families=["activities/A", "activities/B"])
    heldout = row("test", "某个人去哪了？", story_families=["activities/B", "activities/C"])
    report = MODULE.build_report({"train": [first], "test": [heldout]}, [], 0)
    train = next(r for r in report["records"] if r["split"] == "train")
    assert train["heldout_story_overlap"] == ["activities/B"]
    assert "supervised_story_exposure_overlap" in train["quarantine_reasons"]
    assert not report["cross_split_components"]


def test_same_chapter_footprint_and_tasks_restore_missing_lineage():
    assert MODULE.story_footprint("obt/main/level_main_07-11_beg#chunk-1") == MODULE.story_footprint("obt/main/level_main_07-15_end#chunk-3")
    tasks = [{"task_id": "t1", "question": "新问题？", "source_refs": [],
              "evidence": [{"doc_id": "activities/unseen/level_01#chunk-1"}]}]
    report = MODULE.build_report({
        "train": [row("relabel-t1", "新问题？")],
        "test": [row("test", "完全不同？", story_families=["activities/other"])],
    }, tasks, 0)
    train = next(r for r in report["records"] if r["split"] == "train")
    assert train["partition_candidate"] and not train["training_ready"]
    assert train["story_footprints"] == ["activities/unseen"]


def test_unknown_metadata_quarantined_and_duplicates_fail_closed():
    report = MODULE.build_report({"train": [row("a", "问题？")], "test": [row("b", "别的问题？")]}, [], 0)
    assert "missing_story_provenance" in next(r for r in report["records"] if r["split"] == "train")["quarantine_reasons"]
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.build_report({"train": [row("a", "问题"), row("a", "问题")]}, [])


def test_near_duplicate_and_deterministic_order():
    splits = {
        "train": [row("a", "某个角色为什么决定离开家乡？", story_families=["activities/a"])],
        "test": [row("b", "某个角色为何决定离开家乡？", story_families=["activities/b"])],
    }
    report = MODULE.build_report(splits, [])
    assert report["counts"]["train:near_question_overlap"] == 1
    assert report == MODULE.build_report(dict(reversed(list(splits.items()))), [])
