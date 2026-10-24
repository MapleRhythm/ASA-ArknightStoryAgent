"""Synthetic-only tests: cleaning must never manufacture an approval."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(os.environ.get(
    "ASA_CLEANER_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "scripts/clean_soda_api_verifier_dataset.py"),
))
spec = importlib.util.spec_from_file_location("soda_cleaner_target", SCRIPT)
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)

KEYS = ["9fa94eaa04a4a919", "e115e5cf6a2fa71b", "f2e058438aab8018", "391172ae6f3fff4d"]


def record(index=0, *, question="示例问题？", prompt=None, tag=True):
    return {
        "id": f"row{index}", "task_type": "conclusion_generation", "kto_tag": tag,
        "conversations": [
            {"from": "human", "value": f"question: {question}\nround: {index}\n[E1] 原文"},
            {"from": "gpt", "value": '{"next_action":"answer_directly","answer":"待审核答案"}'},
        ],
        "meta": {
            "prompt_key": prompt or f"state{index}",
            "api_verifier_reason": "verifier_chosen_answer_directly",
            "api_verifier": {
                "supported_answer": "审核当时看到的答案",
                "teacher_action_error": "unsupported_answer",
                "teacher_answer_uses_prior_knowledge": True,
                "label_reason": "原始审核意见",
            },
        },
    }


def run_clean(tmp_path, monkeypatch, records, ratio=0.3, output=None):
    source = tmp_path / "input"
    source.mkdir(exist_ok=True)
    (source / "train.json").write_text(json.dumps(records), encoding="utf-8")
    (source / "val.json").write_text("[]", encoding="utf-8")
    raw_journal = '{"prompt_key":"test", "raw_response":"保留空格与原始审核"}\n'
    (source / "api_verifier_records.jsonl").write_text(raw_journal, encoding="utf-8")
    args = SimpleNamespace(
        input_dir=source, output_dir=output or tmp_path / "output",
        dataset_name="synthetic", seed=7, val_ratio=ratio,
    )
    monkeypatch.setattr(cleaner, "parse_args", lambda: args)
    cleaner.main()
    return args.output_dir, raw_journal


@pytest.mark.parametrize("key", KEYS)
def test_known_disputed_positive_is_quarantined_not_rewritten(key):
    original = record(prompt=key)
    before = deepcopy(original)
    kept, stats = cleaner.clean_records([original])
    assert original == before
    assert kept == [], "Unreviewed rewrite must not inherit the original positive label"
    assert stats["quarantined_records"] == 1


def test_quarantine_covers_related_question_states_and_negative_pair():
    rows = [record(prompt=KEYS[0]), record(1, tag=False), record(2, question="其他问题？")]
    before = deepcopy(rows)
    kept, stats = cleaner.clean_records(rows)
    assert rows == before
    assert kept == [rows[2]]
    assert stats["quarantined_records"] == 2


def test_historical_clean_v1_marker_is_quarantined_even_after_prompt_id_changes():
    old = record(prompt="different_prompt")
    old["meta"]["clean_v1_patch_reason"] = "historic patch"
    kept, _ = cleaner.clean_records([old])
    assert kept == []


def test_verifier_only_historical_patch_marker_is_not_missed():
    old = record()
    old["meta"]["api_verifier"]["label_reason"] = "clean_v1: historical patch"
    kept, _ = cleaner.clean_records([old])
    assert kept == []


def test_ordinary_record_and_invalid_json_negative_are_not_rewritten():
    positive, negative = record(), record(1, question="另一题", tag=False)
    negative["conversations"][-1]["value"] = "not json; actual rejected model output"
    rows = [positive, negative]
    before = deepcopy(rows)
    kept, _ = cleaner.clean_records(rows)
    assert rows == before and kept == before


def test_split_is_by_question_across_different_prompt_ids():
    rows = [record(n) for n in range(12)] + [record(12, question="另一问题？")]
    train, val = cleaner.split_records(rows, seed=7, val_ratio=0.3)
    keys = lambda group: {r["conversations"][0]["value"].splitlines()[0] for r in group}
    assert keys(train).isdisjoint(keys(val))
    assert len(train) + len(val) == len(rows)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_invalid_ratio_is_not_silently_clamped(tmp_path, monkeypatch, ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        run_clean(tmp_path, monkeypatch, [record()], ratio=ratio)
    assert not (tmp_path / "output").exists()


def test_conflicting_labels_fail_before_any_output(tmp_path, monkeypatch):
    positive = record()
    negative = deepcopy(positive)
    negative.update(id="negative", kto_tag=False)
    with pytest.raises(ValueError, match="conflict"):
        run_clean(tmp_path, monkeypatch, [positive, negative])
    assert not (tmp_path / "output").exists()


def test_same_id_with_different_content_fails_before_output(tmp_path, monkeypatch):
    first, second = record(), record()
    second["conversations"][-1]["value"] = "different"
    with pytest.raises(ValueError, match="duplicate"):
        run_clean(tmp_path, monkeypatch, [first, second])
    assert not (tmp_path / "output").exists()


def test_missing_id_is_not_silently_accepted(tmp_path, monkeypatch):
    row = record()
    del row["id"]
    with pytest.raises(ValueError, match="id"):
        run_clean(tmp_path, monkeypatch, [row])
    assert not (tmp_path / "output").exists()


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "output"
    target.mkdir()
    sentinel = target / "train.json"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises((ValueError, SystemExit), match="[Oo]utput"):
        run_clean(tmp_path, monkeypatch, [record()], output=target)
    assert sentinel.read_text() == "do not overwrite"


def test_input_directory_cannot_be_used_as_output(tmp_path, monkeypatch):
    with pytest.raises((ValueError, SystemExit), match="[Ii]nput"):
        run_clean(tmp_path, monkeypatch, [record()], output=tmp_path / "input")


@pytest.mark.parametrize("missing", ["train.json", "val.json"])
def test_missing_input_file_is_an_error(tmp_path, monkeypatch, missing):
    source = tmp_path / "input"
    source.mkdir()
    for name in {"train.json", "val.json"} - {missing}:
        (source / name).write_text("[]", encoding="utf-8")
    monkeypatch.setattr(cleaner, "parse_args", lambda: SimpleNamespace(
        input_dir=source, output_dir=tmp_path / "output",
        dataset_name="test", seed=7, val_ratio=0.1,
    ))
    with pytest.raises((FileNotFoundError, ValueError, SystemExit)):
        cleaner.main()
    assert not (tmp_path / "output").exists()


def test_quarantine_and_provenance_preserve_exact_originals(tmp_path, monkeypatch):
    disputed, safe = record(prompt=KEYS[0]), record(1, question="其他问题？")
    output, raw_journal = run_clean(tmp_path, monkeypatch, [disputed, safe])
    kept = json.loads((output / "train.json").read_text()) + json.loads((output / "val.json").read_text())
    assert kept == [safe]
    quarantine = [json.loads(line) for line in (output / "quarantine.jsonl").read_text().splitlines()]
    assert len(quarantine) == 1 and quarantine[0]["record"] == disputed
    assert quarantine[0]["review_required"] is True
    assert (output / "api_verifier_records.jsonl").read_text() == raw_journal
    summary = json.loads((output / "build_summary.json").read_text())
    assert summary["training_ready"] is False and summary["answers_rewritten"] == 0
    assert summary["verifier_judgments_modified"] == 0
    assert summary["records_quarantined"] == 1 and summary["train_val_question_overlap"] == 0
    assert set(summary["input_file_sha256"]) >= {"train.json", "val.json", "api_verifier_records.jsonl"}
