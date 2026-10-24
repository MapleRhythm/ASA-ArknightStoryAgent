"""Exercise the real shard merger only with synthetic temporary datasets."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(os.environ.get(
    "ASA_MERGE_SCRIPT",
    str(Path(__file__).resolve().parents[1] / "scripts/merge_soda_api_verifier_shards.py"),
))
spec = importlib.util.spec_from_file_location("soda_merge_integrity_target", SCRIPT)
merger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger)


def record(index, tag=True, question=None):
    return {
        "id": f"row{index}", "kto_tag": tag, "meta": {"prompt_key": f"state{index}"},
        "conversations": [
            {"from": "human", "value": f"question: {question or '同一个问题？'}\nround: {index}"},
            {"from": "gpt", "value": '{"next_action":"retrieve_more"}'},
        ],
    }


def run_merge(tmp_path, monkeypatch, shards, ratio=0.3):
    inputs = []
    for index, rows in enumerate(shards):
        shard = tmp_path / f"shard{index}"
        shard.mkdir()
        (shard / "train.json").write_text(json.dumps(rows), encoding="utf-8")
        (shard / "val.json").write_text("[]", encoding="utf-8")
        inputs.append(shard)
    args = SimpleNamespace(
        input_dir=inputs, output_dir=tmp_path / "merged", seed=7,
        val_ratio=ratio, dataset_name="synthetic", overwrite=False,
    )
    monkeypatch.setattr(merger, "parse_args", lambda: args)
    merger.main()
    return args.output_dir


def test_merge_regroups_question_across_shards(tmp_path, monkeypatch):
    rows = [record(n) for n in range(12)] + [record(12, question="另一个问题？")]
    output = run_merge(tmp_path, monkeypatch, [rows[::2], rows[1::2]])
    train = json.loads((output / "train.json").read_text())
    val = json.loads((output / "val.json").read_text())
    questions = lambda group: {r["conversations"][0]["value"].splitlines()[0] for r in group}
    assert questions(train).isdisjoint(questions(val))
    assert len(train) + len(val) == len(rows)


def test_merge_fails_before_output_on_cross_shard_label_conflict(tmp_path, monkeypatch):
    positive = record(0)
    negative = deepcopy(positive)
    negative.update(id="negative", kto_tag=False, meta={"prompt_key": "different"})
    with pytest.raises(ValueError, match="conflict"):
        run_merge(tmp_path, monkeypatch, [[positive], [negative]])
    assert not (tmp_path / "merged").exists()


def test_merge_does_not_silently_take_first_conflicting_duplicate_id(tmp_path, monkeypatch):
    positive = record(0)
    negative = deepcopy(positive)
    negative["kto_tag"] = False
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        run_merge(tmp_path, monkeypatch, [[positive], [negative]])
    assert not (tmp_path / "merged").exists()


def test_identical_duplicate_ids_are_deduplicated(tmp_path, monkeypatch):
    positive = record(0)
    output = run_merge(tmp_path, monkeypatch, [[positive], [deepcopy(positive)]])
    rows = json.loads((output / "train.json").read_text()) + json.loads((output / "val.json").read_text())
    assert rows == [positive]


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan")])
def test_merge_does_not_silently_clamp_invalid_split_ratio(tmp_path, monkeypatch, ratio):
    with pytest.raises(ValueError, match="val_ratio"):
        run_merge(tmp_path, monkeypatch, [[record(0)]], ratio=ratio)
    assert not (tmp_path / "merged").exists()


def test_missing_record_id_is_not_silently_lost(tmp_path, monkeypatch):
    invalid = record(0)
    del invalid["id"]
    with pytest.raises(ValueError, match="without an id"):
        run_merge(tmp_path, monkeypatch, [[invalid]])
