import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_exx_clean_binding_sft.py"
SPEC = importlib.util.spec_from_file_location("clean_materializer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_invalid_input_does_not_leave_training_files_or_registry(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "gold_recalibrated_v1_strict_train.jsonl").write_text(
        json.dumps({"id": "bad", "task_type": "grounded_action_generation", "system": "wrong"}) + "\n"
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["script", "--strict-dir", str(source), "--out-dir", str(out)])
    assert MODULE.main() == 1
    assert sorted(p.name for p in out.iterdir()) == ["build_report.json"]
    report = json.loads((out / "build_report.json").read_text())
    assert report["validation_error_count"] == 1
    assert report["training_ready"] is False


def test_structurally_valid_legacy_export_remains_marked_unverified(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    row = {
        "id": "ok", "task_type": "grounded_action_generation", "system": MODULE.SYSTEM,
        "conversations": [
            {"from": "human", "value": f"question: 问题？\nevidence:\n[E1]\n证据\noutput_schema: {MODULE.PROTOCOL}\n"},
            {"from": "gpt", "value": '{"next_action":"abstain","reason":"证据不足"}'},
        ],
    }
    (source / "gold_recalibrated_v1_strict_train.jsonl").write_text(json.dumps(row) + "\n")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["script", "--strict-dir", str(source), "--out-dir", str(out)])
    assert MODULE.main() == 0
    assert (out / "dataset_info.json").exists()
    report = json.loads((out / "build_report.json").read_text())
    assert report["validation_error_count"] == 0 and not report["training_ready"]
