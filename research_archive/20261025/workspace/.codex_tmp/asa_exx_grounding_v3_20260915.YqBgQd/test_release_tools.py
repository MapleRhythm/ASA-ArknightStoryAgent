import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_root(tmp_path, monkeypatch):
    promote = load("promote_fix")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs").mkdir()
    (root / "README.md").write_text("original readme\n")
    (root / "docs/OFFICIAL_EXX_RELEASE_20260915.md").write_text("original doc\n")
    (root / "current-release").symlink_to(promote.OLD)
    (root / "configs").mkdir()
    new = root / promote.NEW
    (new / "configs").mkdir(parents=True)
    (new / "configs/runtime_exx_official.json").write_text("{}")
    for name in ("runtime_inference.json", "runtime_inference_gpu.json", "runtime_exx_official.json"):
        (root / "configs" / name).write_text("{}")
    results = root / "outputs/experiments/exx_grounding_fix_smoke8_20260915_v3/results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text("{}\n")
    review = root / "review.json"
    review.write_text(json.dumps({
        "release": Path(promote.NEW).name, "reviewed_qids": [6, 8, 10, 11, 13, 16, 18, 19],
        "resolved_target_qids": [8, 13, 18], "new_major_errors": 0,
        "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(),
    }))
    monkeypatch.setattr(promote, "ROOT", root)
    monkeypatch.setattr(promote, "BACKUP", root / "outputs/release_backups/exx-20260915-v3")
    monkeypatch.setattr(promote.subprocess, "run", lambda *a, **kw: None)
    return promote, root, review


def test_promotion_dry_run_changes_nothing(tmp_path, monkeypatch):
    promote, root, review = fixture_root(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["promote", "--review", str(review)])
    promote.main()
    assert (root / "current-release").readlink() == Path(promote.OLD)
    assert not promote.BACKUP.exists()


def test_promotion_and_rollback_restore_exact_preimages(tmp_path, monkeypatch):
    promote, root, review = fixture_root(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["promote", "--review", str(review), "--apply"])
    promote.main()
    assert (root / "current-release").readlink() == Path(promote.NEW)
    rollback = load("rollback_exx_20260915_v3")
    monkeypatch.setattr(rollback, "ROOT", root)
    monkeypatch.setattr(rollback, "BACKUP", promote.BACKUP)
    rollback.main()
    assert (root / "current-release").readlink() == Path(promote.OLD)
    assert (root / "README.md").read_text() == "original readme\n"
    assert (root / "docs/OFFICIAL_EXX_RELEASE_20260915.md").read_text() == "original doc\n"
    assert (root / promote.NEW).is_dir()


def test_later_user_edit_blocks_rollback_before_any_restore(tmp_path, monkeypatch):
    promote, root, review = fixture_root(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["promote", "--review", str(review), "--apply"])
    promote.main()
    (root / "README.md").write_text("later user edit")
    published_doc = (root / "docs/OFFICIAL_EXX_RELEASE_20260915.md").read_bytes()
    rollback = load("rollback_exx_20260915_v3")
    monkeypatch.setattr(rollback, "ROOT", root)
    monkeypatch.setattr(rollback, "BACKUP", promote.BACKUP)
    with pytest.raises(AssertionError, match="Later edits"):
        rollback.main()
    assert (root / "current-release").readlink() == Path(promote.NEW)
    assert (root / "docs/OFFICIAL_EXX_RELEASE_20260915.md").read_bytes() == published_doc
