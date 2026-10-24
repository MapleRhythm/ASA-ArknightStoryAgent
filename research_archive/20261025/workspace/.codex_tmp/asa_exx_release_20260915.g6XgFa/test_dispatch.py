from pathlib import Path
from unittest.mock import patch

import pytest
from exx_official_dispatch import dispatch_if_official


def setup(tmp_path):
    entry = tmp_path / "current-release/scripts/run_exx_official.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("pass")
    return entry


def test_default_uses_official(tmp_path):
    entry = setup(tmp_path)
    with patch("runpy.run_path") as run:
        assert dispatch_if_official(tmp_path, ["问题"])
        assert run.call_args.args[0] == str(entry)


def test_historical_config_not_redirected(tmp_path):
    with patch("runpy.run_path") as run:
        assert not dispatch_if_official(tmp_path, ["--runtime-config", "/tmp/legacy.json"])
        run.assert_not_called()


def test_missing_release_never_silently_uses_old(tmp_path):
    with pytest.raises(RuntimeError, match="refusing silent"):
        dispatch_if_official(tmp_path, ["问题"])
