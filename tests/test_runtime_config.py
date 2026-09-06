from pathlib import Path

import pytest

from asa_arknight_story_agent.runtime_config import load_runtime_config


def test_missing_runtime_config_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Runtime config does not exist"):
        load_runtime_config(tmp_path / "missing.json")
