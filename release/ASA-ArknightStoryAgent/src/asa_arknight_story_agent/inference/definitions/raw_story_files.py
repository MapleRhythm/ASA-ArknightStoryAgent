from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from asa_arknight_story_agent.config import DATA_ROOT


@lru_cache(maxsize=1)
def raw_story_text_files() -> tuple[Path, ...]:
    story_root = DATA_ROOT / "story"
    if not story_root.exists():
        return ()
    return tuple(sorted(path for path in story_root.rglob("*.txt") if path.is_file()))
