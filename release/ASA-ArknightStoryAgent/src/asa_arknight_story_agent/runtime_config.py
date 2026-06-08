from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import PROJECT_ROOT


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_config_value(cli_value: Any, config_section: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def resolve_path_value(
    cli_value: Any,
    config_section: dict[str, Any],
    key: str,
    default: Path | str | None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path | None:
    value = cli_value if cli_value is not None else config_section.get(key, default)
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path
