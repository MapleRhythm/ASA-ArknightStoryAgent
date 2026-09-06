from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.config import PROJECT_ROOT


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Runtime config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_config_value(cli_value: Any, config_section: dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def resolve_max_retrieval_rounds(
    cli_value: Any,
    inference_section: dict[str, Any],
    *,
    default: int = 2,
    maximum: int = 3,
) -> int:
    """Resolve and bound the retrieval-round budget used by CLI entry points.

    Two rounds are the normal/default path.  A third round is intentionally
    available only as an explicit recovery experiment, while values outside
    the supported range are kept safe and deterministic.
    """

    value = resolve_config_value(cli_value, inference_section, "max_retrieval_rounds", default)
    if value is None:
        value = default
    return min(maximum, max(1, int(value)))


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
