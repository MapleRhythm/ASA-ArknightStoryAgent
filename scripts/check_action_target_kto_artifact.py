#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "src/config/llama_factory_action_target_hard_negative_kto_v1_config.yaml"


def parse_simple_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def check_file(path: Path, label: str, errors: list[str]) -> bool:
    if path.exists():
        return True
    errors.append(f"missing {label}: {path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check action-target KTO artifact completeness.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    errors: list[str] = []
    warnings: list[str] = []
    if not config_path.exists():
        raise SystemExit(f"missing config: {config_path}")
    cfg = parse_simple_yaml(config_path)
    output_dir = resolve_path(cfg.get("output_dir", ""))
    base_model = resolve_path(cfg.get("model_name_or_path", ""))
    dataset_dir = resolve_path(cfg.get("dataset_dir", ""))

    check_file(base_model / "config.json", "merged SFT base config", errors)
    check_file(dataset_dir / "dataset_info.json", "dataset_info.json", errors)
    check_file(dataset_dir / "train.json", "train.json", errors)
    check_file(dataset_dir / "val.json", "val.json", errors)

    required_adapter_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]
    for file_name in required_adapter_files:
        check_file(output_dir / file_name, file_name, errors)

    optional_files = [
        "trainer_state.json",
        "train_results.json",
        "eval_results.json",
    ]
    for file_name in optional_files:
        if not (output_dir / file_name).exists():
            warnings.append(f"missing optional {file_name}: {output_dir / file_name}")

    adapter_config: dict[str, Any] = {}
    adapter_config_path = output_dir / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_in_adapter = adapter_config.get("base_model_name_or_path")
        if base_in_adapter and Path(str(base_in_adapter)).name != base_model.name:
            warnings.append(
                "adapter base_model_name_or_path differs from YAML model_name_or_path: "
                f"{base_in_adapter} vs {base_model}"
            )

    summary = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "base_model": str(base_model),
        "dataset_dir": str(dataset_dir),
        "artifact_ready": not errors,
        "errors": errors,
        "warnings": warnings,
        "adapter_config_keys": sorted(adapter_config.keys()) if adapter_config else [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
