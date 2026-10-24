from __future__ import annotations


def normalize_grounding_mode(mode: str) -> str:
    grounding_mode = mode.strip().lower()
    if grounding_mode not in {"weak", "strict", "quote", "grounded", "evidence_id"}:
        return "weak"
    return grounding_mode


def grounding_disabled(mode: str) -> bool:
    return mode.strip().lower() in {"off", "none", "disabled", "false", "0"}
