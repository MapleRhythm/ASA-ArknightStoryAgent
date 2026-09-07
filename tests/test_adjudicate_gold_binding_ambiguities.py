from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "adjudicate_gold_binding_ambiguities.py"
)
SPEC = importlib.util.spec_from_file_location("adjudicate_gold_binding_ambiguities", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_messages_include_complete_evidence_without_legacy_1800_char_cutoff() -> None:
    evidence = "甲" * 2200 + "TAIL_MARKER"
    messages = MODULE.messages(
        "主张",
        ["E1"],
        {"E1": evidence},
        {"verdict": "unsupported"},
        {"E1": {"verdict": "unsupported", "reason": "旧判定"}},
        include_prior=False,
    )

    assert "TAIL_MARKER" in messages[-1]["content"]
    assert "原 merged 判定" not in messages[-1]["content"]


def test_truncated_rows_are_selected_for_blind_full_evidence_rejudge() -> None:
    row = {
        "cited_eids": ["E1"],
        "chunk_chars": 600,
        "merged": {"verdict": "unsupported"},
        "individual": {"E1": {"verdict": "unsupported"}},
    }
    fact = {
        "claim": "主张",
        "cited_eids": ["E1"],
        "cited_texts": {"E1": "证据" * 400},
    }

    assert MODULE.adjudication_kind(row, fact, False) is None
    assert MODULE.adjudication_kind(row, fact, True) == "truncated_evidence"


def test_resume_requires_full_evidence_fingerprint_even_for_conflicts() -> None:
    fact = {
        "claim": "主张",
        "cited_eids": ["E1"],
        "cited_texts": {"E1": "完整证据"},
    }
    legacy = {"status": "ok", "label": "supported_by_union", "keep_eids": ["E1"]}
    current = {
        **legacy,
        "full_evidence": True,
        "evidence_sha256": MODULE.evidence_fingerprint(fact),
    }

    assert not MODULE.full_adjudication_matches(fact, legacy)
    assert MODULE.full_adjudication_matches(fact, current)
    assert not MODULE.full_adjudication_matches(
        fact, {**current, "label": "supported_by_union", "keep_eids": ["E9"]}
    )
