from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from asa_arknight_story_agent.inference.evidence.texts import evidence_identity


def query_key(value: str) -> str:
    """Return a stable key for duplicate-query detection.

    This is deliberately limited to whitespace/punctuation normalization.  It
    must not rewrite entities or invent query terms, because those decisions
    belong to the model planner.
    """

    text = re.sub(r"\s+", "", str(value or "")).strip().lower()
    return text


@dataclass(slots=True)
class AdaptiveRoundScheduler:
    """Stateful guard for multi-round retrieval.

    The scheduler does not decide semantic answerability.  It only prevents
    repeated model-generated queries and detects a retrieval round that added
    no new evidence.  In those cases continuing would spend latency without
    increasing the information available to the answer model, so the caller
    should abstain rather than let the model guess.
    """

    issued_query_keys: set[str] = field(default_factory=set)
    evidence_keys: set[str] = field(default_factory=set)
    last_new_evidence_count: int = 0

    def prepare_queries(self, queries: list[str]) -> tuple[list[str], int]:
        fresh: list[str] = []
        duplicate_count = 0
        for raw in queries:
            text = str(raw or "").strip()
            key = query_key(text)
            if not key:
                continue
            if key in self.issued_query_keys or any(query_key(item) == key for item in fresh):
                duplicate_count += 1
                continue
            self.issued_query_keys.add(key)
            fresh.append(text)
        return fresh, duplicate_count

    def observe_evidence(self, evidence: list[dict[str, Any]]) -> int:
        before = len(self.evidence_keys)
        self.evidence_keys.update(evidence_identity(item) for item in evidence)
        self.last_new_evidence_count = len(self.evidence_keys) - before
        return self.last_new_evidence_count

    def can_continue(
        self,
        *,
        round_index: int,
        max_rounds: int,
        pending_queries: list[str],
    ) -> tuple[bool, str]:
        if round_index >= max_rounds:
            return False, "max_rounds_reached"
        if not pending_queries:
            return False, "no_new_queries"
        if round_index == 1 and self.last_new_evidence_count <= 0:
            return True, "new_query_recovery"
        # Allow one recovery round even when the first retrieval was empty:
        # the model may have generated a genuinely different follow-up query.
        # Once a later round also adds no evidence, further looping is only
        # latency and a common path to unsupported guessing.
        if round_index > 1 and self.last_new_evidence_count <= 0:
            return False, "no_new_evidence"
        return True, "new_queries_and_evidence"
