from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineRunState:
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    web_context_evidence: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    retained_chapter_scope: str | None = None
    retained_storyline_scope: str | None = None
    retained_scope_evidence: list[dict[str, Any]] = field(default_factory=list)
    scope_retention_enabled: bool = False
