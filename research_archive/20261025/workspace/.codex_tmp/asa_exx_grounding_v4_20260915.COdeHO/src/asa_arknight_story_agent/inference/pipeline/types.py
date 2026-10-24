from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ModelOutputError(RuntimeError):
    pass


@dataclass(slots=True)
class HypothesisDocument:
    question: str
    intent: str
    query_type: str
    entities: list[str]
    keywords: list[str]
    expected_answer_type: str
    dialogue_context: str = ""


@dataclass(slots=True)
class InferenceResult:
    question: str
    intent: str
    hypothesis: dict[str, Any]
    model_runtime: dict[str, Any]
    retrieval_query: str
    retrieval_trace: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    answer: str


@dataclass(slots=True)
class ConclusionResult:
    next_action: str
    answer: str
    missing_slots: list[str]
    clarification_question: str
    follow_up_hypothesis: HypothesisDocument | None
    supported_facts: list[dict[str, Any]] = field(default_factory=list)
    inferred_facts: list[dict[str, Any]] = field(default_factory=list)
    grounding_warnings: list[str] = field(default_factory=list)
    generation_diagnostics: dict[str, Any] = field(default_factory=dict)
