from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import merge_hypotheses
from asa_arknight_story_agent.inference.planning.follow_up_query_generation import build_follow_up_hypothesis_queries
from asa_arknight_story_agent.inference.pipeline.constants import FOLLOW_UP_HYPOTHESIS_TASK_TYPE
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
from asa_arknight_story_agent.inference.planning.query_understanding import (
    build_retrieval_query,
    resolve_referential_question,
)
from asa_arknight_story_agent.inference.common.text_utils import expand_queries_with_main_chapter_terms


def build_initial_queries(question: str, hypothesis: HypothesisDocument) -> list[str]:
    pending_queries = [
        resolve_referential_question(question, hypothesis.entities),
        build_retrieval_query(hypothesis),
    ]
    pending_queries.extend(build_follow_up_hypothesis_queries(question, hypothesis))
    return expand_queries_with_main_chapter_terms(pending_queries)


def prepare_next_round(
    *,
    pipeline: Any,
    question: str,
    current_hypothesis: HypothesisDocument,
    conclusion: ConclusionResult,
    state: PipelineRunState,
    step_record: dict[str, Any],
    round_index: int,
    progress_callback: Callable[[str], None] | None,
) -> tuple[HypothesisDocument, list[str]]:
    if conclusion.follow_up_hypothesis is not None:
        next_hypothesis = merge_hypotheses(current_hypothesis, conclusion.follow_up_hypothesis)
    else:
        if progress_callback:
            progress_callback(FOLLOW_UP_HYPOTHESIS_TASK_TYPE)
        next_hypothesis = pipeline.build_follow_up_hypothesis(
            question,
            current_hypothesis,
            state.evidence,
            state.retrieval_trace,
            conclusion,
            round_index + 1,
        )
    step_record["follow_up_hypothesis_task_type"] = FOLLOW_UP_HYPOTHESIS_TASK_TYPE
    step_record["follow_up_hypothesis"] = asdict(next_hypothesis)

    pending_queries = [build_retrieval_query(next_hypothesis)]
    pending_queries.extend(build_follow_up_hypothesis_queries(question, next_hypothesis))
    step_record["next_round_queries"] = pending_queries
    return next_hypothesis, pending_queries
