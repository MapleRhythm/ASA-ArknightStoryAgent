from __future__ import annotations

import re
from typing import Any

from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order


def render_retrieval_history(
    retrieval_trace: list[dict[str, Any]],
    *,
    max_rounds: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    if not retrieval_trace:
        return "无"

    blocks: list[str] = []
    selected_steps = retrieval_trace[-max_rounds:] if max_rounds is not None else retrieval_trace
    total_chars = 0
    for step in selected_steps:
        lines = [f"[检索轮次 {step.get('round', '?')}]"]
        planner_action = str(step.get("planner_action") or "initial_retrieval").strip()
        lines.append(f"planner_action: {planner_action}")

        queries = [str(query).strip() for query in step.get("queries") or [] if str(query).strip()]
        if queries:
            lines.append("queries: " + " | ".join(queries[:6]))

        missing_slots = [
            str(slot).strip()
            for slot in step.get("missing_slots") or []
            if str(slot).strip()
        ]
        if missing_slots:
            lines.append("missing_slots: " + " | ".join(missing_slots[:6]))

        clarification_question = str(step.get("clarification_question") or "").strip()
        if clarification_question:
            lines.append(f"clarification_question: {clarification_question}")

        evidence_summary = step.get("evidence_summary") or []
        if evidence_summary:
            evidence_lines = []
            for item in evidence_summary[:3]:
                label = (
                    item.get("stage_code")
                    or item.get("story_name")
                    or item.get("activity_name")
                    or item.get("id")
                    or ""
                )
                snippet = str(item.get("snippet") or "").strip()
                evidence_lines.append(f"{label}: {snippet}")
            lines.append("evidence: " + " | ".join(evidence_lines))

        rendered_block = "\n".join(lines)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def render_generation_history(
    retrieval_trace: list[dict[str, Any]],
    *,
    max_rounds: int | None = None,
    max_total_chars: int | None = None,
) -> str:
    if not retrieval_trace:
        return "无"

    blocks: list[str] = []
    selected_steps = retrieval_trace[-max_rounds:] if max_rounds is not None else retrieval_trace
    total_chars = 0
    for step in selected_steps:
        lines = [f"[生成历史 第{step.get('round', '?')}轮]"]
        hypothesis = step.get("hypothesis") or {}
        if isinstance(hypothesis, dict):
            intent = str(hypothesis.get("intent") or "").strip()
            entities = [str(item).strip() for item in hypothesis.get("entities") or [] if str(item).strip()]
            keywords = [str(item).strip() for item in hypothesis.get("keywords") or [] if str(item).strip()]
            if intent:
                lines.append(f"intent: {intent}")
            if entities:
                lines.append("entities: " + " | ".join(entities[:6]))
            if keywords:
                lines.append("keywords: " + " | ".join(keywords[:8]))

        conclusion = step.get("conclusion") or {}
        if isinstance(conclusion, dict) and conclusion:
            next_action = str(conclusion.get("next_action") or "").strip()
            answer = str(conclusion.get("answer") or "").strip()
            missing_slots = [
                str(item).strip()
                for item in conclusion.get("missing_slots") or []
                if str(item).strip()
            ]
            if next_action:
                lines.append(f"conclusion_action: {next_action}")
            if missing_slots:
                lines.append("conclusion_missing_slots: " + " | ".join(missing_slots[:6]))
            if answer:
                lines.append("conclusion_answer: " + re.sub(r"\s+", " ", answer)[:120])
        rendered_block = "\n".join(lines)
        if max_total_chars is not None and blocks and total_chars + len(rendered_block) > max_total_chars:
            break
        blocks.append(rendered_block)
        total_chars += len(rendered_block)
    return "\n\n".join(blocks)


def build_unresolved_points(
    question: str,
    hypothesis: HypothesisDocument,
    evidence: list[dict[str, Any]],
    retrieval_trace: list[dict[str, Any]],
    previous_missing_slots: list[str],
) -> list[str]:
    del question, hypothesis, evidence, retrieval_trace
    return dedupe_keep_order([item for item in previous_missing_slots if item.strip()])[:8]
