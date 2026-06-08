from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.generation.history_rendering import build_unresolved_points
from asa_arknight_story_agent.inference.payload.json_utils import extract_json_object, repair_json_like_output
from asa_arknight_story_agent.inference.payload.normalization import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument, ModelOutputError
from asa_arknight_story_agent.inference.generation.hypothesis_prompt_rendering import (
    build_follow_up_hypothesis_prompt,
    build_hypothesis_prompt,
)
from asa_arknight_story_agent.inference.planning.query_understanding import build_hypothesis


class HypothesisGenerationMixin:
    def build_hypothesis(self, question: str, dialogue_context: str = "") -> HypothesisDocument:
        prompt = build_hypothesis_prompt(question, dialogue_context)
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(256, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.15,
        )
        raw_output = repair_json_like_output(raw_output)
        payload = extract_json_object(raw_output)
        if not payload:
            print(f"[warn] invalid hypothesis json; fallback=heuristic preview={raw_output[:240]}", flush=True)
            return build_hypothesis(question, dialogue_context)
        try:
            return normalize_hypothesis_payload(
                payload,
                question=question,
                dialogue_context=dialogue_context,
            )
        except ModelOutputError as exc:
            print(f"[warn] invalid hypothesis payload; fallback=heuristic error={exc}", flush=True)
            return build_hypothesis(question, dialogue_context)

    def build_follow_up_hypothesis(
        self,
        question: str,
        current_hypothesis: HypothesisDocument,
        evidence: list[dict[str, Any]],
        retrieval_trace: list[dict[str, Any]],
        previous_conclusion: ConclusionResult,
        current_round: int,
    ) -> HypothesisDocument:
        unresolved_points = build_unresolved_points(
            question,
            current_hypothesis,
            evidence,
            retrieval_trace,
            previous_conclusion.missing_slots,
        )
        prompt = build_follow_up_hypothesis_prompt(
            question=question,
            current_hypothesis=current_hypothesis,
            evidence=evidence,
            unresolved_points=unresolved_points,
            retrieval_trace=retrieval_trace,
            previous_conclusion=previous_conclusion,
            current_round=current_round,
            max_retrieval_rounds=self.max_retrieval_rounds,
            prompt_evidence_top_k=self.prompt_evidence_top_k,
            prompt_evidence=self.prepare_prompt_evidence(question, current_hypothesis, evidence),
        )
        raw_output = self.generator.generate(
            prompt,
            max_tokens=min(384, self.generator.max_tokens),
            temperature=0.1,
            top_p=0.8,
            repeat_penalty=1.15,
        )
        raw_output = repair_json_like_output(raw_output)
        payload = extract_json_object(raw_output)
        if not payload:
            print(f"[warn] invalid follow-up hypothesis json; fallback=heuristic preview={raw_output[:240]}", flush=True)
            return build_hypothesis(question, current_hypothesis.dialogue_context)
        try:
            return normalize_hypothesis_payload(
                payload,
                question=question,
                dialogue_context=current_hypothesis.dialogue_context,
                current_intent=current_hypothesis.intent,
            )
        except ModelOutputError as exc:
            print(f"[warn] invalid follow-up hypothesis payload; fallback=heuristic error={exc}", flush=True)
            return build_hypothesis(question, current_hypothesis.dialogue_context)
