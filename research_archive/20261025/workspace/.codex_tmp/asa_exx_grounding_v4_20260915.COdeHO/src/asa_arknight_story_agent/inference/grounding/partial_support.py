"""Preserve explicitly incomplete facts with current, revalidated references."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import validate_evidence_id_grounding


def preserve_partial_support(*, conclusion, question, evidence, evidence_prompt_text, retrieval_trace):
    if conclusion.next_action not in {"retrieve_more", "abstain"}:
        return conclusion
    mapping = evidence_id_text_map_from_prompt(evidence_prompt_text)
    by_text = {}
    for evidence_id, text in mapping.items():
        by_text.setdefault(text, evidence_id)
    # Prefer the current candidate. Only use a previous snapshot if its exact
    # supporting texts are still visible. Never carry transient E-IDs forward.
    snapshots = [(None, conclusion.generation_diagnostics.get("grounding_partial_support", []))]
    # A later factual candidate may revise the earlier one. Do not resurrect
    # earlier facts over a new rejected/conflicting candidate.
    if not conclusion.generation_diagnostics.get("grounding_candidate"):
        snapshots.extend(
            (step.get("round"), step.get("conclusion", {}).get("generation_diagnostics", {})
             .get("grounding_partial_support", []))
            for step in reversed(retrieval_trace[:-1])
        )
    for source_round, snapshot in snapshots:
        kept = []
        provenance = []
        for item in snapshot:
            old_texts = item.get("evidence_texts", {})
            if not old_texts or any(text not in by_text for text in old_texts.values()):
                continue
            fact = deepcopy(item["fact"])
            fact["evidence_refs"] = [
                {"evidence_id": by_text[text]} for text in dict.fromkeys(old_texts.values())
            ]
            single = replace(conclusion, supported_facts=[fact], inferred_facts=[])
            issues, _ = validate_evidence_id_grounding(
                conclusion=single, evidence=evidence, question=question,
                evidence_prompt_text=evidence_prompt_text,
            )
            if not issues and fact not in kept:
                kept.append(fact)
                provenance.append({"fact": fact, "source_round": source_round, "revalidated": True})
        if kept:
            conclusion.generation_diagnostics["partial_answer_facts"] = provenance
            # Passing lexical checks is not a semantic verdict. Do not surface
            # a partial claim from a rejected answer: it may invert negation,
            # speaker roles or causality while reusing all source words.
            conclusion.generation_diagnostics["partial_support_visibility"] = "diagnostic_only"
            return conclusion
    return conclusion
