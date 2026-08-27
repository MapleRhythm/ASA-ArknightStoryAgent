from __future__ import annotations

from typing import Any

from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import (
    claim_has_unsupported_quote_required_terms,
    grounded_quote_texts,
)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult
from asa_arknight_story_agent.inference.grounding.quote_match_utils import (
    GROUNDING_LONG_TOKEN_MIN_LEN,
    grounding_evidence_pool,
    grounding_extract_answer_tokens,
    normalize_for_evidence_match,
)
from asa_arknight_story_agent.inference.common.text_utils import dedupe_keep_order
from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map,
    evidence_id_text_map_from_prompt,
)


def validate_grounded_quotes(
    *,
    conclusion: ConclusionResult,
    evidence: list[dict[str, Any]],
    question: str,
    evidence_prompt_text: str | None = None,
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    # evidence_id -> 该编号对应的证据文本(与渲染进 prompt 的 [E编号] 一致)
    eid_map = evidence_id_text_map_from_prompt(evidence_prompt_text) or evidence_id_text_map(evidence)
    # 证据全池(用于 evidence_id 缺失时的兜底 quote 核对)
    evidence_pool = normalize_for_evidence_match(evidence_prompt_text or grounding_evidence_pool(evidence))
    if not conclusion.supported_facts:
        issues.append("missing_supported_facts")
        return issues, warnings
    if len(conclusion.supported_facts) > 6:
        issues.append(f"too_many_supported_facts:{len(conclusion.supported_facts)}>6")

    quote_count = 0
    total_quote_chars = 0
    for fact_index, fact in enumerate(conclusion.supported_facts, start=1):
        if not isinstance(fact, dict):
            issues.append(f"supported_fact_{fact_index}_not_object")
            continue
        refs = fact.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            issues.append(f"supported_fact_{fact_index}_missing_evidence_refs")
            continue
        if len(refs) > 2:
            issues.append(f"supported_fact_{fact_index}_too_many_quotes:{len(refs)}>2")
        fact_quote_chars = 0
        for ref_index, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_not_object")
                continue
            # 第1道: evidence_id 必须存在于渲染进 prompt 的 [E编号] 集合(防编造编号)
            eid = str(ref.get("evidence_id") or "").strip()
            if not eid:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_missing_evidence_id")
            elif eid_map and eid not in eid_map:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_evidence_id_not_found:{eid}")
            quote = str(ref.get("quote") or "").strip()
            if not quote:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_missing_quote")
                continue
            quote_count += 1
            fact_quote_chars += len(quote)
            total_quote_chars += len(quote)
            if len(quote) > 80:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_quote_over_80")
            # 第2道: quote 宽松核对——优先核对到 evidence_id 对应的那条证据; 缺 evidence_id 时兜底核对全池
            if eid and eid in eid_map:
                target_pool = normalize_for_evidence_match(eid_map[eid])
            else:
                target_pool = evidence_pool
            if normalize_for_evidence_match(quote) not in target_pool:
                issues.append(f"supported_fact_{fact_index}_ref_{ref_index}_quote_not_found")
        if fact_quote_chars > 160:
            issues.append(f"supported_fact_{fact_index}_quote_total_over_160")
    if quote_count == 0:
        issues.append("missing_quotes")
    if total_quote_chars > 400:
        issues.append(f"answer_quote_total_over_400:{total_quote_chars}")

    quote_pool = normalize_for_evidence_match("\n".join(grounded_quote_texts(conclusion)))
    answer_tokens = grounding_extract_answer_tokens(conclusion.answer, question)
    missing_tokens = [
        token
        for token in answer_tokens
        if len(token) >= GROUNDING_LONG_TOKEN_MIN_LEN and normalize_for_evidence_match(token) not in quote_pool
    ]
    unsupported_relations = claim_has_unsupported_quote_required_terms(conclusion.answer, quote_pool)
    for fact_index, fact in enumerate(conclusion.supported_facts, start=1):
        if isinstance(fact, dict):
            unsupported_fact_terms = claim_has_unsupported_quote_required_terms(str(fact.get("fact") or ""), quote_pool)
            if unsupported_fact_terms:
                issues.append(
                    f"supported_fact_{fact_index}_has_terms_outside_quotes:"
                    + ",".join(unsupported_fact_terms[:8])
                )
    if missing_tokens or unsupported_relations:
        warnings.append(
            "final_answer_has_terms_outside_supported_facts:"
            + ",".join(dedupe_keep_order([*missing_tokens[:8], *unsupported_relations]))
        )
    return issues, warnings
