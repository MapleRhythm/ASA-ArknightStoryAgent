#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata
from typing import Any


LEADING_SPEAKER = re.compile(r"^[^，。！？；：:\n]{1,16}[：:]\s*")


def parse_payload(diagnostics: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("normalized_output", "raw_output"):
        text = str(diagnostics.get(key) or "").strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = re.sub(r"\[E\d+\]", "", value)
    value = value.replace("...", "").replace("…", "")
    # Punctuation variants do not change whether a copied passage exists.
    return "".join(char for char in value if not char.isspace() and not unicodedata.category(char).startswith("P"))


def evidence_map(row: dict[str, Any], trace_row: dict[str, Any]) -> dict[str, str]:
    by_source = {str(item.get("id") or ""): item for item in row.get("evidence") or []}
    result: dict[str, str] = {}
    for packing in trace_row.get("evidence_packing") or []:
        if packing.get("status") != "included" or not packing.get("evidence_id"):
            continue
        source = by_source.get(str(packing.get("source_id") or ""))
        if source is None:
            continue
        result[str(packing["evidence_id"])] = str(
            source.get("evidence_chain_text") or source.get("clean_text") or ""
        )
    return result


def quote_candidates(quote: str) -> list[tuple[str, str]]:
    candidates = [("whole", quote)]
    without_speaker = LEADING_SPEAKER.sub("", quote, count=1)
    if without_speaker != quote:
        candidates.append(("without_speaker", without_speaker))
    return candidates


def analyze_reference(
    *,
    evidence_id: str,
    quote: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    for candidate_kind, candidate in quote_candidates(quote):
        needle = normalize(candidate)
        if not needle:
            continue
        matches = [
            current_id
            for current_id, text in mapping.items()
            if needle in normalize(text)
        ]
        if evidence_id in matches:
            return {
                "status": (
                    "cited_match"
                    if candidate_kind == "whole"
                    else "cited_match_after_speaker_strip"
                ),
                "model_evidence_id": evidence_id,
                "resolved_evidence_ids": matches,
                "quote": quote,
            }
        if len(matches) == 1:
            return {
                "status": (
                    "unique_rebind"
                    if candidate_kind == "whole"
                    else "unique_rebind_after_speaker_strip"
                ),
                "model_evidence_id": evidence_id,
                "resolved_evidence_ids": matches,
                "quote": quote,
            }
        if matches:
            return {
                "status": (
                    "ambiguous_rebind"
                    if candidate_kind == "whole"
                    else "ambiguous_rebind_after_speaker_strip"
                ),
                "model_evidence_id": evidence_id,
                "resolved_evidence_ids": matches,
                "quote": quote,
            }
    return {
        "status": "quote_not_visible",
        "model_evidence_id": evidence_id,
        "resolved_evidence_ids": [],
        "quote": quote,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    status_counts: Counter[str] = Counter()
    rejected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        trace = row.get("retrieval_trace") or []
        if not trace:
            continue
        last = trace[-1]
        conclusion = last.get("conclusion") or {}
        diagnostics = conclusion.get("generation_diagnostics") or {}
        if diagnostics.get("grounding_status") != "rejected_to_abstain":
            continue
        payload = parse_payload(diagnostics)
        mapping = evidence_map(row, last)
        facts: list[dict[str, Any]] = []
        if payload:
            for fact_index, fact in enumerate(payload.get("supported_facts") or [], start=1):
                if not isinstance(fact, dict):
                    continue
                references = [
                    analyze_reference(
                        evidence_id=str(reference.get("evidence_id") or ""),
                        quote=str(reference.get("quote") or ""),
                        mapping=mapping,
                    )
                    for reference in fact.get("evidence_refs") or []
                    if isinstance(reference, dict)
                ]
                status_counts.update(reference["status"] for reference in references)
                facts.append(
                    {
                        "index": fact_index,
                        "fact": fact.get("fact"),
                        "references": references,
                        "has_visible_quote": any(
                            reference["status"] != "quote_not_visible"
                            for reference in references
                        ),
                        "all_quotes_visible": bool(references) and all(
                            reference["status"] != "quote_not_visible"
                            for reference in references
                        ),
                    }
                )
        rejected_rows.append(
            {
                "index": index,
                "question": row.get("question"),
                "candidate_answer": (
                    str((payload or {}).get("final_answer") or "").strip()
                    if payload
                    else ""
                ),
                "validator_issues": diagnostics.get("grounding_issues") or [],
                "facts": facts,
                "all_facts_have_visible_quote": bool(facts) and all(
                    fact["has_visible_quote"] for fact in facts
                ),
                "all_model_quotes_visible": bool(facts) and all(
                    fact["all_quotes_visible"] for fact in facts
                ),
            }
        )

    report = {
        "version": "quote_rebinding_audit_v1",
        "boundaries": [
            "This audit checks only whether model-produced quotes occur in visible prompt evidence.",
            "A visible or uniquely rebound quote does not establish semantic entailment.",
            "Speaker-prefix stripping is reported separately and must not silently rewrite attribution.",
        ],
        "rows": len(rows),
        "validator_rejected_rows": len(rejected_rows),
        "all_facts_have_visible_quote_rows": sum(
            row["all_facts_have_visible_quote"] for row in rejected_rows
        ),
        "all_model_quotes_visible_rows": sum(
            row["all_model_quotes_visible"] for row in rejected_rows
        ),
        "reference_statuses": dict(status_counts.most_common()),
        "rejected": rejected_rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
