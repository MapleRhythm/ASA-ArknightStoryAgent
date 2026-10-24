#!/usr/bin/env python3
"""Offline selector/renderer audit, not an answer-accuracy evaluation.

Replays frozen rerank pools through the actual minimal conclusion renderer.
Separates outer IDs, chain-member metadata and text present in final E-blocks.
Gold IDs/review annotations are used ONLY after selection and rendering.
No retrieval, generation, training export or runtime mutation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_prompt_selection_ablation import (  # noqa: E402
    candidate_from_snapshot,
    hit_id,
    hypothesis_from_snapshot,
    read_rows,
    rerank_candidates,
    sha256_file,
    visible_ids,
)
from asa_arknight_story_agent.inference.evidence.prompt_ordering import (  # noqa: E402
    select_prompt_evidence,
)
from asa_arknight_story_agent.inference.evidence.rendering import (  # noqa: E402
    evidence_id_text_map_from_prompt,
    render_short_evidence_brief,
)
from asa_arknight_story_agent.inference.evidence.texts import (  # noqa: E402
    best_prompt_text,
    document_clean_text,
    evidence_identity,
    prefer_direct_prompt_text,
    prompt_evidence_score,
)
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import (  # noqa: E402
    build_minimal_conclusion_prompt,
)


VERSION = "asa_rendered_evidence_visibility_v2"
STRATEGIES = ("production", "aligned_similarity", "identity_only", "exact_containment")


def normalize(text: str) -> str:
    """Match runtime's whitespace folding, preserving punctuation and negation."""
    return re.sub(r"\s+", " ", text).strip()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_items(
    row: dict[str, Any], candidates: list[dict[str, Any]], *,
    mode: str, strategy: str, top_k: int,
) -> list[dict[str, Any]]:
    # This function must never inspect gold, reviewer notes or answer facts.
    inputs = candidates
    if strategy != "production" and mode == "evidence_id":
        inputs = [prefer_direct_prompt_text(item) for item in inputs]
    if strategy in ("production", "aligned_similarity"):
        return select_prompt_evidence(
            str(row["question"]), hypothesis_from_snapshot(row), inputs,
            prompt_evidence_top_k=top_k,
        )
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    kept: list[dict[str, Any]] = []
    identities: set[str] = set()
    texts: list[str] = []
    for item in inputs:
        identity = evidence_identity(item)
        text = normalize(best_prompt_text(
            item, prefer_direct=bool(item.get("prompt_prefer_clean_text"))
        ))
        if identity in identities:
            continue
        if strategy == "exact_containment" and text and any(text in old for old in texts):
            continue
        identities.add(identity)
        texts.append(text)
        kept.append(item)
    return sorted(kept, key=prompt_evidence_score, reverse=True)[:top_k]


def render_single(item: dict[str, Any], *, mode: str, per_doc: int) -> str:
    direct = mode == "evidence_id"
    return render_short_evidence_brief(
        [prefer_direct_prompt_text(item) if direct else item],
        max_chars_per_doc=per_doc, max_total_chars=0,
        preserve_complete_evidence=direct or mode == "quote", label_on_own_line=direct,
    )


def packing_trace(
    selected: list[dict[str, Any]], *, mode: str, per_doc: int,
    total: int, packing: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Trace the exact short-renderer accounting; assert against it downstream.

    Quote now preserves complete blocks and counts separator newlines. E-ID
    retains its historical budget accounting. Quote's runtime policy is skip;
    E-ID's is break. The other policy is an offline counterfactual.
    """
    if packing not in ("break", "skip"):
        raise ValueError(f"unknown packing: {packing}")
    included: list[dict[str, Any]] = []
    lines: list[str] = []
    trace: list[dict[str, Any]] = []
    seen: set[str] = set()
    used = 0
    stopped = False
    for item in selected:
        one = render_single(item, mode=mode, per_doc=per_doc)
        doc = item.get("document") or {}
        key = str(doc.get("id") or item.get("doc_index") or "").strip()
        if not key and one:
            key = next(iter(evidence_id_text_map_from_prompt(one).values()))[:160]
        entry: dict[str, Any] = {
            "id": hit_id(item), "counted_chars_before": used,
            "remaining_counted_chars": max(0, total - used) if total > 0 else None,
        }
        if not one:
            entry["status"] = "empty"
        elif key in seen:
            entry["status"] = "duplicate_id"
        else:
            seen.add(key)
            line = one.replace("[E1]", f"[E{len(lines) + 1}]", 1)
            cost = len(line) + (1 if lines and mode == "quote" else 0)
            fits = total <= 0 or used + cost <= total
            entry.update({"block_chars": len(line), "counted_cost": cost, "fits_remaining": fits})
            if stopped:
                entry["status"] = "after_budget_break"
            elif not fits:
                entry["status"] = "budget_break" if packing == "break" else "budget_skip"
                stopped = packing == "break"
            else:
                entry["status"] = "included"
                entry["evidence_id"] = f"E{len(lines) + 1}"
                included.append(item)
                lines.append(line)
                used += cost
        trace.append(entry)
    return included, trace, "\n".join(lines)


def source_texts(row: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for values in row["stages"].values():
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            item = candidate_from_snapshot(value, index)
            doc_id = hit_id(item)
            text = normalize(document_clean_text(item))
            if not text:
                continue
            if doc_id in output and output[doc_id] != text:
                raise ValueError(f"conflicting source text for {doc_id}")
            output[doc_id] = text
    return output


def coverage(ids: list[str], present: set[str]) -> dict[str, Any]:
    required = set(ids)
    if not required:
        raise ValueError("coverage denominator cannot be empty")
    found = sorted(required & present)
    return {
        "hits": len(found), "total": len(required), "recall": len(found) / len(required),
        "all": required <= present, "missing": sorted(required - present),
    }


def audit_case(
    row: dict[str, Any], review: dict[str, Any], *, mode: str, strategy: str,
    packing: str, top_k: int, per_doc: int, total: int, rendering: str = "runtime",
) -> dict[str, Any]:
    candidates = rerank_candidates(row)
    selected = select_items(row, candidates, mode=mode, strategy=strategy, top_k=top_k)
    if rendering not in ("runtime", "direct_only"):
        raise ValueError(f"unknown rendering policy: {rendering}")
    # A separate ablation: preserve selector, scores, rank and budgets, changing
    # only the text passed to the renderer. Do not dedupe a second time here.
    render_items = (
        [prefer_direct_prompt_text(item) for item in selected]
        if rendering == "direct_only" else selected
    )
    packed, trace, expected = packing_trace(
        render_items, mode=mode, per_doc=per_doc, total=total, packing=packing
    )
    runtime_packing = "skip" if mode == "quote" else "break"
    uses_runtime_packing = packing == runtime_packing
    prompt, brief = build_minimal_conclusion_prompt(
        question=str(row["question"]), current_hypothesis=hypothesis_from_snapshot(row),
        evidence=candidates, current_round=1, max_retrieval_rounds=2,
        prompt_evidence_top_k=top_k,
        # Exercise real runtime packing on the unfiltered selected pool.
        # Only counterfactual policies need prepacking before the renderer.
        prompt_evidence=render_items if uses_runtime_packing else packed,
        evidence_max_chars_per_doc=per_doc, evidence_max_total_chars=total,
        grounding_mode=mode,
    )
    if brief != expected or (brief and brief not in prompt):
        raise ValueError("packing trace does not match actual runtime prompt")
    blocks = evidence_id_text_map_from_prompt(brief)
    if len(blocks) != len(packed):
        raise ValueError("unexpected E-block syntax in source text")
    old_gold = list(row["gold_evidence_ids"])
    minimal = list(review["minimal_sufficient_evidence_ids"])
    facts = review["answer_facts"]
    groups = [fact["support_groups"] for fact in facts]
    required_ids = set(old_gold + minimal)
    required_ids.update(doc_id for alternatives in groups for group in alternatives for doc_id in group)
    if not groups or any(not alternatives or any(not group for group in alternatives) for alternatives in groups):
        raise ValueError("review must contain nonempty support groups")
    sources = source_texts(row)
    missing_sources = required_ids - sources.keys()
    if missing_sources:
        raise ValueError(f"missing source text for reviewed/gold IDs: {sorted(missing_sources)}")
    visible: set[str] = set()
    details = []
    for doc_id in sorted(required_ids):
        text = sources[doc_id]
        matches = [eid for eid, body in blocks.items() if text in normalize(body)]
        if matches:
            visible.add(doc_id)
        details.append({
            "id": doc_id, "source_chars": len(text),
            "source_sha256": text_sha(text), "full_text_visible_in": matches,
        })
    fact_group_covered = [
        any(set(group) <= visible for group in alternatives) for alternatives in groups
    ]
    outer = {hit_id(item) for item in selected}
    metadata = visible_ids(selected)
    def stages(ids: list[str]) -> dict[str, Any]:
        return {
            "outer_ids": coverage(ids, outer),
            "chain_metadata_ids": coverage(ids, metadata),
            "rendered_full_text": coverage(ids, visible),
        }
    return {
        "id": row["id"], "question": row["question"], "rendering": rendering,
        "runtime_packing_verified": uses_runtime_packing,
        "selected_ids": [hit_id(item) for item in selected],
        "rendered_ids": [hit_id(item) for item in packed],
        "selected_matches_snapshot_prompt": [
            hit_id(item) for item in selected
        ] == [item["id"] for item in row["stages"].get("prompt", [])],
        "block_count": len(blocks), "evidence_chars": len(brief),
        "counted_chars": sum(
            entry.get("counted_cost", 0) for entry in trace if entry["status"] == "included"
        ),
        "prompt_sha256": text_sha(prompt), "evidence_sha256": text_sha(brief),
        "blocks": [
            {"evidence_id": eid, "source_id": hit_id(item),
             "body_chars": len(body), "body_sha256": text_sha(body)}
            for (eid, body), item in zip(blocks.items(), packed)
        ],
        "packing_trace": trace,
        "legacy_gold": stages(old_gold), "reviewed_minimal": stages(minimal),
        "reviewed_fact_group_full_text_coverage": fact_group_covered,
        "all_reviewed_fact_groups_full_text_visible": all(fact_group_covered),
        "reference_text_visibility": details,
        "evidence_brief": brief,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"questions": len(rows)}
    for basis in ("legacy_gold", "reviewed_minimal"):
        result[basis] = {}
        for stage in ("outer_ids", "chain_metadata_ids", "rendered_full_text"):
            values = [row[basis][stage] for row in rows]
            result[basis][stage] = {
                "all_questions": sum(value["all"] for value in values),
                "question_macro_recall": sum(value["recall"] for value in values) / len(values),
                "hits": sum(value["hits"] for value in values),
                "total": sum(value["total"] for value in values),
            }
    result.update({
        "all_reviewed_fact_groups_full_text_visible": sum(
            row["all_reviewed_fact_groups_full_text_visible"] for row in rows
        ),
        "mean_blocks": sum(row["block_count"] for row in rows) / len(rows),
        "mean_evidence_chars": sum(row["evidence_chars"] for row in rows) / len(rows),
        "questions_with_budget_break": sum(any(
            entry["status"] == "budget_break" for entry in row["packing_trace"]
        ) for row in rows),
        "questions_with_budget_skip": sum(any(
            entry["status"] == "budget_skip" for entry in row["packing_trace"]
        ) for row in rows),
        "later_blocks_that_fit_after_break": sum(
            entry["status"] == "after_budget_break" and entry["fits_remaining"]
            for row in rows for entry in row["packing_trace"]
        ),
    })
    return result


def profile_from_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    generation = config["inference"]
    if generation["conclusion_prompt_mode"] != "minimal":
        raise ValueError("this audit only supports minimal prompts")
    if generation["answer_grounding_mode"] not in ("quote", "evidence_id"):
        raise ValueError("unsupported grounding mode")
    for flag in ("enable_mmr", "enable_set_coverage_selection", "enable_pyramid_order",
                 "enable_evidence_pinning", "enable_crag_refinement"):
        if generation.get(flag):
            raise ValueError(f"unsupported selector option: {flag}")
    if generation.get("web_context", {}).get("force_prompt_evidence"):
        raise ValueError("forced web evidence is unsupported")
    return {
        "name": path.stem, "path": str(path), "sha256": sha256_file(path),
        "mode": generation["answer_grounding_mode"],
        "top_k": generation["prompt_evidence_top_k"],
        "per_doc": generation["prompt_evidence_max_chars_per_doc"],
        "total": generation["prompt_conclusion_evidence_max_total_chars"],
        "runtime_packing": "skip" if generation["answer_grounding_mode"] == "quote" else "break",
        "complete_quote_blocks": generation["answer_grounding_mode"] == "quote",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--reviewed", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshots, reviews = read_rows(args.snapshots), read_rows(args.reviewed)
    by_id = {row["id"]: row for row in reviews}
    snapshot_ids = {row["id"] for row in snapshots}
    if (len(by_id) != len(reviews) or len(snapshot_ids) != len(snapshots)
            or snapshot_ids != by_id.keys() or not snapshots):
        raise ValueError("require unique, matching snapshot/review IDs")
    if any(row.get("review_status") != "approved" for row in reviews):
        raise ValueError("review annotations must be explicitly approved")
    profiles = [profile_from_config(path) for path in args.config]
    variants = {}
    for profile in profiles:
        for strategy in STRATEGIES:
            for packing in ("break", "skip"):
                rows = [audit_case(
                    row, by_id[row["id"]], mode=profile["mode"], strategy=strategy,
                    packing=packing, top_k=profile["top_k"],
                    per_doc=profile["per_doc"], total=profile["total"],
                ) for row in snapshots]
                key = f'{profile["name"]}/{strategy}/{packing}'
                variants[key] = {"summary": summarize(rows), "rows": rows}
        if profile["mode"] == "quote":
            rows = [audit_case(
                row, by_id[row["id"]], mode="quote", strategy="production",
                packing=profile["runtime_packing"], top_k=profile["top_k"],
                per_doc=profile["per_doc"], total=profile["total"],
                rendering="direct_only",
            ) for row in snapshots]
            key = f'{profile["name"]}/production/{profile["runtime_packing"]}/direct_only'
            variants[key] = {"summary": summarize(rows), "rows": rows}
    report = {
        "version": VERSION, "evaluation_only": True, "training_ready": False,
        "limitations": [
            "Frozen development questions, not a new blind test or production E2E.",
            "Full-passage visibility is a lexical proxy, not answer accuracy or semantic sufficiency.",
            "A missing full passage may still leave all necessary answer spans visible.",
            "Review annotations are prior Codex review, not independent human gold.",
            "Chain-member IDs are metadata; only final rendered E-block text is visible.",
            "No gold, answer or reviewer note is used for selection or packing.",
            "Scores and rerank pools are frozen snapshots, not fresh deployed retrieval.",
            "Quote runtime now uses complete-block skip packing; E-ID retains break packing.",
            "Other packing policies and non-production selectors are offline counterfactuals only.",
            "Quote direct_only rendering is an offline counterfactual; output protocol is unchanged.",
        ],
        "snapshots": {"path": str(args.snapshots), "sha256": sha256_file(args.snapshots)},
        "reviewed": {"path": str(args.reviewed), "sha256": sha256_file(args.reviewed)},
        "profiles": profiles,
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                Path(__file__).resolve(), ROOT / "scripts/evaluate_prompt_selection_ablation.py",
                *sorted((ROOT / "src/asa_arknight_story_agent/inference/evidence").glob("*.py")),
                ROOT / "src/asa_arknight_story_agent/inference/generation/minimal_conclusion_prompt.py",
                ROOT / "src/asa_arknight_story_agent/inference/generation/exx_prompt.py",
                ROOT / "src/asa_arknight_story_agent/inference/generation/chat_prompt_format.py",
                ROOT / "src/asa_arknight_story_agent/inference/common/text_utils.py",
            ]
        },
        "variants": variants,
    }
    if args.output.exists():
        raise ValueError("refusing to overwrite a frozen visibility audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: value["summary"] for key, value in variants.items()
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
