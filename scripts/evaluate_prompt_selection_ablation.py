#!/usr/bin/env python3
"""Compare prompt-evidence selectors on one frozen rerank candidate pool.

Input should be produced by ``build_retrieval_stage_snapshots.py``.  The tool
does not run retrieval or generation and never changes runtime configuration.
It reports evidence-ID coverage for top-score, MMR, query-token coverage and an
oracle upper bound at the requested prompt budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from asa_arknight_story_agent.inference.evidence.prompt_mmr import (  # noqa: E402
    select_prompt_evidence_mmr,
)
from asa_arknight_story_agent.inference.evidence.prompt_ordering import (  # noqa: E402
    select_prompt_evidence,
    select_prompt_evidence_coverage,
)
from asa_arknight_story_agent.inference.evidence.prompt_similarity import (  # noqa: E402
    dedupe_prompt_evidence_candidates,
)
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument  # noqa: E402
from asa_arknight_story_agent.inference.planning.query_understanding import (  # noqa: E402
    build_hypothesis,
)


ABLATION_VERSION = "asa_prompt_selection_ablation_v1"


class AblationError(ValueError):
    """Raised when a snapshot lacks data required by the ablation."""


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            for key in ("rows", "records", "items"):
                if isinstance(value.get(key), list):
                    values = value[key]
                    break
            else:
                values = [value]
        else:
            raise AblationError(f"expected JSON object/array: {path}")
    if not all(isinstance(row, dict) for row in values):
        raise AblationError("all snapshot rows must be JSON objects")
    return [row for row in values if isinstance(row, dict)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def hypothesis_from_snapshot(row: dict[str, Any]) -> HypothesisDocument:
    question = str(row.get("question") or "").strip()
    value = row.get("hypothesis")
    if not isinstance(value, dict):
        return build_hypothesis(question)
    return HypothesisDocument(
        question=str(value.get("question") or question),
        intent=str(value.get("intent") or "plot_reasoning"),
        query_type=str(value.get("query_type") or row.get("query_type") or "reasoning"),
        entities=_strings(value.get("entities")),
        keywords=_strings(value.get("keywords")),
        expected_answer_type=str(value.get("expected_answer_type") or ""),
        dialogue_context=str(value.get("dialogue_context") or ""),
    )


def candidate_from_snapshot(value: dict[str, Any], index: int) -> dict[str, Any]:
    doc_id = str(
        value.get("id")
        or value.get("doc_id")
        or value.get("evidence_id")
        or ""
    ).strip()
    if not doc_id:
        raise AblationError(f"rerank candidate {index} missing id")
    document = {
        "id": doc_id,
        "clean_text": str(value.get("clean_text") or value.get("text") or ""),
        "search_text": str(
            value.get("search_text")
            or value.get("clean_text")
            or value.get("text")
            or ""
        ),
        "activity_name": value.get("activity_name"),
        "story_name": value.get("story_name"),
        "stage_code": value.get("stage_code"),
        "source_path": value.get("source_path"),
    }
    result: dict[str, Any] = {
        "doc_index": value.get("doc_index", index),
        "document": document,
    }
    for key in (
        "score",
        "dense_score",
        "sparse_score",
        "minirag_score",
        "fusion_score",
        "rerank_score",
        "evidence_chain_score",
        "evidence_chain_model_score",
        "evidence_chain_roles",
        "evidence_chain_text",
        "supplemental_source",
    ):
        if value.get(key) is not None:
            result[key] = value[key]
    return result


def rerank_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    stages = row.get("stages")
    if not isinstance(stages, dict) or not isinstance(stages.get("rerank"), list):
        raise AblationError("missing stages.rerank list")
    candidates = [
        candidate_from_snapshot(value, index)
        for index, value in enumerate(stages["rerank"])
        if isinstance(value, dict)
    ]
    if not candidates:
        raise AblationError("stages.rerank must contain candidates")
    return candidates


def gold_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("gold_evidence_ids")
    if not isinstance(value, list):
        raise AblationError("missing gold_evidence_ids list")
    result = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not result:
        raise AblationError("gold_evidence_ids must not be empty")
    return result


def hit_id(item: dict[str, Any]) -> str:
    return str((item.get("document") or {}).get("id") or "").strip()


def oracle_select(
    candidates: list[dict[str, Any]], gold: set[str], *, limit: int
) -> list[dict[str, Any]]:
    """Return the best possible gold coverage after the common dedupe step."""
    deduped = dedupe_prompt_evidence_candidates(candidates)
    selected = [item for item in deduped if hit_id(item) in gold][:limit]
    selected_ids = {hit_id(item) for item in selected}
    for item in deduped:
        if len(selected) >= limit:
            break
        item_id = hit_id(item)
        if item_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item_id)
    return selected


def selection_metrics(gold: set[str], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = [hit_id(item) for item in selected if hit_id(item)]
    hit_count = len(gold.intersection(selected_ids))
    return {
        "selected_ids": selected_ids,
        "selected_count": len(selected_ids),
        "gold_count": len(gold),
        "gold_hit_count": hit_count,
        "recall": hit_count / len(gold) if gold else 0.0,
        "any_hit": int(hit_count > 0),
        "all_gold": int(hit_count == len(gold) and bool(gold)),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "recall": 0.0,
            "any_hit_rate": 0.0,
            "all_gold_rate": 0.0,
            "selected_count_mean": 0.0,
        }
    count = len(records)
    return {
        "count": count,
        "recall": sum(row["recall"] for row in records) / count,
        "any_hit_rate": sum(row["any_hit"] for row in records) / count,
        "all_gold_rate": sum(row["all_gold"] for row in records) / count,
        "selected_count_mean": sum(row["selected_count"] for row in records) / count,
    }


def family_macro(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[row["question_family"]].append(row)
    summaries = [aggregate(rows) for rows in by_family.values()]
    if not summaries:
        return {
            "family_count": 0,
            "recall": 0.0,
            "any_hit_rate": 0.0,
            "all_gold_rate": 0.0,
        }
    count = len(summaries)
    return {
        "family_count": count,
        "recall": sum(item["recall"] for item in summaries) / count,
        "any_hit_rate": sum(item["any_hit_rate"] for item in summaries) / count,
        "all_gold_rate": sum(item["all_gold_rate"] for item in summaries) / count,
    }


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    top_ks: list[int],
    mmr_lambda: float,
) -> dict[str, Any]:
    top_ks = sorted({int(value) for value in top_ks if int(value) > 0})
    if not rows or not top_ks:
        raise AblationError("rows and positive top_ks are required")
    strategy_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id") or index)
        try:
            question = str(row.get("question") or "").strip()
            if not question:
                raise AblationError("missing question")
            hypothesis = hypothesis_from_snapshot(row)
            candidates = rerank_candidates(row)
            gold = set(gold_ids(row))
            family = str(
                row.get("question_family")
                or row.get("family")
                or row.get("family_id")
                or question
            )
        except AblationError as exc:
            failures.append({"id": row_id, "error": str(exc)})
            continue
        for top_k in top_ks:
            selectors: dict[
                str, Callable[[], list[dict[str, Any]]]
            ] = {
                f"top_score@{top_k}": lambda top_k=top_k: select_prompt_evidence(
                    question,
                    hypothesis,
                    candidates,
                    prompt_evidence_top_k=top_k,
                ),
                f"mmr@{top_k}": lambda top_k=top_k: select_prompt_evidence_mmr(
                    candidates,
                    prompt_evidence_top_k=top_k,
                    lambda_mult=mmr_lambda,
                ),
                f"coverage@{top_k}": lambda top_k=top_k: select_prompt_evidence_coverage(
                    question,
                    hypothesis,
                    candidates,
                    prompt_evidence_top_k=top_k,
                ),
                f"oracle@{top_k}": lambda top_k=top_k: oracle_select(
                    candidates, gold, limit=top_k
                ),
            }
            for name, selector in selectors.items():
                metrics = selection_metrics(gold, selector())
                strategy_records[name].append(
                    {
                        "id": row_id,
                        "question": question,
                        "question_family": family,
                        **metrics,
                    }
                )
    if not strategy_records:
        raise AblationError("no evaluable snapshots")
    return {
        "ablation_version": ABLATION_VERSION,
        "rows_total": len(rows),
        "rows_evaluable": max(len(records) for records in strategy_records.values()),
        "rows_failed": len(failures),
        "failure_examples": failures[:20],
        "top_ks": top_ks,
        "mmr_lambda": mmr_lambda,
        "strategies": {
            name: {
                "summary": aggregate(records),
                "family_macro": family_macro(records),
                "rows": records,
            }
            for name, records in sorted(strategy_records.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, action="append", dest="top_ks")
    parser.add_argument("--mmr-lambda", type=float, default=0.72)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_rows(
        read_rows(args.snapshots),
        top_ks=args.top_ks or [10, 16],
        mmr_lambda=min(1.0, max(0.0, args.mmr_lambda)),
    )
    result["snapshots"] = str(args.snapshots.resolve())
    result["snapshots_sha256"] = sha256_file(args.snapshots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "rows_total": result["rows_total"],
        "rows_evaluable": result["rows_evaluable"],
        "rows_failed": result["rows_failed"],
        "strategies": {
            name: {
                "summary": value["summary"],
                "family_macro": value["family_macro"],
            }
            for name, value in result["strategies"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
