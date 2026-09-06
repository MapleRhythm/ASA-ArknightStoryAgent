#!/usr/bin/env python3
"""Summarize v3 Exx grounding audits without mixing protocol failures in."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def load_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError(f"invalid audit payload: {path}")
    return value


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    results = [row for row in payload["results"] if isinstance(row, dict)]
    status = collections.Counter(str(row.get("status") or "unknown") for row in results)
    valid = [row for row in results if row.get("status") == "ok" and isinstance(row.get("judgement"), dict)]
    judgements = [row["judgement"] for row in valid]
    facts = [
        fact
        for judgement in judgements
        for fact in judgement.get("facts", [])
        if isinstance(fact, dict)
    ]
    support = collections.Counter(str(fact.get("support") or "unknown") for fact in facts)
    relevance = collections.Counter(
        str(fact.get("question_relevance") or "unknown") for fact in facts
    )
    return {
        "rows": len(results),
        "status": dict(sorted(status.items())),
        "semantic_denominator": len(valid),
        "set_support": dict(
            sorted(
                collections.Counter(
                    str(judgement.get("set_support") or "unknown")
                    for judgement in judgements
                ).items()
            )
        ),
        "fact_count": len(facts),
        "fact_support": dict(sorted(support.items())),
        "question_relevance": dict(sorted(relevance.items())),
        "citation_complete": sum(bool(fact.get("citation_complete")) for fact in facts),
        "critical_unsupported_claims": sum(
            int(judgement.get("critical_unsupported_claims") or 0)
            for judgement in judgements
        ),
        "context_sufficiency": dict(
            sorted(
                collections.Counter(
                    str(judgement.get("context_sufficiency") or "unknown")
                    for judgement in judgements
                ).items()
            )
        ),
        "action_appropriateness": dict(
            sorted(
                collections.Counter(
                    str(judgement.get("action_appropriateness") or "unknown")
                    for judgement in judgements
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audits", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        str(path): summarize(load_audit(path))
        for path in args.audits
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
