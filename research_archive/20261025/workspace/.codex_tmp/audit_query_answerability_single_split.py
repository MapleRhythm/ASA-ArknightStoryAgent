#!/usr/bin/env python3
"""Audit one query-answerability teacher shard against its frozen input."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument(
        "--expected-prompt-version",
        default="query_answerability_v3.5_positive_caps",
    )
    parser.add_argument("--audit-module-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.audit_module_dir.resolve()))
    from audit_query_answerability_labels import (  # type: ignore[import-not-found]
        audit_split,
        compare_to_expected,
        public_summary,
        scan_expected_input,
    )

    actual = audit_split(
        args.labels,
        args.expected_candidates,
        args.expected_prompt_version,
    )
    expected = scan_expected_input(args.input, args.expected_candidates)
    coverage = compare_to_expected(actual, expected)
    report = {
        "split": public_summary(actual),
        "source_coverage": coverage,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    coverage_failed = any(
        coverage.get(key, 0)
        for key in (
            "missing_query_key_count",
            "extra_query_key_count",
            "changed_candidate_pool_count",
            "input_error_count",
        )
    )
    if actual["error_count"] or coverage_failed:
        raise RuntimeError("single-split label audit failed; see report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
