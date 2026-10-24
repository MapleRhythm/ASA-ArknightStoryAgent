#!/usr/bin/env python3
"""Stream-audit query-answerability label shards before pair export."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


POSITIVE_LABELS = {"direct_answer", "partial_answer", "supporting_context"}
VALID_LABELS = POSITIVE_LABELS | {"hard_context", "irrelevant"}
LABEL_TIER = {
    "direct_answer": 0,
    "partial_answer": 1,
    "supporting_context": 2,
    "hard_context": 3,
    "irrelevant": 4,
}
MAX_LABEL_COUNTS = {
    "direct_answer": 12,
    "partial_answer": 20,
    "supporting_context": 16,
    "hard_context": 16,
}
EXPECTED_PROTOCOL = "query_answerability_v3"
EXPECTED_PROMPT_VERSION = "query_answerability_v3.5_positive_caps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--val", type=Path, action="append", required=True)
    parser.add_argument(
        "--train-input",
        type=Path,
        help="Frozen query-only train input used to require exact key/pool coverage.",
    )
    parser.add_argument(
        "--val-input",
        type=Path,
        help="Frozen query-only validation input used to require exact key/pool coverage.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--expected-prompt-version", default=EXPECTED_PROMPT_VERSION)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def candidate_pool_fingerprint(candidates: list[dict[str, Any]]) -> str:
    stable = sorted(
        (
            str(item.get("candidate_id") or ""),
            str(item.get("kind") or ""),
            str(item.get("text") or ""),
        )
        for item in candidates
    )
    payload = json.dumps(stable, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scan_expected_input(
    path: Path, expected_candidates: int
) -> dict[str, Any]:
    keys: set[tuple[str, str]] = set()
    fingerprints: dict[tuple[str, str], str] = {}
    families: set[str] = set()
    errors: list[str] = []
    for line_number, row in iter_jsonl(path):
        location = f"{path}:{line_number}"
        key = (str(row.get("source_name") or ""), str(row.get("query") or ""))
        candidates = row.get("candidates") or []
        if not key[0] or not key[1]:
            errors.append(location + " empty source_name/query")
        if key in keys:
            errors.append(location + " duplicate expected query key")
        keys.add(key)
        family = str(row.get("story_family") or "").strip()
        if not family:
            errors.append(location + " empty story_family")
        else:
            families.add(family)
        if not isinstance(candidates, list) or len(candidates) != expected_candidates:
            errors.append(location + f" candidate_count={len(candidates)}")
            continue
        if any(not isinstance(item, dict) for item in candidates):
            errors.append(location + " candidates must be objects")
            continue
        fingerprints[key] = candidate_pool_fingerprint(candidates)
    return {
        "path": str(path),
        "query_keys": keys,
        "pool_fingerprints": fingerprints,
        "story_families": families,
        "error_count": len(errors),
        "errors": errors[:100],
    }


def audit_split(
    paths: list[Path],
    expected_candidates: int,
    expected_prompt_version: str,
    *,
    max_trainable_positives: int = 5,
) -> dict[str, Any]:
    keys: set[tuple[str, str]] = set()
    families: set[str] = set()
    counts: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    duplicate_keys: Counter[str] = Counter()
    errors: list[str] = []
    usable_keys: set[tuple[str, str]] = set()
    provider_for_key: dict[tuple[str, str], str] = {}
    pool_fingerprint_for_key: dict[tuple[str, str], str] = {}

    for path in paths:
        for line_number, row in iter_jsonl(path):
            location = f"{path}:{line_number}"
            key = (str(row.get("source_name") or ""), str(row.get("query") or ""))
            provider = str(row.get("teacher_provider") or path.parent.name)
            if not key[0] or not key[1]:
                errors.append(location + " empty source_name/query")
            if not str(row.get("story_family") or "").strip():
                errors.append(location + " empty story_family")
            if key in keys:
                duplicate_keys[f"{provider_for_key[key]}->{provider}"] += 1
            else:
                provider_for_key[key] = provider
            keys.add(key)
            family = str(row.get("story_family") or "")
            if family:
                families.add(family)
            providers[provider] += 1
            counts["rows"] += 1

            candidates = row.get("candidates") or []
            teacher = row.get("teacher")
            if not isinstance(teacher, dict):
                errors.append(location + " teacher must be an object")
                continue
            ranking = teacher.get("ranking") or []
            if row.get("teacher_protocol") != EXPECTED_PROTOCOL:
                errors.append(location + " unexpected protocol")
            if row.get("teacher_prompt_version") != expected_prompt_version:
                errors.append(location + " unexpected prompt version")
            if len(candidates) != expected_candidates:
                errors.append(location + f" candidate_count={len(candidates)}")
            if not isinstance(candidates, list) or any(
                not isinstance(item, dict) for item in candidates
            ):
                errors.append(location + " candidates must be a list of objects")
                continue
            if not isinstance(ranking, list) or any(
                not isinstance(item, dict) for item in ranking
            ):
                errors.append(location + " ranking must be a list of objects")
                continue
            candidate_id_list = [str(item.get("candidate_id") or "") for item in candidates]
            if not all(candidate_id_list) or len(candidate_id_list) != len(set(candidate_id_list)):
                errors.append(location + " empty/duplicate candidate IDs")
            non_document = sum(
                str(item.get("kind") or "document") != "document" for item in candidates
            )
            if non_document:
                errors.append(location + f" non_document_candidates={non_document}")
            fingerprint = candidate_pool_fingerprint(candidates)
            if key in pool_fingerprint_for_key and pool_fingerprint_for_key[key] != fingerprint:
                errors.append(location + " duplicate query has a different candidate pool")
            else:
                pool_fingerprint_for_key[key] = fingerprint
            if len(ranking) != len(candidates):
                errors.append(location + f" ranking_count={len(ranking)}")
                continue
            candidate_ids = {str(item.get("candidate_id") or "") for item in candidates}
            ranked_ids = [str(item.get("candidate_id") or "") for item in ranking]
            if len(ranked_ids) != len(set(ranked_ids)) or set(ranked_ids) != candidate_ids:
                errors.append(location + " ranking ID mismatch")
                continue

            ranks = [item.get("rank") for item in ranking]
            if ranks != list(range(1, len(ranking) + 1)):
                errors.append(location + " ranking ranks are not contiguous 1..N")
                continue

            labels = [str(item.get("label") or "") for item in ranking]
            invalid = sorted(set(labels) - VALID_LABELS)
            if invalid:
                errors.append(location + f" invalid_labels={invalid}")
                continue
            tiers = [LABEL_TIER[label] for label in labels]
            if tiers != sorted(tiers):
                errors.append(location + " ranking violates D>P>S>X>I tier order")
            per_label = Counter(labels)
            for label, maximum in MAX_LABEL_COUNTS.items():
                if per_label[label] > maximum:
                    errors.append(
                        location + f" label cap exceeded: {label}={per_label[label]}>{maximum}"
                    )
            candidate_text = {
                str(item.get("candidate_id") or ""): str(item.get("text") or "")
                for item in candidates
            }
            for item in ranking:
                candidate_id = str(item.get("candidate_id") or "")
                label = str(item.get("label") or "")
                quote = str(item.get("support_quote") or "")
                normalized_quote = normalized_text(quote)
                if label in POSITIVE_LABELS:
                    if not (4 <= len(normalized_quote) <= 120):
                        errors.append(location + f" invalid positive quote length: {candidate_id}")
                    elif normalized_quote not in normalized_text(candidate_text[candidate_id]):
                        errors.append(location + f" non-verbatim positive quote: {candidate_id}")
                elif quote.strip():
                    errors.append(location + f" X/I quote must be empty: {candidate_id}")
            counts.update("label_" + label for label in labels)
            decision = str(
                teacher.get("query_decision", {}).get("status")
                or "answerable"
            )
            if decision not in {"answerable", "ambiguous"}:
                errors.append(location + f" invalid query decision={decision}")
            if decision == "ambiguous":
                counts["ambiguous"] += 1
                if any(label in POSITIVE_LABELS for label in labels):
                    errors.append(location + " ambiguous row has positive labels")
            else:
                positive_count = sum(label in POSITIVE_LABELS for label in labels)
                if positive_count == 0:
                    counts["no_positive"] += 1
                elif positive_count > max_trainable_positives:
                    counts[f"over_broad_positive_gt{max_trainable_positives}"] += 1
                else:
                    counts["usable"] += 1
                    usable_keys.add(key)

    return {
        "paths": [str(path) for path in paths],
        "counts": dict(sorted(counts.items())),
        "providers": dict(sorted(providers.items())),
        "unique_query_keys": len(keys),
        "usable_query_keys": len(usable_keys),
        "story_families": sorted(families),
        "duplicate_query_keys": sum(duplicate_keys.values()),
        "duplicate_provider_edges": dict(sorted(duplicate_keys.items())),
        "errors": errors[:100],
        "error_count": len(errors),
        "query_keys": keys,
        "pool_fingerprints": pool_fingerprint_for_key,
    }


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"query_keys", "pool_fingerprints"}
    }


def compare_to_expected(
    actual: dict[str, Any], expected: dict[str, Any] | None
) -> dict[str, Any]:
    if expected is None:
        return {"checked": False}
    missing = expected["query_keys"] - actual["query_keys"]
    extra = actual["query_keys"] - expected["query_keys"]
    changed_pools = sorted(
        key
        for key in expected["query_keys"] & actual["query_keys"]
        if expected["pool_fingerprints"].get(key)
        != actual["pool_fingerprints"].get(key)
    )
    return {
        "checked": True,
        "input": expected["path"],
        "expected_query_keys": len(expected["query_keys"]),
        "labelled_query_keys": len(actual["query_keys"]),
        "missing_query_key_count": len(missing),
        "missing_query_key_examples": sorted(missing)[:10],
        "extra_query_key_count": len(extra),
        "extra_query_key_examples": sorted(extra)[:10],
        "changed_candidate_pool_count": len(changed_pools),
        "changed_candidate_pool_examples": changed_pools[:10],
        "input_error_count": expected["error_count"],
        "input_errors": expected["errors"],
    }


def main() -> int:
    args = parse_args()
    train = audit_split(
        args.train, args.expected_candidates, args.expected_prompt_version
    )
    val = audit_split(args.val, args.expected_candidates, args.expected_prompt_version)
    expected_train = (
        scan_expected_input(args.train_input, args.expected_candidates)
        if args.train_input
        else None
    )
    expected_val = (
        scan_expected_input(args.val_input, args.expected_candidates)
        if args.val_input
        else None
    )
    train_coverage = compare_to_expected(train, expected_train)
    val_coverage = compare_to_expected(val, expected_val)
    overlap = train["query_keys"] & val["query_keys"]
    train_families = set(train["story_families"])
    val_families = set(val["story_families"])
    family_overlap = train_families & val_families
    report = {
        "train": public_summary(train),
        "val": public_summary(val),
        "source_coverage": {
            "train": train_coverage,
            "val": val_coverage,
        },
        "leakage": {
            "query_key_overlap_count": len(overlap),
            "query_key_overlap_examples": sorted(overlap)[:10],
            "story_family_overlap_count": len(family_overlap),
            "story_family_overlap_examples": sorted(family_overlap)[:10],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    coverage_failed = any(
        coverage.get(key, 0)
        for coverage in (train_coverage, val_coverage)
        if coverage.get("checked")
        for key in (
            "missing_query_key_count",
            "extra_query_key_count",
            "changed_candidate_pool_count",
            "input_error_count",
        )
    )
    if (
        train["error_count"]
        or val["error_count"]
        or overlap
        or family_overlap
        or coverage_failed
    ):
        raise RuntimeError("label audit failed; see report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
