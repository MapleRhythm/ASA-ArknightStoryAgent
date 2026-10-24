#!/usr/bin/env python3
"""Create a deterministic, balanced claim/chunk manifest for agreement eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "${REMOTE_STORE}/exx_grounding_v1/data/binding_hardneg_v2_20260904.jsonl"


def read_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        positives = [str(x).strip() for x in row.get("positive_texts") or [] if str(x).strip()]
        negatives = [
            h
            for h in row.get("hard_negatives") or []
            if isinstance(h, dict)
            and h.get("glm_verdict") == "unsupported"
            and str(h.get("text") or "").strip()
        ]
        if positives and negatives and str(row.get("claim") or "").strip():
            result.append(row)
    return result


def allocate_quotas(groups: dict[str, list[dict[str, Any]]], n: int) -> dict[str, int]:
    """Proportional allocation with at least one row per non-empty stratum."""
    names = sorted(groups)
    total = sum(len(groups[name]) for name in names)
    if n >= len(names):
        quotas = {name: 1 for name in names}
    else:
        # This branch is not used by the requested 20/250 row samples, but
        # keeps the helper well-defined for tiny exploratory runs.
        ranked = sorted(names, key=lambda name: (-len(groups[name]), name))
        return {name: int(name in ranked[:n]) for name in names}
    remaining = n - sum(quotas.values())
    raw = {name: n * len(groups[name]) / total for name in names}
    for name in names:
        quotas[name] += max(0, int(raw[name]) - 1)
    # Correct the floor/minimum allocation to exactly n using fractional parts.
    while sum(quotas.values()) > n:
        candidates = [name for name in names if quotas[name] > 1]
        name = min(candidates, key=lambda x: (raw[x] - int(raw[x]), -len(groups[x]), x))
        quotas[name] -= 1
    remainders = sorted(
        names,
        key=lambda name: (raw[name] - int(raw[name]), len(groups[name]), name),
        reverse=True,
    )
    index = 0
    while sum(quotas.values()) < n:
        name = remainders[index % len(remainders)]
        if quotas[name] < len(groups[name]):
            quotas[name] += 1
        index += 1
    return quotas


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("query_type") or "unknown")].append(row)
    rng = random.Random(seed)
    for name in groups:
        rng.shuffle(groups[name])
    quotas = allocate_quotas(groups, n)
    selected: list[dict[str, Any]] = []
    for name in sorted(groups):
        selected.extend(groups[name][: quotas[name]])
    rng.shuffle(selected)
    return selected


def stable_index(key: str, size: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def build_pairs(rows: list[dict[str, Any]], seed: int, chunk_chars: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        claim = str(row["claim"]).strip()
        positives = [str(x).strip() for x in row.get("positive_texts") or [] if str(x).strip()]
        negatives = [
            h
            for h in row.get("hard_negatives") or []
            if isinstance(h, dict)
            and h.get("glm_verdict") == "unsupported"
            and str(h.get("text") or "").strip()
        ]
        base_key = f"{row.get('record_id')}|{row.get('fact_index')}|{seed}"
        positive_text = positives[stable_index(base_key + "|positive", len(positives))]
        negative = negatives[stable_index(base_key + "|negative", len(negatives))]
        row_key = f"{row.get('record_id')}::{row.get('fact_index')}::{row_number}"
        common = {
            "row_key": row_key,
            "record_id": row.get("record_id"),
            "question_key": row.get("question_key"),
            "fact_index": row.get("fact_index"),
            "query": row.get("query"),
            "query_type": row.get("query_type") or "unknown",
            "claim": claim,
        }
        pos = dict(common)
        pos.update(
            {
                "pair_id": hashlib.sha256((row_key + "|positive").encode()).hexdigest()[:20],
                "side": "positive",
                "gold_label": 1,
                "evidence_id": (row.get("positive_eids") or [None])[0],
                "source_glm_verdict": "gold_positive",
                "chunk_original_chars": len(positive_text),
                "chunk_text": positive_text[:chunk_chars],
                "chunk_truncated": len(positive_text) > chunk_chars,
            }
        )
        neg = dict(common)
        negative_text = str(negative["text"]).strip()
        neg.update(
            {
                "pair_id": hashlib.sha256((row_key + "|negative").encode()).hexdigest()[:20],
                "side": "negative",
                "gold_label": 0,
                "evidence_id": negative.get("eid"),
                "source_glm_verdict": negative.get("glm_verdict"),
                "chunk_original_chars": len(negative_text),
                "chunk_text": negative_text[:chunk_chars],
                "chunk_truncated": len(negative_text) > chunk_chars,
                "source_glm_reason": negative.get("glm_reason", ""),
            }
        )
        pairs.extend([pos, neg])
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--chunk-chars", type=int, default=600)
    args = parser.parse_args()
    if args.pair_count <= 0 or args.pair_count % 2:
        raise SystemExit("--pair-count must be a positive even number")
    all_rows = eligible_rows(read_rows(args.input))
    n_rows = args.pair_count // 2
    if n_rows > len(all_rows):
        raise SystemExit(f"requested {n_rows} rows but only {len(all_rows)} eligible")
    selected = sample_rows(all_rows, n_rows, args.seed)
    pairs = build_pairs(selected, args.seed, args.chunk_chars)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {
        "seed": args.seed,
        "pair_count": len(pairs),
        "row_count": len(selected),
        "chunk_chars": args.chunk_chars,
        "eligible_fact_rows": len(all_rows),
        "query_type_rows": dict(Counter(str(r.get("query_type") or "unknown") for r in selected)),
        "query_type_pairs": dict(Counter(str(p["query_type"]) for p in pairs)),
        "side_pairs": dict(Counter(str(p["side"]) for p in pairs)),
        "truncated_pairs": sum(bool(p["chunk_truncated"]) for p in pairs),
    }
    with open(str(args.output) + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
