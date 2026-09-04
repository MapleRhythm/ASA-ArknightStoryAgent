#!/usr/bin/env python3
"""Build clean pointwise/pairwise binding-verifier data.

The source hard-negative file contains both audited and provisional candidates.
This script joins it with the GLM gold recalibration manifest and emits only:

* clean/partial gold bindings as positives (partial keeps only confirmed E-IDs);
* candidates explicitly judged ``unsupported`` by GLM as negatives.

Un-audited candidates and suspected missed positives are intentionally omitted.
The output is grouped by source record before the deterministic train/eval split
so a question cannot leak across the two sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def classify(judgement: dict[str, Any]) -> tuple[str, list[str]]:
    merged = judgement.get("merged") or {}
    individual = judgement.get("individual") or {}
    eids = [str(item) for item in judgement.get("cited_eids") or []]
    merged_verdict = merged.get("verdict")
    supported = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "supported"]
    unsupported = [eid for eid in eids if (individual.get(eid) or {}).get("verdict") == "unsupported"]
    uncertain = [
        eid
        for eid in eids
        if (individual.get(eid) or {}).get("verdict") in {"skip", "error", None}
    ]
    if merged_verdict != "supported":
        return "unsupported", []
    if uncertain:
        kept = supported
        return ("partial" if kept else "unsupported"), kept
    if unsupported:
        return ("partial" if supported else "unsupported"), supported
    return "clean_support", eids


def stable_eval(record_id: str, eval_ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < eval_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(args.manifest)
    task_lookup = {
        (str(row.get("row_id") or ""), int(row.get("fact_index", -1))): row
        for row in manifest
    }
    judgement_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in read_jsonl(args.judgements):
        key = (str(row.get("row_id") or ""), int(row.get("fact_index", -1)))
        judgement_lookup[key] = row

    pairs: list[dict[str, Any]] = []
    stats = Counter()
    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in read_jsonl(args.hardneg):
        record_id = str(source.get("record_id") or "")
        key = (record_id, int(source.get("fact_index", -1)))
        task = task_lookup.get(key)
        judgement = judgement_lookup.get(key)
        if task is None or judgement is None:
            stats["missing_join"] += 1
            continue
        status, kept_eids = classify(judgement)
        stats[f"gold_status:{status}"] += 1
        if status not in {"clean_support", "partial"} or not kept_eids:
            stats["dropped_nonpositive_gold"] += 1
            continue
        positive = "\n\n".join(
            f"[{eid}]\n{str((task.get('cited_texts') or {}).get(eid) or '').strip()}"
            for eid in kept_eids
            if str((task.get("cited_texts") or {}).get(eid) or "").strip()
        ).strip()
        if not positive:
            stats["dropped_empty_positive"] += 1
            continue
        confirmed = [
            item
            for item in source.get("hard_negatives") or []
            if str(item.get("glm_verdict") or "") == "unsupported"
        ]
        stats["confirmed_candidates"] += len(confirmed)
        stats["provisional_candidates"] += sum(
            str(item.get("glm_verdict") or "") == "not_run"
            for item in source.get("hard_negatives") or []
        )
        stats["suspected_missed_positives"] += len(source.get("suspected_missed_positives") or [])
        for item in confirmed:
            negative = str(item.get("text") or "").strip()
            if not negative or negative == positive:
                stats["dropped_empty_or_equal_negative"] += 1
                continue
            pair = {
                "query": str(source.get("claim") or task.get("claim") or "").strip(),
                "query_type": str(source.get("query_type") or task.get("query_type") or "unknown"),
                "source_name": str(source.get("source_name") or record_id),
                "positive": positive,
                "negative": negative,
                "positive_score": 1.0,
                "negative_score": 0.0,
                "negative_type": "binding_glm_confirmed_unsupported",
                "answer": "",
                "answer_evidence": kept_eids,
                "answer_focus": "",
                "positive_chain": kept_eids,
                "negative_chain": [str(item.get("eid") or "")],
                "record_id": record_id,
                "fact_index": int(source.get("fact_index", -1)),
                "gold_status": status,
                "negative_glm_reason": str(item.get("glm_reason") or ""),
                "negative_cos_sim": item.get("cos_sim"),
            }
            pairs.append(pair)
            group_records[record_id].append(pair)

    eval_ids = {
        record_id
        for record_id in group_records
        if stable_eval(record_id, args.eval_ratio, args.seed)
    }
    # Keep both partitions non-empty when data permits.
    if not eval_ids and len(group_records) > 1:
        eval_ids.add(sorted(group_records)[0])
    train = [pair for pair in pairs if pair["record_id"] not in eval_ids]
    eval_rows = [pair for pair in pairs if pair["record_id"] in eval_ids]
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(eval_rows)

    for name, rows in (("pairwise.jsonl", pairs), ("train.jsonl", train), ("eval.jsonl", eval_rows)):
        with (args.out_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(compact(row) + "\n")
    report = {
        "source": {
            "hardneg": str(args.hardneg),
            "manifest": str(args.manifest),
            "judgements": str(args.judgements),
        },
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "pairs": len(pairs),
        "train_pairs": len(train),
        "eval_pairs": len(eval_rows),
        "groups": len(group_records),
        "train_groups": len({row["record_id"] for row in train}),
        "eval_groups": len({row["record_id"] for row in eval_rows}),
        "query_types": dict(Counter(str(row.get("query_type") or "unknown") for row in pairs)),
        "gold_status": dict(Counter(str(row.get("gold_status") or "unknown") for row in pairs)),
        "negative_type": dict(Counter(str(row.get("negative_type") or "unknown") for row in pairs)),
        "stats": dict(stats),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(
            {
                "binding_verifier_train": {"file_name": "train.jsonl"},
                "binding_verifier_eval": {"file_name": "eval.jsonl"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
