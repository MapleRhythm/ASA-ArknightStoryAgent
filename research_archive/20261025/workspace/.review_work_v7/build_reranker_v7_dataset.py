#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PART_RE = re.compile(r"_part\d+(?=\.json$)", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"\[E(\d+)\][\s\S]*?(?=\s*\[E\d+\]|\Z)")
DROP_NEGATIVE_TYPES = {"shuffled_order", "weak_original_gold"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def char_ngrams(value: Any, size: int = 3) -> set[str]:
    text = normalized_text(value)
    if len(text) < size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def overlap_coefficient(left: Any, right: Any) -> float:
    left_grams = char_ngrams(left)
    right_grams = char_ngrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams))


def story_family(source_name: Any) -> str:
    name = Path(str(source_name or "unknown")).name
    return PART_RE.sub("", name).removesuffix(".json")


def stable_score(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_families(
    listwise: list[dict[str, Any]], *, seed: int, val_ratio: float, test_ratio: float
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in listwise:
        grouped[story_family(record.get("source_name"))].append(record)
    ordered = sorted(grouped, key=lambda family: stable_score(family, seed))
    targets = {
        "test": max(1, round(len(listwise) * test_ratio)),
        "val": max(1, round(len(listwise) * val_ratio)),
    }
    assignment = {family: "train" for family in ordered}
    for split in ("test", "val"):
        count = 0
        for family in ordered:
            if assignment[family] != "train":
                continue
            assignment[family] = split
            count += len(grouped[family])
            if count >= targets[split]:
                break
    return assignment


def query_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("source_name") or ""), str(record.get("query") or "")


def assignment_for_pair(record: dict[str, Any], assignment: dict[str, str]) -> str | None:
    """Return a split only when the pair's story family is explicitly known.

    Patch-only records cannot be proven to belong to the training stories.
    Silently defaulting them to train weakens the family-holdout guarantee.
    """
    return assignment.get(story_family(record.get("source_name")))


def clean_pair(record: dict[str, Any], stats: Counter[str]) -> dict[str, Any] | None:
    negative_type = str(record.get("negative_type") or "unknown")
    if negative_type in DROP_NEGATIVE_TYPES:
        stats[f"drop_type:{negative_type}"] += 1
        return None
    positive = str(record.get("positive") or "").strip()
    negative = str(record.get("negative") or "").strip()
    if not str(record.get("query") or "").strip() or not positive or not negative:
        stats["drop_empty"] += 1
        return None
    if normalized_text(positive) == normalized_text(negative):
        stats["drop_equal_text"] += 1
        return None
    # Online mining uses a permissive hit detector so as not to miss useful
    # hard negatives. Apply a stricter final guard before training: a candidate
    # that contains nearly all positive trigrams is likely another rendering of
    # the gold passage and must not be labelled negative.
    if negative_type.startswith("online_high_rank_") and overlap_coefficient(positive, negative) >= 0.8:
        stats["drop_online_probable_false_negative"] += 1
        return None
    answer_evidence = {str(item) for item in record.get("answer_evidence") or [] if str(item)}
    negative_chain = {str(item) for item in record.get("negative_chain") or [] if str(item)}
    # A negative containing every labelled answer segment is usually a reordered
    # or annotation-conflicted positive. Teaching it as negative creates the very
    # false-abstention behaviour this dataset is meant to reduce.
    if answer_evidence and answer_evidence.issubset(negative_chain):
        stats[f"drop_negative_covers_gold:{negative_type}"] += 1
        return None
    output = dict(record)
    output["story_family"] = story_family(record.get("source_name"))
    return output


def choose_pairs(records: list[dict[str, Any]], *, max_negatives: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[query_key(record)].append(record)
    output: list[dict[str, Any]] = []
    rng = random.Random(seed)
    priority = {
        "online_high_rank_chain": 0,
        "online_high_rank_doc": 1,
        "same_entity_distractor": 2,
        "answer_adjacent": 3,
        "partial_answer": 4,
        "background_only": 5,
    }
    for key in sorted(grouped):
        candidates = grouped[key]
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: priority.get(str(item.get("negative_type") or ""), 6))
        seen_types: set[str] = set()
        selected: list[dict[str, Any]] = []
        for record in candidates:
            negative_type = str(record.get("negative_type") or "unknown")
            if negative_type in seen_types:
                continue
            seen_types.add(negative_type)
            selected.append(record)
            if len(selected) >= max_negatives:
                break
        if len(selected) < max_negatives:
            selected_keys = {
                (normalized_text(item.get("positive")), normalized_text(item.get("negative")))
                for item in selected
            }
            for record in candidates:
                key2 = (normalized_text(record.get("positive")), normalized_text(record.get("negative")))
                if key2 in selected_keys:
                    continue
                selected.append(record)
                selected_keys.add(key2)
                if len(selected) >= max_negatives:
                    break
        output.extend(selected)
    return output


def listwise_for_split(
    records: list[dict[str, Any]], assignment: dict[str, str], split: str
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        family = story_family(record.get("source_name"))
        if assignment.get(family, "train") != split:
            continue
        copied = dict(record)
        copied["story_family"] = family
        output.append(copied)
    return output


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "queries": len({query_key(record) for record in records}),
        "story_families": len({story_family(record.get("source_name")) for record in records}),
        "query_types": dict(Counter(str(record.get("query_type") or "unknown") for record in records)),
        "negative_types": dict(Counter(str(record.get("negative_type") or "unknown") for record in records)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-resistant ASA reranker v7 splits.")
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument("--listwise", type=Path, required=True)
    parser.add_argument("--mined-pairwise", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-negatives-per-query", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairwise = read_jsonl(args.pairwise)
    listwise = read_jsonl(args.listwise)
    if args.mined_pairwise and args.mined_pairwise.exists():
        pairwise.extend(read_jsonl(args.mined_pairwise))
    assignment = split_families(
        listwise, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )
    stats: Counter[str] = Counter()
    clean = [item for record in pairwise if (item := clean_pair(record, stats)) is not None]
    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "val", "test")}
    for record in clean:
        split = assignment_for_pair(record, assignment)
        if split is None:
            stats["drop_unknown_story_family"] += 1
            continue
        by_split[split].append(record)
    by_split = {
        split: choose_pairs(records, max_negatives=args.max_negatives_per_query, seed=args.seed + index)
        for index, (split, records) in enumerate(by_split.items())
    }
    listwise_splits = {
        split: listwise_for_split(listwise, assignment, split) for split in ("train", "val", "test")
    }
    # Strong leakage audit: no story family and no exact query/source group may cross splits.
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_families = {story_family(item.get("source_name")) for item in by_split[left]}
        right_families = {story_family(item.get("source_name")) for item in by_split[right]}
        if left_families & right_families:
            raise RuntimeError(f"story-family leakage between {left} and {right}")
        if {query_key(item) for item in by_split[left]} & {query_key(item) for item in by_split[right]}:
            raise RuntimeError(f"query/source leakage between {left} and {right}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        write_jsonl(args.output_dir / f"{split}_pairwise.jsonl", by_split[split])
        write_jsonl(args.output_dir / f"{split}_listwise.jsonl", listwise_splits[split])
    manifest = {
        "pairwise_source": str(args.pairwise),
        "listwise_source": str(args.listwise),
        "mined_pairwise_source": str(args.mined_pairwise) if args.mined_pairwise else None,
        "seed": args.seed,
        "split_unit": "story_family",
        "val_ratio_target": args.val_ratio,
        "test_ratio_target": args.test_ratio,
        "max_negatives_per_query": args.max_negatives_per_query,
        "cleaning_stats": dict(stats),
        "pairwise": {split: summarize(by_split[split]) for split in by_split},
        "listwise": {
            split: {
                "records": len(records),
                "story_families": len({story_family(item.get("source_name")) for item in records}),
                "query_types": dict(Counter(str(item.get("query_type") or "unknown") for item in records)),
            }
            for split, records in listwise_splits.items()
        },
        "families": {
            split: sorted(family for family, assigned in assignment.items() if assigned == split)
            for split in ("train", "val", "test")
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
