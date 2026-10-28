#!/usr/bin/env python3
"""Risk-stratified label inventory; never edits labels or declares semantic errors.

Only aggregate JSON may be public. Row details and evidence bundles MUST go to
an explicit private directory. Optional token counts require a local tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"min": ordered[0], "median": statistics.median(ordered),
            "p90_nearest_rank": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
            "max": ordered[-1], "mean": statistics.mean(ordered)}


def inspect(row, position, tokenizer=None):
    raw = row["target"]
    if not isinstance(raw, str):
        raise ValueError(f"row {position}: target must be serialized JSON string")
    label = json.loads(raw)
    facts = label.get("supported_facts", [])
    # Offsets refer to the *original serialized target*, including JSON escaping.
    # Search the next exact fact string; fail rather than invent offsets.
    cursor = 0
    spans = []
    for i, fact in enumerate(facts, 1):
        encoded = json.dumps(fact["fact"], ensure_ascii=False)
        start = raw.find(encoded, cursor)
        if start < 0:
            # Alternate ASCII JSON serialization is legal input.
            encoded = json.dumps(fact["fact"], ensure_ascii=True)
            start = raw.find(encoded, cursor)
        if start < 0:
            raise ValueError(f"row {position}: cannot locate serialized fact {i}")
        end = start + len(encoded)
        spans.append({"index": i, "target_char_start": start,
                      "target_char_end": end, "fact_chars": len(fact["fact"]),
                      "evidence_ids": fact["evidence_ids"]})
        cursor = end
    refs = {e for f in facts for e in f["evidence_ids"]}
    documents = {d["evidence_id"]: d for d in row.get("documents", [])}
    cited_docs = [documents[e]["document_id"] for e in refs if e in documents]
    # A passage document_id commonly ends in #chunk-NNNN; same family does not
    # establish same scene, so these are selection signals, not error labels.
    stages = {d.split("#chunk-")[0] for d in cited_docs}
    flags = []
    kind = row.get("targeted_kind", "unspecified")
    if kind in {"minimal_complete", "partial_support"} and len(facts) >= 4:
        flags.append("high_fact_core_or_partial")
    if kind == "partial_support" and label["next_action"] != "answer_directly":
        flags.append("partial_action_review")
    if len(stages) >= 2:
        flags.append("multi_stage_citations")
    unknown = [i for i, f in enumerate(facts, 1)
               if any(s in f["fact"] for s in ("未说明", "未明确", "未确认", "无法确认", "未知"))]
    result = {"position": position, "id": row["id"], "kind": kind,
              "action": label["next_action"], "target_chars": len(raw),
              "fact_count": len(facts), "fact_spans": spans,
              "cited_stage_count": len(stages), "unknown_marker_fact_indices": unknown,
              "flags_for_review_not_errors": flags}
    if tokenizer is not None:
        tokens = tokenizer(raw, add_special_tokens=False,
                           return_offsets_mapping=True)
        offsets = tokens["offset_mapping"]
        result["target_tokens_no_eos"] = len(tokens["input_ids"])
        for span in spans:
            # Counts overlapping tokens; boundary tokens may overlap fields and
            # must not be interpreted as a disjoint CE-loss decomposition.
            span["overlapping_target_tokens"] = sum(
                a < span["target_char_end"] and b > span["target_char_start"]
                for a, b in offsets)
    return result


def aggregate(details):
    return {"rows": len(details),
            "actions": dict(sorted(Counter(x["action"] for x in details).items())),
            "fact_counts": dict(sorted(Counter(x["fact_count"] for x in details).items())),
            "target_chars": quantiles([x["target_chars"] for x in details]),
            "target_tokens_no_eos": quantiles([x["target_tokens_no_eos"] for x in details
                                                if "target_tokens_no_eos" in x]),
            "selection_flags_not_errors": dict(Counter(f for x in details
                                     for f in x["flags_for_review_not_errors"]))}


def run(datasets, private_output, aggregate_output, top_k=2, tokenizer=None):
    private_output.mkdir(parents=True, exist_ok=True)
    report = {"kind": "available_samples_label_diagnostic",
              "semantic_accuracy_measured": False,
              "sampling": "deterministic risk strata plus high-fact controls; not random",
              "scope": "Each dataset is separate; no inference to absent 1117 or final repaired 180.",
              "units": "Unicode characters are not model tokens. Token counts optional and local-only.",
              "datasets": {}}
    for name, path in datasets:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("dataset name must be alphanumeric, underscore or hyphen")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate ids")
        details = [inspect(r, i, tokenizer) for i, r in enumerate(rows, 1)]
        groups = defaultdict(list)
        for item in details:
            groups[item["kind"]].append(item)
        report["datasets"][name] = {"source_sha256": digest(path),
            "all": aggregate(details),
            "by_kind": {k: aggregate(v) for k, v in sorted(groups.items())}}
        selected = set()
        for group in groups.values():
            for item in sorted(group, key=lambda x: (-x["fact_count"], -x["target_chars"], x["id"]))[:top_k]:
                selected.add(item["id"])
            selected.update(x["id"] for x in group
                            if "partial_action_review" in x["flags_for_review_not_errors"])
        with (private_output / f"{name}.inventory.jsonl").open("w") as out:
            for item in details:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
        with (private_output / f"{name}.review_bundle.jsonl").open("w") as out:
            for row, item in zip(rows, details):
                if row["id"] not in selected:
                    continue
                out.write(json.dumps({"diagnostic": item, "question": row["question"],
                    "round": row.get("round"), "target": row["target"],
                    "documents": row["documents"], "review_status": "unreviewed",
                    "ready_for_training": False}, ensure_ascii=False) + "\n")
        report["datasets"][name]["selected_review_rows"] = len(selected)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=JSONL")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    parser.add_argument("--top-k-per-kind", type=int, default=2)
    parser.add_argument("--local-tokenizer", type=Path)
    args = parser.parse_args()
    if args.top_k_per_kind < 1:
        parser.error("top-k must be positive")
    tokenizer = None
    if args.local_tokenizer:
        if not args.local_tokenizer.is_dir():
            parser.error("local tokenizer directory is absent; network download forbidden")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer, local_files_only=True,
                                                   trust_remote_code=False, use_fast=True)
        if not tokenizer.is_fast:
            parser.error("offset token accounting requires a local fast tokenizer")
    datasets = [(name, Path(path)) for name, path in
                (arg.split("=", 1) for arg in args.dataset)]
    run(datasets, args.private_output, args.aggregate_output,
        args.top_k_per_kind, tokenizer)


if __name__ == "__main__":
    main()
