#!/usr/bin/env python3
"""Audit question/source components and supervised story exposure across splits.

This is a conservative pretraining manifest, NOT a claim that semantic question
families have been independently annotated. Near-duplicate text is a heuristic;
shared retrieved story footprints are exposure risks, not proof of answer
leakage. The corpus merely existing in an index is not counted as supervision.
No labels are edited, and this tool emits no train.json or training registry.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_exx_eval_split import normalize_question, record_question
from build_recalibrated_retrieval_manifest import read_json_rows


def content_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("meta") or {}
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError("metadata_must_be_object")
    return value


def story_footprint(doc_id: str) -> str:
    path = str(doc_id).replace("\\", "/").split("#", 1)[0].strip()
    parts = path.split("/")
    if path.startswith("obt/main/"):
        chapter = re.search(r"level_(?:main|st)_(\d+)-", path)
        if chapter:
            return f"obt/main/chapter_{chapter.group(1)}"
    if parts[0] == "activities" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] == "obt" and len(parts) > 2:
        return "/".join(parts[:3])
    return "/".join(parts[:2]) if path else ""


def signals(row: dict[str, Any], tasks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = metadata(row)
    diagnostic = row.get("diagnostic") or {}
    if not isinstance(diagnostic, dict):
        raise ValueError("invalid_diagnostic")
    identities = {str(row["id"])}
    for info in (row, meta, diagnostic):
        for key in ("source_id", "source_matrix_id", "task_id"):
            if info.get(key):
                identities.add(str(info[key]))
    if str(row["id"]).startswith("relabel-"):
        identities.add(str(row["id"]).removeprefix("relabel-"))
    task = next((tasks[key] for key in sorted(identities) if key in tasks), {})
    for info in (meta, task):
        for ref in info.get("source_refs") or []:
            if not isinstance(ref, dict) or not ref.get("id"):
                raise ValueError("invalid_source_ref")
            identities.add(str(ref["id"]))
    questions = {
        normalize_question(str(value)).casefold()
        for value in (
            record_question(row), row.get("question"), meta.get("source_question"),
            task.get("question"),
        )
        if value
    }
    questions.discard("")
    tokens = {f"id:{value}" for value in identities}
    tokens.update(f"question:{value}" for value in questions)
    for info in (row, meta, diagnostic):
        for key in ("question_key", "source_question_key", "event_family"):
            if info.get(key):
                namespace = "question_key" if key.endswith("question_key") else key
                tokens.add(f"{namespace}:{info[key]}")
    doc_ids = set()
    for info in (row, meta):
        for key in ("evidence_doc_ids", "gold_evidence_ids"):
            doc_ids.update(str(value) for value in info.get(key) or [])
    for evidence in task.get("evidence") or []:
        if evidence.get("doc_id"):
            doc_ids.add(str(evidence["doc_id"]))
    for info in (row, meta, diagnostic):
        doc_ids.update(str(value) for value in (info.get("evidence_id_to_corpus_id") or {}).values())
    footprints = {story_footprint(value) for value in doc_ids}
    for info in (row, meta, diagnostic):
        footprints.update(story_footprint(str(value)) for value in info.get("story_families") or [])
        family = str(info.get("question_family") or "")
        if family:
            # Historical strict9 question_family is a pipe-joined corpus
            # footprint, not an independently annotated semantic family.
            if "/" in family:
                footprints.update(story_footprint(value) for value in family.split("|"))
            else:
                tokens.add(f"question_family:{family}")
    footprints.discard("")
    return {"tokens": tokens, "questions": questions, "story_footprints": footprints}


def build_report(
    splits: dict[str, list[dict[str, Any]]],
    tasks: list[dict[str, Any]],
    near_threshold: float = 0.88,
) -> dict[str, Any]:
    task_map = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in task_map:
            raise ValueError("missing_or_duplicate_task_id")
        task_map[task_id] = task
    records = []
    for split, rows in sorted(splits.items()):
        seen = set()
        for row in rows:
            if not row.get("id") or row["id"] in seen:
                raise ValueError("missing_or_duplicate_split_row_id")
            seen.add(row["id"])
            records.append({
                "split": split, "id": row["id"], "row_sha256": content_sha256(row),
                **signals(row, task_map),
            })
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    direct_edges = []
    for index, row in enumerate(records):
        for token in sorted(row["tokens"]):
            if token in first:
                other = first[token]
                parent[root(index)] = root(other)
                if records[other]["split"] != row["split"]:
                    direct_edges.append({
                        "left": [records[other]["split"], records[other]["id"]],
                        "right": [row["split"], row["id"]],
                        "signal": token,
                    })
            else:
                first[token] = index
    components = defaultdict(list)
    for index in range(len(records)):
        components[root(index)].append(index)
    heldout_questions = sorted({
        q for row in records if row["split"] != "train" for q in row["questions"]
    })
    heldout_footprints = {
        f for row in records if row["split"] != "train" for f in row["story_footprints"]
    }
    manifest, component_reports = [], []
    counts = Counter()
    for members in components.values():
        all_tokens = sorted({t for i in members for t in records[i]["tokens"]})
        group_id = hashlib.sha256("\n".join(all_tokens).encode()).hexdigest()
        group_splits = {records[i]["split"] for i in members}
        footprints = {f for i in members for f in records[i]["story_footprints"]}
        group_questions = {q for i in members for q in records[i]["questions"]}
        near_matches = set()
        if "train" in group_splits and len(group_splits) == 1 and near_threshold:
            for question in sorted(group_questions):
                near_matches.update(difflib.get_close_matches(
                    question, heldout_questions, n=1, cutoff=near_threshold,
                ))
        story_overlap = sorted(footprints & heldout_footprints)
        cross_split = "train" in group_splits and len(group_splits) > 1
        if len(group_splits) > 1:
            component_reports.append({
                "component_id": group_id,
                "members": [[records[i]["split"], records[i]["id"]] for i in members],
                "questions": sorted(group_questions),
            })
        for index in members:
            row = records[index]
            reasons = []
            if row["split"] != "train":
                reasons.append("heldout_source")
            else:
                if cross_split:
                    reasons.append("cross_split_question_or_lineage")
                if near_matches:
                    reasons.append("near_question_overlap")
                if story_overlap:
                    reasons.append("supervised_story_exposure_overlap")
                if not group_questions:
                    reasons.append("missing_question")
                if not footprints:
                    reasons.append("missing_story_provenance")
            entry = {
                key: row[key] for key in ("id", "split", "row_sha256")
            } | {
                "component_id": group_id,
                "questions": sorted(row["questions"]),
                "story_footprints": sorted(footprints),
                "heldout_story_overlap": story_overlap if row["split"] == "train" else [],
                "near_heldout_questions": sorted(near_matches) if row["split"] == "train" else [],
                "quarantine_reasons": reasons,
                "partition_candidate": not reasons,
                "training_ready": False,
            }
            manifest.append(entry)
            counts[f"{row['split']}:rows"] += 1
            if row["split"] == "train":
                counts["train:partition_candidates"] += int(not reasons)
                counts.update(f"train:{reason}" for reason in reasons)
    return {
        "protocol": "asa_training_lineage_audit_v1",
        "counts": dict(sorted(counts.items())),
        "components": len(components),
        "cross_split_components": component_reports,
        "direct_overlap_edges": direct_edges,
        "near_threshold": near_threshold,
        "records": sorted(manifest, key=lambda row: (row["split"], row["id"])),
        "limitations": [
            "Exact source/question identity uses transitive components, not a random row split.",
            "Text similarity is heuristic; undetected paraphrases may remain.",
            "Story footprints describe supervised evidence exposure, not semantic answer leakage or all corpus indexing.",
            "Unknown provenance is quarantined; partition candidates still require full-evidence and set-level label validation.",
            "No legacy model becomes held-out retroactively by rebuilding this manifest.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", required=True, help="train|val|test|development=/path")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--near-threshold", type=float, default=0.88)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite audit")
    if args.near_threshold != 0 and not 0 < args.near_threshold <= 1:
        parser.error("near threshold must be zero or (0,1]")
    splits, paths = {}, {}
    for spec in args.split:
        name, path = spec.split("=", 1)
        if name in splits or name not in {"train", "val", "test", "development"}:
            parser.error("duplicate or unknown split name")
        paths[name] = Path(path)
        splits[name] = read_json_rows(paths[name])
    if "train" not in splits or not any(rows for name, rows in splits.items() if name != "train"):
        parser.error("need train and nonempty heldout sources")
    report = build_report(splits, read_json_rows(args.tasks) if args.tasks else [], args.near_threshold)
    report["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (paths | ({"tasks": args.tasks} if args.tasks else {})).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "cross_split_components": len(report["cross_split_components"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
