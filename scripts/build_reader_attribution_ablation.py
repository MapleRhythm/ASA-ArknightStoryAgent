#!/usr/bin/env python3
"""Build a development-smoke, content-preserving reader attribution ablation.

No references or reviewer notes enter generation messages. The baseline prompt
is byte-identical to the frozen reader diagnostic. The two interventions retain
every non-whitespace character, E-ID, document order and evidence budget; only
dialogue layout and then generic reading instructions change. This is evaluation
data, not a training export.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reader_evidence_diagnostic import documents_from_snapshot, read_rows
from asa_arknight_story_agent.inference.evidence.texts import document_clean_text
from asa_arknight_story_agent.inference.evidence.rendering import (
    evidence_id_text_map_from_prompt,
)


VARIANTS = ("flat", "dialogue_lines", "dialogue_lines_attribution")
READING_GUIDANCE = (
    "hypothesis只是检索线索，不是事实证据。逐段区分说话人、被谈论的人和叙述者；"
    "下一行未重复姓名不表示换了说话人，不能把被反驳的观点归给反驳者。"
    "区分角色的猜测、敷衍自称、公开表态与原文确认的身份或私下态度；"
    "不要把职业当成种族，也不要把寻找者与被寻找者、原因与结果倒置。"
    "保留否定和转折的含义，逐项回答问题的所有子问；"
    "证据明确支持全部所问要点时直接回答，无法由原文消解的关系不要补写。"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nonwhitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def evidence_region(prompt: str) -> tuple[str, str, str]:
    prefix, marker, rest = prompt.partition("\nevidence:\n")
    if not marker:
        raise ValueError("expected canonical evidence section")
    body, marker_end, suffix = rest.partition("\noutput_schema:")
    if not marker_end:
        raise ValueError("expected canonical output_schema boundary")
    return prefix + marker, body, marker_end + suffix


def build_rows(
    matrix: list[dict[str, Any]], snapshots: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row["id"]): row for row in snapshots}
    if len(by_id) != len(snapshots):
        raise ValueError("duplicate snapshot ID")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in matrix:
        diagnostic = row["diagnostic"]
        source_condition = str(diagnostic["condition"])
        if source_condition not in {"minimal_sufficient", "sufficient_with_distractors"}:
            continue
        source_id = str(diagnostic["source_id"])
        documents, _ = documents_from_snapshot(by_id[source_id])
        prompt = str(row["conversations"][0]["value"])
        prefix, original_body, suffix = evidence_region(prompt)
        mapping = diagnostic["evidence_id_to_corpus_id"]
        original_map = evidence_id_text_map_from_prompt(original_body)
        if list(mapping) != list(original_map):
            raise ValueError(f"{row['id']}: E-ID/order mismatch")
        blocks = []
        for eid, doc_id in mapping.items():
            raw = document_clean_text(documents[doc_id]).replace("\r\n", "\n")
            if re.sub(r"\s+", " ", raw).strip() != original_map[eid]:
                raise ValueError(f"{row['id']}/{eid}: frozen text differs from source")
            # Never infer or insert speaker names in the original transcript.
            blocks.append(f"[{eid}]\n{raw}")
        lined_body = "\n".join(blocks)
        if nonwhitespace(lined_body) != nonwhitespace(original_body):
            raise ValueError(f"{row['id']}: intervention changed evidence content")
        for variant in VARIANTS:
            out = copy.deepcopy(row)
            out["id"] = f"{row['id']}::{variant}"
            if out["id"] in seen:
                raise ValueError("duplicate matrix ID")
            seen.add(out["id"])
            body = original_body if variant == "flat" else lined_body
            new_prompt = prefix + body + suffix
            if variant == "dialogue_lines_attribution":
                new_prompt += "\nreading_guidance: " + READING_GUIDANCE
            out["conversations"][0]["value"] = new_prompt
            out["diagnostic"].update(
                {
                    "condition": f"{source_condition}::{variant}",
                    "source_condition": source_condition,
                    "source_matrix_id": row["id"],
                    "variant": variant,
                    "evaluation_only": True,
                    "evidence_chars": len(body),
                    "evidence_nonwhitespace_sha256": sha256_text(nonwhitespace(body)),
                    "generation_prompt_sha256": sha256_text(new_prompt),
                    "evidence_text_mode": "direct_document",
                }
            )
            rows.append(out)
            counts[out["diagnostic"]["condition"]] += 1
    if not rows:
        raise ValueError("no eligible diagnostic rows")
    return rows, {
        "protocol": "asa_reader_attribution_ablation_v1",
        "evaluation_only": True,
        "rows": len(rows),
        "question_families": len({r["diagnostic"]["question_family"] for r in rows}),
        "conditions": dict(sorted(counts.items())),
        "baseline_prompt_identical": True,
        "evidence_nonwhitespace_identical": True,
        "reference_answers_in_generation_prompt": False,
        "rules": READING_GUIDANCE,
        "limitations": [
            "Strict9 is an already-inspected development smoke, not a fresh blind test.",
            "Only fixed-evidence Exx reading is evaluated, not production quote/e2e.",
            "Generic guidance was motivated by known failure types; independent validation is required.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists():
        parser.error("refusing to overwrite ablation artifacts")
    rows, report = build_rows(read_rows(args.matrix), read_rows(args.snapshots))
    report["inputs"] = {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (("matrix", args.matrix), ("snapshots", args.snapshots))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
