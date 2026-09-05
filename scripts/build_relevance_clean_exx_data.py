#!/usr/bin/env python3
"""Build relevance-aware Exx SFT/KTO data from blind set-audit judgements.

Only facts independently judged as entailed and direct/supporting are kept.
The original prediction is never overwritten; it becomes the rejected KTO
side only when a safe cleaned answer can be constructed.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    return [row for row in value if isinstance(row, dict)]


def parse_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(row.get("raw_output") or row.get("output") or "")
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prompt(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    return str(conversations[0].get("value") or "") if conversations else ""


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_record(
    source: dict[str, Any], output: dict[str, Any], *, tag: bool, suffix: str, source_kind: str
) -> dict[str, Any]:
    digest = hashlib.sha256((str(source.get("id")) + suffix + canonical(output)).encode()).hexdigest()[:16]
    return {
        "id": f"relevance_clean_{digest}_{suffix}",
        "task_type": source.get("task_type", "grounded_action_generation"),
        "system": source.get("system", ""),
        "tools": source.get("tools", ""),
        "kto_tag": tag,
        "conversations": [
            {"from": "human", "value": prompt(source)},
            {"from": "gpt", "value": canonical(output)},
        ],
        "meta": json.dumps(
            {
                "source": source_kind,
                "source_id": source.get("id"),
                "protocol": "audit_exx_set_grounding_v3",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def cleaned_payload(payload: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("next_action") != "answer_directly":
        return None
    facts = payload.get("supported_facts")
    judged = judgement.get("facts")
    if not isinstance(facts, list) or not isinstance(judged, list) or len(facts) != len(judged):
        return None
    kept = [
        fact
        for fact, audit in zip(facts, judged)
        if audit.get("support") == "entailed"
        and audit.get("question_relevance") in {"direct", "supporting"}
    ]
    if not kept:
        return None
    return {"next_action": "answer_directly", "supported_facts": kept}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    rows = load_json(args.predictions)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    results = {
        str(row.get("id")): row
        for row in audit.get("results", [])
        if isinstance(row, dict) and row.get("status") == "ok"
    }
    counts = Counter()
    sft: list[dict[str, Any]] = []
    kto: list[dict[str, Any]] = []
    for source in rows:
        original = parse_payload(source)
        judged = results.get(str(source.get("id")))
        judgement = judged.get("judgement") if judged else None
        if not original or not isinstance(judgement, dict):
            counts["missing_or_invalid_audit"] += 1
            continue
        repaired = cleaned_payload(original, judgement)
        if repaired is None:
            counts["excluded_no_safe_answer"] += 1
            continue
        sft.append(make_record(source, repaired, tag=True, suffix="chosen", source_kind="relevance_clean"))
        counts["sft_chosen"] += 1
        if canonical(repaired) != canonical(original):
            kto.extend(
                [
                    make_record(source, repaired, tag=True, suffix="chosen", source_kind="relevance_clean"),
                    make_record(source, original, tag=False, suffix="rejected", source_kind="original_rejected"),
                ]
            )
            counts["kto_pairs"] += 1
        else:
            counts["unchanged_safe"] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows_out in (("sft.json", sft), ("kto.json", kto)):
        (args.out_dir / name).write_text(
            json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "predictions": str(args.predictions),
        "audit": str(args.audit),
        "counts": dict(sorted(counts.items())),
        "sft_rows": len(sft),
        "kto_rows": len(kto),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
