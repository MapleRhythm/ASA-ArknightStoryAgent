#!/usr/bin/env python3
"""Build split-aware Exx candidates from independently checked set judgements.

Deleting bad facts does not prove that the remaining answer is complete or
coherent. Changed answers are staged for a second set audit, never promoted
straight into SFT/KTO. Validation/test/unknown sources are diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import difflib
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exx_set_grounding import validate_judgement
from build_exx_eval_split import normalize_question, record_question, sha256_file
from glm_exx_semantic_reward import extract_judge_context, payload_is_judge_eligible


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"non-object row in {path}")
    return value


def parse_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = str(row.get("raw_output") or row.get("output") or "")
    try:
        value = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def prompt(row: dict[str, Any]) -> str:
    conversations = row.get("conversations") or []
    return str(conversations[0].get("value") or "") if conversations else ""


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_record(
    source: dict[str, Any], output: dict[str, Any], *, tag: bool, suffix: str, source_kind: str,
    source_split: str = "unknown", training_eligible: bool = False,
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
                "source_split": source_split,
                "training_eligible": training_eligible,
                "requires_reaudit": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def cleaned_payload(payload: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any] | None:
    """Return only an unchanged, complete and jointly supported answer.

    A fact-level filter alone cannot certify coverage or cross-fact reference
    resolution after deletion. See ``propose_repair`` for non-training output.
    """
    if payload.get("next_action") != "answer_directly":
        return None
    try:
        validate_judgement(judgement, payload)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        judgement.get("set_support") != "complete"
        or judgement.get("context_sufficiency") != "sufficient"
        or judgement.get("action_appropriateness") != "appropriate"
    ):
        return None
    return payload


def propose_repair(payload: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any] | None:
    """Produce a candidate requiring a *new* set-level audit before training."""
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
        and audit.get("citation_complete") is True
        and audit.get("question_relevance") in {"direct", "supporting"}
    ]
    if not kept:
        return None
    return {"next_action": "answer_directly", "supported_facts": kept}


def source_ids(row: dict[str, Any]) -> set[str]:
    ids = {str(row["id"])} if row.get("id") else set()
    meta = row.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if isinstance(meta, dict) and meta.get("source_id"):
        ids.add(str(meta["source_id"]))
    return ids


def heldout_overlap(
    row: dict[str, Any], heldout: list[dict[str, Any]], threshold: float = 0.88
) -> bool:
    key = normalize_question(record_question(row))
    ids = source_ids(row)
    questions = {normalize_question(record_question(other)) for other in heldout}
    questions.discard("")
    return (
        any(ids & source_ids(other) for other in heldout)
        or key in questions
        or bool(key and difflib.get_close_matches(key, sorted(questions), n=1, cutoff=threshold))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source-split", choices=("train", "validation", "test", "unknown"), default="unknown",
        help="Only train sources can produce training filenames; other splits are diagnostic-only.",
    )
    parser.add_argument(
        "--heldout-json", type=Path, action="append", default=[],
        help="Required for train sources. Excludes matching IDs, questions and near-duplicates.",
    )
    args = parser.parse_args()
    if args.source_split == "train" and not args.heldout_json:
        parser.error("train sources require --heldout-json leakage checks")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    rows = load_json(args.predictions)
    heldout = [row for path in args.heldout_json for row in load_json(path)]
    if args.source_split == "train" and (
        not heldout or any(not record_question(row) for row in heldout)
    ):
        parser.error("heldout data must be non-empty and every row must have a question")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    audit_rows = audit.get("results", [])
    results = {}
    for row in audit_rows:
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError("audit rows must have non-empty IDs")
        key = str(row["id"])
        if key in results:
            raise ValueError(f"duplicate audit ID: {key}")
        results[key] = row
    if any(not row.get("id") for row in rows) or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("prediction IDs must be non-empty and unique")
    counts = Counter()
    sft: list[dict[str, Any]] = []
    kto: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for source in rows:
        if not record_question(source):
            counts["excluded_missing_question"] += 1
            continue
        if heldout_overlap(source, heldout):
            counts["excluded_heldout_overlap"] += 1
            continue
        original = parse_payload(source)
        judged = results.get(str(source.get("id")))
        judgement = judged.get("judgement") if judged else None
        if (
            not original or not judged or judged.get("status") != "ok"
            or not isinstance(judgement, dict)
        ):
            counts["missing_or_invalid_audit"] += 1
            continue
        try:
            validate_judgement(judgement, original)
            context = extract_judge_context(prompt(source))
            if not payload_is_judge_eligible(original, context["evidence"]):
                raise ValueError("invalid_prediction_protocol")
        except (KeyError, TypeError, ValueError):
            counts["excluded_protocol_or_binding_mismatch"] += 1
            continue
        accepted = cleaned_payload(original, judgement)
        # Keep independently appropriate non-answer actions too. An answer-only
        # recipe teaches premature answers and hides the sufficiency boundary.
        if (
            original.get("next_action") in {"retrieve_more", "abstain"}
            and judgement.get("context_sufficiency") == "insufficient"
            and judgement.get("action_appropriateness") == "appropriate"
        ):
            accepted = original
        if accepted is not None:
            sft.append(
                make_record(
                    source, accepted, tag=True, suffix="chosen", source_kind="relevance_verified",
                    source_split=args.source_split, training_eligible=args.source_split == "train",
                )
            )
            counts["sft_chosen"] += 1
            counts[f"accepted_action:{original['next_action']}"] += 1
        else:
            candidate = propose_repair(original, judgement)
            if candidate is None:
                counts["excluded_no_safe_subset"] += 1
            else:
                repairs.append({
                    **source,
                    "raw_output": canonical(candidate),
                    "repair_provenance": {
                        "original_output": canonical(original),
                        "source_split": args.source_split,
                        "training_eligible": False,
                        "requires_reaudit": True,
                        "original_set_support": judgement.get("set_support"),
                        "audit_sha256": sha256_file(args.audit),
                    },
                })
                counts["staged_for_set_reaudit"] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.source_split == "train" else "diagnostic_"
    for name, rows_out in (
        (f"{prefix}sft.json", sft), (f"{prefix}kto.json", kto), ("repair_candidates.json", repairs)
    ):
        (args.out_dir / name).write_text(
            json.dumps(rows_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "predictions": str(args.predictions),
        "audit": str(args.audit),
        "counts": dict(sorted(counts.items())),
        "sft_rows": len(sft),
        "kto_rows": len(kto),
        "repair_candidates": len(repairs),
        "source_split": args.source_split,
        "training_eligible": args.source_split == "train",
        "policy": "No subset promotion; edited answers require a fresh set audit. No inferred KTO pairs.",
        "source_sha256": sha256_file(args.predictions),
        "audit_sha256": sha256_file(args.audit),
        "heldout_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.heldout_json
        ],
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
