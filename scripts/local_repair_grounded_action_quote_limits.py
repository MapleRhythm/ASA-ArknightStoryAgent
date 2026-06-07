#!/usr/bin/env python3
"""Deterministically repair grounded_action_v1 quote-limit violations.

This is a fallback for records where API repair is slow or unstable. It only
repairs structural grounding constraints that can be checked locally:
- quote must be a continuous span in the prompt evidence
- quote length / per-fact quote count / total quote limits
- evidence_id must point at a prompt evidence block containing the quote

    It does not claim to solve semantic support beyond these local constraints.
    Quotes that cannot be located exactly in the prompt evidence are discarded,
    not replaced with heuristic substitutes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QC_PATH = PROJECT_ROOT / "scripts" / "qc_repair_grounded_action_sft_with_api.py"
spec = importlib.util.spec_from_file_location("grounded_action_qc", QC_PATH)
qc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qc)

EVIDENCE_HEADER_RE = re.compile(r"^E\d+\s+(.+?#chunk-\d+)\s*$")
STOP_LINE_PREFIXES = ("规则：", "输出", "请输出", "minirag_hints", "minirag_hints_not_evidence")
PUNCT_RE = re.compile(r"([。！？；;!?])")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_evidence_blocks(prompt: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current_id: str | None = None
    in_evidence = False
    for raw_line in str(prompt or "").splitlines():
        line = raw_line.rstrip("\n")
        if line.strip() == "证据：":
            in_evidence = True
            continue
        if not in_evidence:
            continue
        header = EVIDENCE_HEADER_RE.match(line.strip())
        if header:
            current_id = header.group(1)
            blocks.setdefault(current_id, [])
            continue
        if current_id is not None and any(line.startswith(prefix) for prefix in STOP_LINE_PREFIXES):
            current_id = None
            continue
        if current_id is not None:
            blocks[current_id].append(line)
    return {key: "\n".join(value).strip() for key, value in blocks.items()}


def normalize(text: str) -> str:
    return qc.normalize_for_match(text)


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in PUNCT_RE.finditer(text):
        end = match.end()
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    if not parts and text.strip():
        parts.append(text.strip())
    return parts


def keyword_chars(text: str) -> set[str]:
    ignored = set(" \t\r\n，。！？；：、,.!?;:\"'“”‘’（）()[]【】{}<>《》…-—_0123456789")
    return {char for char in str(text or "") if char not in ignored}


def score_text(candidate: str, hint: str) -> int:
    keys = keyword_chars(hint)
    if not keys:
        return 0
    return sum(1 for char in candidate if char in keys)


def best_window(text: str, hint: str, *, max_chars: int, preferred_chars: int = 60) -> str:
    compact = str(text or "").strip()
    if len(compact) <= max_chars:
        return compact

    sentences = split_sentences(compact)
    valid_sentences = [item for item in sentences if 0 < len(item) <= max_chars]
    if valid_sentences:
        return max(valid_sentences, key=lambda item: (score_text(item, hint), min(len(item), preferred_chars)))

    window = min(max_chars, preferred_chars)
    if len(compact) <= window:
        return compact
    best_start = 0
    best_score = -1
    step = max(1, window // 3)
    for start in range(0, max(1, len(compact) - window + 1), step):
        candidate = compact[start : start + window].strip()
        score = score_text(candidate, hint)
        if score > best_score:
            best_score = score
            best_start = start
    return compact[best_start : best_start + window].strip()


def find_quote_block(quote: str, blocks: dict[str, str], preferred_id: str = "") -> tuple[str, str] | None:
    quote_norm = normalize(quote)
    if not quote_norm:
        return None
    if preferred_id in blocks and quote_norm in normalize(blocks[preferred_id]):
        return preferred_id, blocks[preferred_id]
    for evidence_id, text in blocks.items():
        if quote_norm in normalize(text):
            return evidence_id, text
    return None


def repair_quote(
    ref: dict[str, Any],
    *,
    blocks: dict[str, str],
    hint: str,
    max_quote_chars: int,
) -> dict[str, str] | None:
    evidence_id = str(ref.get("evidence_id") or "").strip()
    quote = str(ref.get("quote") or "").strip()
    matched = find_quote_block(quote, blocks, evidence_id)
    if matched is not None:
        matched_id, _ = matched
        repaired_quote = best_window(quote, hint, max_chars=max_quote_chars)
        if normalize(repaired_quote) in normalize(blocks.get(matched_id, "")):
            return {"evidence_id": matched_id, "quote": repaired_quote}
    return None


def trim_answer_quote_total(payload: dict[str, Any], *, max_total: int) -> None:
    if max_total <= 0:
        return
    supported = payload.get("supported_facts")
    if not isinstance(supported, list):
        return

    def total_quote_chars() -> int:
        total = 0
        for fact in supported:
            for ref in fact.get("evidence_refs", []) if isinstance(fact, dict) else []:
                total += len(str(ref.get("quote") or ""))
        return total

    while total_quote_chars() > max_total:
        changed = False
        for fact in reversed(supported):
            if not isinstance(fact, dict):
                continue
            refs = fact.get("evidence_refs")
            if isinstance(refs, list) and len(refs) > 1:
                refs.pop()
                changed = True
                break
        if changed:
            continue
        if len(supported) > 1:
            supported.pop()
            continue
        break


def repair_payload(payload: dict[str, Any], prompt: str, args: argparse.Namespace) -> dict[str, Any] | None:
    normalized = qc.normalize_repaired_payload(payload)
    if normalized.get("next_action") != "answer_directly":
        return normalized
    blocks = parse_evidence_blocks(prompt)
    if not blocks:
        return None

    supported = normalized.get("supported_facts")
    if not isinstance(supported, list) or not supported:
        return None
    repaired_facts: list[dict[str, Any]] = []
    final_answer = str(normalized.get("final_answer") or "")
    for fact in supported[: args.max_supported_facts]:
        if not isinstance(fact, dict):
            continue
        fact_text = str(fact.get("fact") or "").strip()
        refs = fact.get("evidence_refs")
        if not fact_text or not isinstance(refs, list):
            continue
        repaired_refs: list[dict[str, str]] = []
        seen_refs: set[tuple[str, str]] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            hint = fact_text + "\n" + final_answer
            repaired_ref = repair_quote(ref, blocks=blocks, hint=hint, max_quote_chars=args.max_quote_chars)
            if repaired_ref is None:
                continue
            key = (repaired_ref["evidence_id"], normalize(repaired_ref["quote"]))
            if key in seen_refs:
                continue
            seen_refs.add(key)
            repaired_refs.append(repaired_ref)
            if len(repaired_refs) >= args.max_quotes_per_fact:
                break
        if repaired_refs:
            repaired_facts.append({"id": fact.get("id") or f"fact_{len(repaired_facts) + 1}", "fact": fact_text, "evidence_refs": repaired_refs})
    if not repaired_facts:
        return None
    normalized["supported_facts"] = repaired_facts
    trim_answer_quote_total(normalized, max_total=args.max_answer_quote_total_chars)
    return normalized


def load_source_records(input_dir: Path, splits: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    split_by_id: dict[str, str] = {}
    for split in splits:
        path = input_dir / split
        if not path.exists():
            continue
        for record in read_json(path):
            rid = str(record.get("id") or f"{split}:{len(records)}")
            record["id"] = rid
            records.append(record)
            split_by_id[rid] = split
    return records, split_by_id


def load_api_successes(audit_path: Path, records_by_id: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    successes: dict[str, dict[str, Any]] = {}
    if not audit_path.exists():
        return successes
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(row.get("id") or "")
        if not rid or row.get("post_validate_issues"):
            continue
        original = records_by_id.get(rid)
        if original is None:
            continue
        repaired, issues = qc.apply_audit(original, row.get("api") or {}, args=args)
        if repaired is not None and not issues:
            successes[rid] = repaired
    return successes


def validate(record: dict[str, Any], args: argparse.Namespace) -> list[str]:
    return qc.validate_payload(
        qc.parse_assistant(record),
        evidence_text=qc.user_prompt(record),
        max_quote_chars=args.max_quote_chars,
        max_quotes_per_fact=args.max_quotes_per_fact,
        max_fact_quote_total_chars=args.max_fact_quote_total_chars,
        max_supported_facts=args.max_supported_facts,
        max_answer_quote_total_chars=args.max_answer_quote_total_chars,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locally repair grounded_action_v1 quote constraints.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="grounded_action_sft_quote80_local_repaired")
    parser.add_argument("--api-audit-jsonl", type=Path, default=None, help="Optional audit.jsonl whose successful rows should be reused first.")
    parser.add_argument("--splits", default="train.json,val.json")
    parser.add_argument("--max-quote-chars", type=int, default=80)
    parser.add_argument("--max-quotes-per-fact", type=int, default=2)
    parser.add_argument("--max-fact-quote-total-chars", type=int, default=160)
    parser.add_argument("--max-supported-facts", type=int, default=6)
    parser.add_argument("--max-answer-quote-total-chars", type=int, default=400)
    parser.add_argument("--api-model", default="reused_api_audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir.is_absolute() else PROJECT_ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    audit_path = None
    if args.api_audit_jsonl is not None:
        audit_path = args.api_audit_jsonl if args.api_audit_jsonl.is_absolute() else PROJECT_ROOT / args.api_audit_jsonl
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    records, split_by_id = load_source_records(input_dir, splits)
    records_by_id = {str(record["id"]): record for record in records}
    api_successes = load_api_successes(audit_path, records_by_id, args) if audit_path is not None else {}

    stats: Counter[str] = Counter()
    output_by_id: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for record in records:
        rid = str(record["id"])
        if rid in api_successes:
            output_by_id[rid] = api_successes[rid]
            stats["api_success_used"] += 1
            continue
        cloned = qc.clone_with_rewritten_prompt(record)
        issues = validate(cloned, args)
        if not issues:
            output_by_id[rid] = cloned
            stats["local_keep_clean"] += 1
            continue
        payload = qc.parse_assistant(record)
        repaired_payload = repair_payload(payload or {}, qc.user_prompt(record), args)
        if repaired_payload is None:
            rejected.append({"id": rid, "split": split_by_id.get(rid), "issues": issues, "reason": "local_repair_failed"})
            stats["local_repair_failed"] += 1
            continue
        cloned["conversations"][-1]["value"] = qc.compact_json(qc.normalize_repaired_payload(repaired_payload))
        post_issues = validate(cloned, args)
        if post_issues:
            rejected.append({"id": rid, "split": split_by_id.get(rid), "issues": issues, "post_issues": post_issues, "reason": "post_validate_failed"})
            stats["local_repair_post_failed"] += 1
            continue
        meta = cloned.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["local_quote_repair"] = {"source_issues": issues}
        output_by_id[rid] = cloned
        stats["local_repaired"] += 1

    split_outputs: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    for record in records:
        repaired = output_by_id.get(str(record["id"]))
        if repaired is not None:
            split_outputs.setdefault(split_by_id.get(str(record["id"]), "train.json"), []).append(repaired)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in split_outputs.items():
        write_json(output_dir / split, items)
    write_json(output_dir / "dataset_info.json", qc.dataset_info(args.dataset_name, list(split_outputs)))
    if rejected:
        write_json(output_dir / "rejected.local_quote_repair.json", rejected)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "api_audit_jsonl": str(audit_path) if audit_path else None,
        "input_records": len(records),
        "kept_records": sum(len(items) for items in split_outputs.values()),
        "rejected_records": len(rejected),
        "stats": dict(stats),
        "quote_limits": {
            "max_quote_chars": args.max_quote_chars,
            "max_quotes_per_fact": args.max_quotes_per_fact,
            "max_fact_quote_total_chars": args.max_fact_quote_total_chars,
            "max_supported_facts": args.max_supported_facts,
            "max_answer_quote_total_chars": args.max_answer_quote_total_chars,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
