#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_prompt(record: dict[str, Any]) -> str:
    query = str(record.get("query") or "").strip()
    query_type = str(record.get("query_type") or "").strip()
    answer_focus = str(record.get("answer_focus") or "").strip()
    parts = [
        "判断哪条证据链更能充分、直接、完整回答用户问题。",
        f"用户问题：{query}",
    ]
    if query_type:
        parts.append(f"问题类别：{query_type}")
    if answer_focus:
        parts.append(f"答案焦点：{answer_focus}")
    parts.append("优先选择包含答案揭示点、因果/时序完整、背景噪声更少的证据链。")
    return "\n".join(parts)


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = build_prompt(record)
    chosen = str(record.get("positive") or "").strip()
    rejected = str(record.get("negative") or "").strip()
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "query": record.get("query", ""),
            "query_type": record.get("query_type", ""),
            "source_name": record.get("source_name", ""),
            "negative_type": record.get("negative_type", ""),
            "positive_score": record.get("positive_score"),
            "negative_score": record.get("negative_score"),
            "positive_chain": record.get("positive_chain", []),
            "negative_chain": record.get("negative_chain", []),
            "positive_chain_role_tags": record.get("positive_chain_role_tags", []),
            "negative_chain_role_tags": record.get("negative_chain_role_tags", []),
            "positive_chain_structure": record.get("positive_chain_structure", {}),
            "negative_chain_structure": record.get("negative_chain_structure", {}),
            "answer": record.get("answer", ""),
            "answer_evidence": record.get("answer_evidence", []),
            "answer_focus": record.get("answer_focus", ""),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert evidence-chain reranker_pairwise.jsonl into prompt/chosen/rejected DPO JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v1/reranker_pairwise.jsonl"),
        help="Path to reranker_pairwise.jsonl exported by scripts/evidence_chain_dataset.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/evidence_chain_reranker/batch_v1/reranker_dpo.jsonl"),
        help="Output DPO JSONL path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    records = read_jsonl(input_path)
    converted = [item for record in records if (item := convert_record(record)) is not None]
    write_jsonl(output_path, converted)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "input_records": len(records),
                "dpo_records": len(converted),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
