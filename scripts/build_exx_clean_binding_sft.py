#!/usr/bin/env python3
"""Materialize a versioned Exx SFT dataset from GLM-recalibrated JSONL.

The recalibration job writes one JSON object per source row.  Llama-Factory
expects JSON arrays plus a dataset_info.json registry, so this utility makes a
new, immutable dataset directory without touching the original data or model
artifacts.  It also performs the same structural checks used by the runtime
validator, including prompt-local E-ID bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EVIDENCE_RE = re.compile(r"^\[(E\d+)\]\s*$", re.MULTILINE)
PROTOCOL = "grounded_action_exx_v1"
SYSTEM = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
FORBIDDEN = {"quote", "final_answer", "inferred_facts", "evidence_refs", "answer"}
QUESTION_RE = re.compile(r"^question:\s*(.+)$", re.MULTILINE)
HYPOTHESIS_RE = re.compile(r"^hypothesis:\s*(.+)$", re.MULTILINE)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_no}")
            rows.append(value)
    return rows


def payload(row: dict[str, Any]) -> dict[str, Any]:
    conversations = row.get("conversations") or []
    if len(conversations) < 2:
        raise ValueError(f"{row.get('id')}: missing conversations")
    value = conversations[-1].get("value")
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, dict):
        raise ValueError(f"{row.get('id')}: assistant payload is not an object")
    return value


def visible_eids(row: dict[str, Any]) -> set[str]:
    conversations = row.get("conversations") or []
    prompt = str(conversations[0].get("value") or "") if conversations else ""
    return set(EVIDENCE_RE.findall(prompt))


def normalize_legacy_follow_up(row: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Turn the recalibrator's legacy string fallback into the runtime shape.

    Older finalize runs emitted a plain explanatory string when all facts were
    removed.  The Exx protocol requires a hypothesis object.  We reuse the
    model-visible question/hypothesis as a conservative query seed and add a
    short explicit missing-evidence request; this changes only the new clean
    dataset, never the archived recalibration artifact.
    """
    if out.get("next_action") != "retrieve_more" or not isinstance(
        out.get("follow_up_hypothesis"), str
    ):
        return out
    conversations = row.get("conversations") or []
    prompt = str(conversations[0].get("value") or "") if conversations else ""
    question_match = QUESTION_RE.search(prompt)
    hypothesis_match = HYPOTHESIS_RE.search(prompt)
    question = question_match.group(1).strip() if question_match else ""
    hypothesis: dict[str, Any] = {}
    if hypothesis_match:
        try:
            parsed = json.loads(hypothesis_match.group(1))
            if isinstance(parsed, dict):
                hypothesis = parsed
        except json.JSONDecodeError:
            pass
    allowed = ("query_type", "entities", "keywords", "expected_answer_type", "dialogue_context")
    follow_up = {key: hypothesis[key] for key in allowed if key in hypothesis}
    follow_up.setdefault("query_type", "fact")
    follow_up.setdefault("entities", [])
    follow_up.setdefault("keywords", [])
    follow_up.setdefault("expected_answer_type", "")
    follow_up.setdefault("dialogue_context", "")
    text = str(out.get("follow_up_hypothesis") or "").strip()
    follow_up["question"] = (
        f"{question}（请检索能直接证明问题核心事实的证据）" if question else text
    )
    normalized = dict(out)
    normalized["follow_up_hypothesis"] = follow_up
    return normalized


def validate_row(row: dict[str, Any]) -> tuple[str, int]:
    """Return (action, fact_count), raising on a structurally unsafe row."""
    if not row.get("id"):
        raise ValueError("row without id")
    task_type = str(row.get("task_type") or "")
    if task_type != "grounded_action_generation":
        return "non_grounded", 0
    if row.get("system") != SYSTEM:
        raise ValueError(f"{row.get('id')}: non-canonical system")
    conversations = row.get("conversations") or []
    prompt = str(conversations[0].get("value") or "") if conversations else ""
    if f"output_schema: {PROTOCOL}" not in prompt:
        raise ValueError(f"{row.get('id')}: wrong protocol")
    visible = visible_eids(row)
    if not visible:
        raise ValueError(f"{row.get('id')}: no visible evidence ids")
    out = normalize_legacy_follow_up(row, payload(row))
    if FORBIDDEN.intersection(out):
        raise ValueError(f"{row.get('id')}: legacy output fields present")
    action = out.get("next_action")
    if action not in {"answer_directly", "retrieve_more", "abstain"}:
        raise ValueError(f"{row.get('id')}: invalid action {action!r}")
    if action == "answer_directly":
        if set(out) != {"next_action", "supported_facts"}:
            raise ValueError(f"{row.get('id')}: answer schema")
        facts = out.get("supported_facts")
        if not isinstance(facts, list) or not 1 <= len(facts) <= 8:
            raise ValueError(f"{row.get('id')}: fact count")
        for index, fact in enumerate(facts, 1):
            if not isinstance(fact, dict) or set(fact) != {"fact", "evidence_ids"}:
                raise ValueError(f"{row.get('id')}: fact {index} schema")
            ids = fact.get("evidence_ids")
            if (
                not str(fact.get("fact") or "").strip()
                or not isinstance(ids, list)
                or not 1 <= len(ids) <= 2
                or len({str(item) for item in ids}) != len(ids)
                or any(str(item) not in visible for item in ids)
            ):
                raise ValueError(f"{row.get('id')}: fact {index} binding")
        return action, len(facts)
    if action == "retrieve_more":
        follow_up = out.get("follow_up_hypothesis")
        if set(out) != {"next_action", "follow_up_hypothesis"} or not isinstance(follow_up, dict):
            raise ValueError(f"{row.get('id')}: retrieve schema")
        if not str(follow_up.get("question") or "").strip():
            raise ValueError(f"{row.get('id')}: empty follow-up question")
    else:
        if set(out) != {"next_action", "reason"} or not str(out.get("reason") or "").strip():
            raise ValueError(f"{row.get('id')}: abstain schema")
    return action, 0


def registry(prefix: str) -> dict[str, Any]:
    common = {
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
            "function_tag": "function_call",
        },
    }
    return {
        f"{prefix}_{split}": {"file_name": f"{split}.json", **common}
        for split in ("train", "val", "test")
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="exx_binding_clean_sft")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "source_dir": str(args.strict_dir),
        "output_dir": str(args.out_dir),
        "prefix": args.prefix,
        "splits": {},
        "validation_errors": [],
    }
    for split in ("train", "val", "test"):
        source = args.strict_dir / f"gold_recalibrated_v1_strict_{split}.jsonl"
        rows = read_jsonl(source) if source.exists() else []
        actions = Counter()
        task_types = Counter()
        facts = 0
        for row in rows:
            task_types[str(row.get("task_type") or "")] += 1
            try:
                action, fact_count = validate_row(row)
            except ValueError as exc:
                report["validation_errors"].append({"split": split, "error": str(exc)})
                continue
            if action != "non_grounded":
                actions[action] += 1
                facts += fact_count
        # Arrow columns are homogeneous; retain metadata but serialize it.
        converted: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            if row.get("task_type") == "grounded_action_generation":
                conversations = list(row.get("conversations") or [])
                if len(conversations) >= 2:
                    try:
                        normalized = normalize_legacy_follow_up(row, payload(row))
                        if normalized != payload(row):
                            conversations[-1] = dict(conversations[-1])
                            conversations[-1]["value"] = compact(normalized)
                            row["conversations"] = conversations
                    except (TypeError, ValueError, json.JSONDecodeError):
                        # Validation above records the actionable error; keep
                        # the original row for auditability.
                        pass
            for key in ("tools", "meta"):
                if not isinstance(row.get(key), str):
                    row[key] = compact(row.get(key, ""))
            converted.append(row)
        target = args.out_dir / f"{split}.json"
        target.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["splits"][split] = {
            "rows": len(converted),
            "task_types": dict(task_types),
            "actions": dict(actions),
            "supported_fact_count": facts,
            "sha256": sha256(target),
            "source": str(source),
        }

    (args.out_dir / "dataset_info.json").write_text(
        json.dumps(registry(args.prefix), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["validation_error_count"] = len(report["validation_errors"])
    (args.out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "validation_errors"}, ensure_ascii=False, indent=2))
    if report["validation_errors"]:
        print(json.dumps(report["validation_errors"][:20], ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
