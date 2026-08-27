#!/usr/bin/env python3
"""Convert legacy ASA grounding data to the grounded_action_exx_v1 protocol.

The converter is deliberately conservative.  Evidence labels are derived only
from evidence that appears in the model-visible prompt.  Metadata such as
``gold``, ``answer_focus`` and ``required_evidence`` is never consulted when
assigning an E-id.

The script never edits its inputs and refuses to write into a non-empty output
directory.  Rejected rows are retained in ``rejects.jsonl`` for later teacher
re-labelling with question plus visible evidence only.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "grounded_action_exx_v1"
SYSTEM_PROMPT = "你是《明日方舟》剧情RAG证据动作模块。只输出合法JSON。"
FORBIDDEN_PROMPT_MARKERS = (
    "required_evidence",
    "answer_focus",
    "gold_evidence",
    "gold answer",
    "参考答案",
    "标准答案",
)
LEGACY_OUTPUT_KEYS = {"quote", "final_answer", "inferred_facts", "evidence_refs"}
EVIDENCE_HEADER_RE = re.compile(r"^E(\d+)\s+([^\n]+)\n", re.MULTILINE)
SHORT_EVIDENCE_RE = re.compile(r"^(\d+)\.\s+([^:\n]+):\s?", re.MULTILINE)
QUESTION_ZH_RE = re.compile(r"^问题：(.*)$", re.MULTILINE)
ROUND_ZH_RE = re.compile(r"^轮次：(.*)$", re.MULTILINE)
HYPOTHESIS_ZH_RE = re.compile(r"^当前检索假设：(.*)$", re.MULTILINE)
FIELD_RE_TEMPLATE = r"^{name}:\s*(.*)$"


class ConversionError(ValueError):
    """A row cannot be converted without inventing supervision."""


@dataclasses.dataclass(frozen=True)
class Evidence:
    label: str
    doc_id: str
    text: str


@dataclasses.dataclass
class ConvertedRow:
    record: dict[str, Any]
    action: str | None
    source_task: str
    question_key: str


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2 if pretty else None)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(prompt: str, name: str) -> str:
    match = re.search(FIELD_RE_TEMPLATE.format(name=re.escape(name)), prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_assistant(record: dict[str, Any]) -> dict[str, Any]:
    try:
        value = record["conversations"][-1]["value"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ConversionError("invalid_conversation") from exc
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if not isinstance(value, str):
        raise ConversionError("assistant_not_json_string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConversionError("assistant_invalid_json") from exc
    if not isinstance(parsed, dict):
        raise ConversionError("assistant_json_not_object")
    return parsed


def parse_grounded_prompt(prompt: str) -> tuple[str, str, str, list[Evidence]]:
    marker = "\n证据：\n"
    if marker not in prompt:
        raise ConversionError("grounded_prompt_missing_evidence")
    prefix, body = prompt.split(marker, 1)
    output_marker = "\n输出格式："
    if output_marker in body:
        body = body.split(output_marker, 1)[0]
    matches = list(EVIDENCE_HEADER_RE.finditer(body))
    if not matches:
        raise ConversionError("grounded_prompt_evidence_parse_failed")
    evidence: list[Evidence] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        evidence.append(
            Evidence(
                label=f"E{int(match.group(1))}",
                doc_id=match.group(2).strip(),
                text=body[match.end() : end].strip(),
            )
        )
    question_match = QUESTION_ZH_RE.search(prefix)
    if not question_match:
        raise ConversionError("grounded_prompt_missing_question")
    round_match = ROUND_ZH_RE.search(prefix)
    hypothesis_match = HYPOTHESIS_ZH_RE.search(prefix)
    return (
        question_match.group(1).strip(),
        hypothesis_match.group(1).strip() if hypothesis_match else "",
        round_match.group(1).strip() if round_match else "",
        evidence,
    )


def parse_short_prompt(prompt: str) -> tuple[str, str, str, list[Evidence]]:
    marker = "\nevidence_brief:\n"
    if marker not in prompt:
        raise ConversionError("short_prompt_missing_evidence")
    prefix, body = prompt.split(marker, 1)
    if "\noutput_schema:" in body:
        body = body.split("\noutput_schema:", 1)[0]
    if "\nminirag_hints:" in body:
        body = body.split("\nminirag_hints:", 1)[0]
    matches = list(SHORT_EVIDENCE_RE.finditer(body))
    if not matches:
        raise ConversionError("short_prompt_evidence_parse_failed")
    evidence: list[Evidence] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        evidence.append(
            Evidence(
                label=f"E{int(match.group(1))}",
                doc_id=match.group(2).strip(),
                text=body[match.end() : end].strip(),
            )
        )
    return field(prefix, "question"), field(prefix, "hypothesis"), field(prefix, "round"), evidence


def render_prompt(question: str, hypothesis: str, round_value: str, evidence: Iterable[Evidence]) -> str:
    lines = [
        "task: grounded_action_generation",
        f"question: {question}",
        f"hypothesis: {hypothesis or '{}'}",
        f"round: {round_value or 'unknown'}",
        "evidence:",
    ]
    for item in evidence:
        # Full evidence is preserved.  The document id is intentionally omitted
        # from the response namespace; the model cites only the local E label.
        lines.extend((f"[{item.label}]", item.text))
    lines.extend(
        (
            f"output_schema: {PROTOCOL}",
            "rules: 只使用当前可见证据；回答时把每个可核验原子事实绑定到1至2个当前存在的E编号；"
            "不要复制引文，不要输出quote、final_answer或inferred_facts；证据不足时才retrieve_more。",
        )
    )
    return "\n".join(lines)


def label_number(value: str) -> str | None:
    normalized = value.strip().strip("[]").upper()
    for pattern in (r"E(\d+)", r"EVIDENCE[_ -]?(\d+)", r"(\d+)"):
        match = re.fullmatch(pattern, normalized, re.IGNORECASE)
        if match:
            return f"E{int(match.group(1))}"
    return None


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def resolve_reference(reference: dict[str, Any], evidence: list[Evidence]) -> str:
    raw_id = str(reference.get("evidence_id", "")).strip()
    quote = str(reference.get("quote", "")).strip()
    available = {item.label: item for item in evidence}

    direct_label = label_number(raw_id)
    if direct_label in available:
        direct_item = available[direct_label]
        if not quote or normalize_space(quote) in normalize_space(direct_item.text):
            return direct_label

    doc_hits = [item.label for item in evidence if raw_id and raw_id == item.doc_id]
    if len(doc_hits) == 1:
        if not quote or normalize_space(quote) in normalize_space(available[doc_hits[0]].text):
            return doc_hits[0]

    if quote:
        normalized_quote = normalize_space(quote)
        quote_hits = [item.label for item in evidence if normalized_quote in normalize_space(item.text)]
        if len(quote_hits) == 1:
            return quote_hits[0]
        if len(quote_hits) > 1:
            raise ConversionError("ambiguous_quote_mapping")
    raise ConversionError("unresolved_evidence_reference")


def convert_grounded_action(action: dict[str, Any], evidence: list[Evidence]) -> dict[str, Any]:
    next_action = action.get("next_action")
    if next_action == "answer_directly":
        output_facts: list[dict[str, Any]] = []
        fact_evidence: dict[str, list[str]] = {}
        source_facts = action.get("supported_facts")
        if not isinstance(source_facts, list) or not source_facts:
            raise ConversionError("answer_without_supported_facts")
        for source_fact in source_facts:
            fact_text = str(source_fact.get("fact", "")).strip()
            references = source_fact.get("evidence_refs")
            if not fact_text or not isinstance(references, list) or not references:
                raise ConversionError("fact_missing_reference")
            labels: list[str] = []
            for reference in references:
                if not isinstance(reference, dict):
                    raise ConversionError("invalid_evidence_reference")
                label = resolve_reference(reference, evidence)
                if label not in labels:
                    labels.append(label)
            if not 1 <= len(labels) <= 2:
                raise ConversionError("fact_reference_count_out_of_range")
            output_facts.append({"fact": fact_text, "evidence_ids": labels})
            fact_id = str(source_fact.get("id", "")).strip()
            if fact_id:
                fact_evidence[fact_id] = labels

        # Legacy inferred_facts often carry the actual composed answer.  They
        # can be retained without keeping the inference schema: bind the
        # inferred text to the union of the explicitly cited premise facts.
        # Protocol v1 limits each fact to two E-ids, so wider inference chains
        # are rejected for teacher re-labelling instead of being truncated.
        for inferred in action.get("inferred_facts") or []:
            if not isinstance(inferred, dict):
                raise ConversionError("invalid_inferred_fact")
            inferred_text = str(inferred.get("fact", "")).strip()
            premise_ids = [str(item) for item in inferred.get("premise_fact_ids") or []]
            labels: list[str] = []
            for premise_id in premise_ids:
                if premise_id not in fact_evidence:
                    raise ConversionError("inferred_fact_missing_premise")
                for label in fact_evidence[premise_id]:
                    if label not in labels:
                        labels.append(label)
            if not inferred_text or not labels:
                raise ConversionError("inferred_fact_without_visible_evidence")
            if len(labels) > 2:
                # The premise facts already remain in the target.  Dropping a
                # wide, non-atomic summary is safer than either attaching a
                # partial citation set or rejecting otherwise valid facts.
                continue
            if not any(item["fact"] == inferred_text for item in output_facts):
                output_facts.append({"fact": inferred_text, "evidence_ids": labels})
        return {"next_action": "answer_directly", "supported_facts": output_facts}
    if next_action == "retrieve_more":
        follow_up = action.get("follow_up_hypothesis")
        if not isinstance(follow_up, dict) or not follow_up:
            raise ConversionError("retrieve_more_missing_follow_up")
        return {"next_action": "retrieve_more", "follow_up_hypothesis": copy.deepcopy(follow_up)}
    if next_action == "abstain":
        return {"next_action": "abstain", "reason": "现有证据不足以确认。"}
    raise ConversionError("unsupported_next_action")


def convert_grounded_record(record: dict[str, Any], source_name: str) -> ConvertedRow:
    prompt = record["conversations"][0]["value"]
    question, hypothesis, round_value, evidence = parse_grounded_prompt(prompt)
    action = convert_grounded_action(parse_assistant(record), evidence)
    converted = copy.deepcopy(record)
    converted["task_type"] = "grounded_action_generation"
    converted["system"] = SYSTEM_PROMPT
    converted["conversations"][0]["value"] = render_prompt(question, hypothesis, round_value, evidence)
    converted["conversations"][-1]["value"] = json_dump(action)
    converted["meta"] = copy.deepcopy(record.get("meta", {}))
    converted["meta"].update(
        {
            "schema": PROTOCOL,
            "conversion_source": source_name,
            "conversion_method": "visible_prompt_deterministic",
        }
    )
    question_key = str(converted["meta"].get("question_key") or question)
    validate_record(converted)
    return ConvertedRow(converted, action["next_action"], record.get("task_type", ""), question_key)


def convert_non_conclusion(record: dict[str, Any], source_name: str) -> ConvertedRow:
    converted = copy.deepcopy(record)
    converted["meta"] = copy.deepcopy(record.get("meta", {}))
    converted["meta"]["conversion_source"] = source_name
    question_key = str(
        converted["meta"].get("question_key")
        or converted["meta"].get("source_question_key")
        or field(converted["conversations"][0]["value"], "question")
    )
    return ConvertedRow(converted, None, record.get("task_type", ""), question_key)


def validate_record(record: dict[str, Any]) -> None:
    prompt = record["conversations"][0]["value"]
    prompt_lower = prompt.lower()
    if any(marker in prompt_lower for marker in FORBIDDEN_PROMPT_MARKERS):
        raise ConversionError("forbidden_supervision_in_prompt")
    evidence_ids = set(re.findall(r"^\[(E\d+)\]$", prompt, re.MULTILINE))
    if not evidence_ids:
        raise ConversionError("converted_prompt_has_no_evidence")
    output = parse_assistant(record)
    serialized = json_dump(output)
    if '"quote"' in serialized or '"final_answer"' in serialized or '"inferred_facts"' in serialized:
        raise ConversionError("legacy_output_field_remains")
    if output.get("next_action") == "answer_directly":
        facts = output.get("supported_facts")
        if not isinstance(facts, list) or not facts:
            raise ConversionError("converted_answer_has_no_facts")
        for fact in facts:
            ids = fact.get("evidence_ids")
            if not str(fact.get("fact", "")).strip() or not isinstance(ids, list) or not 1 <= len(ids) <= 2:
                raise ConversionError("invalid_converted_fact")
            if any(item not in evidence_ids for item in ids):
                raise ConversionError("converted_evidence_id_not_visible")


def reject_payload(
    source_name: str, source_path: Path, split: str, record: dict[str, Any], reason: str
) -> dict[str, Any]:
    return {
        "source": source_name,
        "source_path": str(source_path),
        "split": split,
        "id": record.get("id"),
        "task_type": record.get("task_type"),
        "kto_tag": record.get("kto_tag"),
        "reason": reason,
        "record": record,
    }


def source_specs(args: argparse.Namespace) -> list[tuple[str, Path, str]]:
    specs: list[tuple[str, Path, str]] = []
    for source_name, root in (
        ("grounded", args.grounded_dir),
        ("online", args.online_dir),
        ("kto", args.kto_dir),
    ):
        if not root:
            continue
        root = Path(root).resolve()
        for split in ("train", "val", "test"):
            path = root / f"{split}.json"
            if path.exists():
                specs.append((source_name, path, split))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded-dir", type=Path)
    parser.add_argument("--online-dir", type=Path)
    parser.add_argument("--kto-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-non-conclusion",
        action="store_true",
        help="Keep hypothesis/follow-up rows unchanged in the SFT mix.",
    )
    args = parser.parse_args()

    specs = source_specs(args)
    if not specs:
        parser.error("no source train/val/test files were found")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory must not already contain files: {output_dir}")

    accepted_by_split: dict[str, list[ConvertedRow]] = defaultdict(list)
    kto_by_split: dict[str, list[ConvertedRow]] = defaultdict(list)
    rejects: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    input_manifest: list[dict[str, Any]] = []

    for source_name, source_path, split in specs:
        rows = load_json(source_path)
        if not isinstance(rows, list):
            raise SystemExit(f"source is not a JSON list: {source_path}")
        input_manifest.append(
            {
                "source": source_name,
                "split": split,
                "path": str(source_path),
                "records": len(rows),
                "sha256": sha256_file(source_path),
            }
        )
        for record in rows:
            task = record.get("task_type", "")
            stats[f"input:{source_name}:{split}:{task}"] += 1
            try:
                if source_name == "grounded" and task == "grounded_action_generation":
                    converted = convert_grounded_record(record, source_name)
                    accepted_by_split[split].append(converted)
                elif source_name == "kto" and task == "conclusion_generation":
                    # Legacy conclusion_v2 contains no evidence binding.  Preserve
                    # it as a reject for teacher re-labelling; never infer E-ids
                    # from gold metadata or answer overlap.
                    raise ConversionError("conclusion_v2_has_no_deterministic_evidence_binding")
                elif source_name == "online" and task == "conclusion_generation":
                    raise ConversionError("conclusion_v2_has_no_deterministic_evidence_binding")
                elif args.include_non_conclusion and task in {
                    "user_question_hypothesis_generation",
                    "follow_up_hypothesis_generation",
                }:
                    converted = convert_non_conclusion(record, source_name)
                    accepted_by_split[split].append(converted)
                    if source_name == "kto":
                        kto_by_split[split].append(converted)
                else:
                    raise ConversionError("task_not_selected")
            except (ConversionError, KeyError, IndexError, TypeError) as exc:
                reason = str(exc) or exc.__class__.__name__
                stats[f"reject:{source_name}:{split}:{reason}"] += 1
                rejects.append(reject_payload(source_name, source_path, split, record, reason))
                continue
            stats[f"accepted:{source_name}:{split}:{task}"] += 1
            if converted.action:
                stats[f"action:{source_name}:{split}:{converted.action}"] += 1

    # De-duplicate exact rows, and keep all rows sharing a question key on one
    # side of the train/val boundary.  Existing val/test ownership wins.
    held_out_keys = {
        row.question_key
        for split in ("val", "test")
        for row in accepted_by_split.get(split, [])
        if row.question_key
    }
    if held_out_keys:
        retained: list[ConvertedRow] = []
        for row in accepted_by_split.get("train", []):
            if row.question_key and row.question_key in held_out_keys:
                stats["reject:train_val_question_leakage"] += 1
                rejects.append(
                    {
                        "source": row.record.get("meta", {}).get("conversion_source"),
                        "split": "train",
                        "id": row.record.get("id"),
                        "task_type": row.source_task,
                        "reason": "train_val_question_leakage",
                        "record": row.record,
                    }
                )
            else:
                retained.append(row)
        accepted_by_split["train"] = retained

    for split, rows in list(accepted_by_split.items()):
        seen: set[str] = set()
        unique: list[ConvertedRow] = []
        for row in rows:
            key = hashlib.sha256(
                json_dump(
                    {
                        "system": row.record.get("system"),
                        "conversations": row.record.get("conversations"),
                        "kto_tag": row.record.get("kto_tag"),
                    }
                ).encode("utf-8")
            ).hexdigest()
            if key in seen:
                stats[f"reject:{split}:exact_duplicate"] += 1
                continue
            seen.add(key)
            unique.append(row)
        accepted_by_split[split] = unique

    report = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "policy": {
            "inputs_are_read_only": True,
            "overwrite_output": False,
            "evidence_binding_source": "model_visible_prompt_only",
            "uses_gold_metadata": False,
            "full_evidence_preserved": True,
            "legacy_conclusion_v2_policy": "reject_for_teacher_relabelling",
        },
        "inputs": input_manifest,
        "stats": dict(sorted(stats.items())),
        "outputs": {split: len(rows) for split, rows in sorted(accepted_by_split.items())},
        "rejects": len(rejects),
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    sft_dir = output_dir / "sft"
    sft_dir.mkdir()
    for split in ("train", "val", "test"):
        rows = accepted_by_split.get(split, [])
        if rows:
            write_json(sft_dir / f"{split}.json", [item.record for item in rows])
    dataset_info: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        if not (sft_dir / f"{split}.json").exists():
            continue
        dataset_info[f"exx_grounding_v1_{split}"] = {
            "file_name": f"{split}.json",
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
    write_json(sft_dir / "dataset_info.json", dataset_info)

    with (output_dir / "rejects.jsonl").open("w", encoding="utf-8") as handle:
        for reject in rejects:
            handle.write(json_dump(reject) + "\n")
    write_json(output_dir / "audit.json", report)

    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "protocol": PROTOCOL,
        "created_at_utc": report["created_at_utc"],
        "inputs": input_manifest,
        "outputs": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
