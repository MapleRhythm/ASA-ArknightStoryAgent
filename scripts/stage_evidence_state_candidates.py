#!/usr/bin/env python3
"""Stage paired evidence states from a content-bound, audited train export.

No labels here are training-ready. In particular, deleting cited passages (even
with overlap closure) does not prove insufficiency. Every changed visible
context needs a fresh evidence-only set audit. Missing-state follow-up queries
are question-only probes, NOT reviewed retrieval supervision.
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
from audit_binding_hardneg import parse_evidence
from audit_exx_set_grounding import audited_input_sha256, prompt_value
from audit_training_lineage import content_sha256
from build_relevance_clean_exx_data import (
    audit_binding_issue, canonical, cleaned_payload, load_json,
    parse_payload, sha256_file, source_training_issue,
)
from glm_exx_semantic_reward import extract_judge_context, payload_is_judge_eligible
from asa_arknight_story_agent.inference.generation.exx_prompt import (
    EXX_SYSTEM_PROMPT, render_exx_user_prompt,
)

PROTOCOL = "asa_evidence_state_candidates_v1"


def unique(rows):
    result = {}
    for row in rows:
        if not row.get("id") or str(row["id"]) in result:
            raise ValueError("duplicate_or_missing_id")
        result[str(row["id"])] = row
    return result


def checked_file(info):
    if not isinstance(info, dict) or not info.get("path") or not info.get("sha256"):
        raise ValueError("missing_file_provenance")
    path = Path(info["path"])
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"file_content_mismatch:{path.name}")
    return path


def load_verified_sources(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("source_split") != "train"
        or report.get("training_eligible") is not True
        or report.get("audit_content_and_contract_verified") is not True
        or not report.get("heldout_inputs")
    ):
        raise ValueError("requires_verified_train_export")
    accepted_path = checked_file((report.get("outputs") or {}).get("sft.json"))
    source_path = checked_file({"path": report["predictions"], "sha256": report["source_sha256"]})
    audit_path = checked_file({"path": report["audit"], "sha256": report["audit_sha256"]})
    lineage_path = checked_file(report.get("lineage_audit"))
    for item in report["heldout_inputs"]:
        checked_file(item)
    if report.get("candidate_report"):
        checked_file(report["candidate_report"])
    if report.get("secondary_review"):
        checked_file(report["secondary_review"])
    for previous in report.get("prior_secondary_reviews") or []:
        for name in ("review", "predictions", "audit"):
            checked_file(previous[name])
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    partitions = unique([row for row in lineage["records"] if row["split"] == "train"])
    sources = unique(load_json(source_path))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    judgements = unique(audit["results"])
    accepted = load_json(accepted_path)
    unique(accepted)
    if len(accepted) != report["sft_rows"]:
        raise ValueError("export_row_count_mismatch")
    out = []
    for record in accepted:
        meta = record.get("meta") or {}
        meta = json.loads(meta) if isinstance(meta, str) else meta
        source = sources.get(str(meta.get("source_id")))
        if (
            source is None or source_training_issue(record)
            or meta.get("training_eligible") is not True
            or meta.get("audited_source_sha256") != content_sha256(source)
            or record["conversations"][0]["value"] != prompt_value(source)
            or record["conversations"][-1]["value"] != canonical(parse_payload(source))
            or record.get("system") != source.get("system")
        ):
            raise ValueError("export_source_content_mismatch")
        judged = judgements.get(str(source["id"]))
        payload = parse_payload(source)
        if payload.get("next_action") != "answer_directly":
            continue
        if (
            not judged or judged.get("status") != "ok"
            or audit_binding_issue(source, judged, audit["summary"])
            or cleaned_payload(payload, judged.get("judgement") or {}) is None
        ):
            raise ValueError("export_source_no_longer_approved")
        partition = partitions.get(str(source["id"]))
        if (
            not partition or partition.get("partition_candidate") is not True
            or partition.get("quarantine_reasons") or not partition.get("component_id")
        ):
            raise ValueError("export_source_partition_not_approved")
        out.append((source, partition["component_id"]))
    return out


def overlap_graph(evidence: dict[str, str], width: int = 40):
    """Conservative lexical duplicate closure, never a semantic support test."""
    compact = {eid: re.sub(r"\s+", "", text) for eid, text in evidence.items()}
    grams = {
        eid: {text[i:i + width] for i in range(max(0, len(text) - width + 1))}
        for eid, text in compact.items()
    }
    graph = {eid: set() for eid in evidence}
    ids = list(evidence)
    for index, left in enumerate(ids):
        for right in ids[index + 1:]:
            a, b = compact[left], compact[right]
            containment = min(len(a), len(b)) >= 24 and (a in b or b in a)
            if a == b or containment or not grams[left].isdisjoint(grams[right]):
                graph[left].add(right)
                graph[right].add(left)
    return graph


def removal_closure(seeds: set[str], graph: dict[str, set[str]]):
    if not seeds or not seeds <= graph.keys():
        raise ValueError("invalid_removal_seed")
    removed, frontier = set(seeds), list(seeds)
    while frontier:
        for other in graph[frontier.pop()] - removed:
            removed.add(other)
            frontier.append(other)
    return removed


def question_only_hypothesis(question: str):
    # No inherited hypothesis, gold answer, removed evidence or judge rationale.
    return {
        "question": question, "query_type": "reasoning", "entities": [],
        "keywords": [], "expected_answer_type": "short_text",
    }


def render_state(source, component, condition, visible, payload, round_value, removed):
    question = extract_judge_context(prompt_value(source))["question"]
    eid_map = {old: f"E{index}" for index, old in enumerate(visible, 1)}
    output = copy.deepcopy(payload)
    if output["next_action"] == "answer_directly":
        for fact in output["supported_facts"]:
            fact["evidence_ids"] = [eid_map[eid] for eid in fact["evidence_ids"]]
    evidence_text = "\n".join(f"[{eid_map[eid]}]\n{text}" for eid, text in visible.items())
    prompt = render_exx_user_prompt(
        question=question, hypothesis=question_only_hypothesis(question),
        round_value=round_value, evidence_text=evidence_text,
    )
    if not payload_is_judge_eligible(output, evidence_text):
        raise ValueError("state_protocol_invalid")
    text = canonical(output)
    identity = canonical({
        "protocol": PROTOCOL, "source_id": source["id"], "prompt": prompt,
        "output": text, "system": EXX_SYSTEM_PROMPT,
    })
    row = {
        "id": "evidence_state_" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        "task_type": "grounded_action_generation", "system": EXX_SYSTEM_PROMPT,
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": text}],
        "raw_output": text,
        "meta": {
            "source": PROTOCOL, "source_id": source["id"], "source_question": question,
            "component_id": component, "source_split": "train",
            "training_eligible": False, "requires_reaudit": True,
        },
        "state_provenance": {
            "protocol": PROTOCOL, "source_id": source["id"],
            "source_row_sha256": content_sha256(source), "component_id": component,
            "condition": condition, "round": round_value,
            "source_eid_to_visible_eid": eid_map, "removed_source_eids": sorted(removed),
            "evidence_sufficiency": "pending_independent_visible_context_audit",
            "lexical_overlap_removal_does_not_prove_insufficiency": True,
            "hypothesis_uses_question_only": True,
            "retrieval_query_supervision_ready": False,
        },
    }
    row["state_provenance"]["audited_input_sha256"] = audited_input_sha256(row)
    return row


def stage(sources, limit_components=0):
    rows, seen_components, seen_rows, counts = [], set(), set(), Counter()
    for source, component in sorted(sources, key=lambda item: (item[1], str(item[0]["id"]))):
        if component in seen_components:
            counts["skipped_same_component"] += 1
            continue
        if limit_components and len(seen_components) >= limit_components:
            break
        seen_components.add(component)
        if source_training_issue(source):
            raise ValueError("source_not_train")
        evidence = parse_evidence(prompt_value(source))
        payload = parse_payload(source)
        if not payload or not payload_is_judge_eligible(payload, extract_judge_context(prompt_value(source))["evidence"]):
            raise ValueError("source_protocol_invalid")
        if payload["next_action"] != "answer_directly":
            raise ValueError("requires_answer_source")
        cites = {eid for fact in payload["supported_facts"] for eid in fact["evidence_ids"]}
        states = [
            ("full", {eid: text for eid, text in evidence.items() if eid in cites}, set(evidence) - cites),
            # These are naturally retrieved same-source passages, not certified
            # hard negatives; some may be alternate supporting evidence.
            ("distractor", evidence, set()),
        ]
        graph = overlap_graph(evidence)
        options = {
            frozenset(removal_closure(set(fact["evidence_ids"]), graph))
            for fact in payload["supported_facts"]
        }
        viable = sorted(
            (removed for removed in options if set(evidence) - removed),
            key=lambda removed: (len(removed), sorted(removed)),
        )
        if viable:
            removed = set(viable[0])
            states.append(("missing_candidate", {
                eid: text for eid, text in evidence.items() if eid not in removed
            }, removed))
            counts["overlap_extra_removed"] += min(
                len(removed - set(fact["evidence_ids"]))
                for fact in payload["supported_facts"]
                if set(fact["evidence_ids"]) <= removed
            )
        else:
            counts["missing_skipped_no_nonempty_context"] += 1
        question = extract_judge_context(prompt_value(source))["question"]
        for condition, visible, removed in states:
            for round_value in ("1/2", "2/2"):
                proposed = payload
                if condition == "missing_candidate":
                    proposed = (
                        {"next_action": "retrieve_more", "follow_up_hypothesis": question_only_hypothesis(question)}
                        if round_value == "1/2"
                        else {"next_action": "abstain", "reason": "当前证据不足以完整回答问题。"}
                    )
                row = render_state(source, component, condition, visible, proposed, round_value, removed)
                if row["id"] in seen_rows:
                    counts["duplicate_visible_state_skipped"] += 1
                    continue
                seen_rows.add(row["id"])
                rows.append(row)
                counts[f"condition:{condition}"] += 1
                counts[f"proposed_action:{proposed['next_action']}"] += 1
    return rows, {
        "protocol": PROTOCOL, "rows": len(rows), "components": len(seen_components),
        "counts": dict(sorted(counts.items())), "training_ready": False,
        "independent_human_gold": False,
        "all_changed_states_require_set_reaudit": True,
        "component_is_not_independent_semantic_family": True,
        "missing_actions_are_proposals_not_labels": True,
        "retrieval_query_supervision_ready": False,
        "note": "Question-only follow-ups probe action suitability, not retrieval quality. "
                "No training export/registry; no hidden evidence in hypothesis. "
                "Length-matched controls and independent semantic-family evaluation remain required.",
        "records": [{"id": row["id"], **row["state_provenance"]} for row in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-components", type=int, default=0)
    args = parser.parse_args()
    report_path = args.output.with_suffix(".report.json")
    if args.output.exists() or report_path.exists() or args.limit_components < 0:
        parser.error("output must be new and limit-components nonnegative")
    rows, report = stage(load_verified_sources(args.export_report), args.limit_components)
    report["export_report"] = {"path": str(args.export_report), "sha256": sha256_file(args.export_report)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = sha256_file(args.output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
