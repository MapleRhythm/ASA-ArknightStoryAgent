#!/usr/bin/env python3
"""Verify predictions and distinguish action/format/semantic-proxy metrics."""
from __future__ import annotations

import collections
import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from build_inputs import ROOT, REPO, load, sha, write

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/paper_idea1_alternative_proof_20260909/tools"))
from analyze_pilot import protocol_problems
from asa_arknight_story_agent.inference.payload.conclusion_payload import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import validate_evidence_id_grounding
from asa_arknight_story_agent.inference.pipeline.query_flow import prepare_next_round
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState


NAMES = [("林舟", "周宁", "陈禾"), ("顾川", "许岚", "沈青"),
         ("方予", "陆安", "江月"), ("叶秋", "苏禾", "唐宁")]


def no_duplicates(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def checked_prediction(source, pred):
    if source["id"] != pred["id"] or source["system"] != pred["system"]:
        raise ValueError("ID/system mismatch")
    if source["conversations"][0] != pred["conversations"][0] or source["diagnostic"] != pred["diagnostic"]:
        raise ValueError("prompt/reference mismatch")
    if pred["raw_output"] != pred["conversations"][-1]["value"] or sha(pred["raw_output"]) != pred["raw_output_sha256"]:
        raise ValueError("output binding mismatch")
    try:
        payload = json.loads(pred["raw_output"], object_pairs_hook=no_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, str(exc)
    return (payload, None) if isinstance(payload, dict) else (None, "non-object JSON")


def normalize_result(source, payload):
    diag = source["diagnostic"]
    fields = diag["input_fields"]
    h = fields["hypothesis"]
    if isinstance(h, str):
        h = json.loads(h)
    filtered = {key: value for key, value in h.items() if key in HypothesisDocument.__dataclass_fields__}
    # Diagnostic-only reconstruction: question-only experiment inputs omit this
    # internal dataclass field. Never alter the actual model prompt.
    filtered.setdefault("intent", "fact")
    try:
        hypothesis = HypothesisDocument(**filtered)
        normalized = normalize_conclusion_payload(
            copy.deepcopy(payload), question=fields["question"], dialogue_context="",
            current_intent=hypothesis.intent, current_hypothesis=hypothesis,
            max_round_reached=fields["round"] == "2/2")
        record = {"accepted": True, "action": normalized.next_action,
                  "analysis_default_intent": "fact" if "intent" not in h else None,
                  "scope": "local reconstruction, not remote serving result",
                  "normalized": asdict(normalized)}
        if normalized.next_action == "answer_directly":
            evidence_text = "\n".join(f"[{eid}]\n{doc['text']}" for eid, doc in zip(
                fields["evidence_ids"], fields["documents"], strict=True))
            issues, warnings = validate_evidence_id_grounding(
                conclusion=normalized, evidence=[], question=fields["question"],
                evidence_prompt_text=evidence_text)
            record.update({"low_level_eid_issues": issues, "low_level_eid_warnings": warnings,
                           "low_level_eid_no_hard_issue": not issues})
        if normalized.next_action == "retrieve_more" and normalized.follow_up_hypothesis is not None:
            _, pending = prepare_next_round(
                pipeline=None, question=fields["question"], current_hypothesis=hypothesis,
                conclusion=normalized, state=PipelineRunState(), step_record={},
                round_index=1, progress_callback=None)
            record["all_pending_queries"] = pending
            record["retrieval_executed"] = False
        return record
    except Exception as exc:
        return {"accepted": False, "error_type": type(exc).__name__, "error": str(exc)}


def symbolic_proxy(source, payload):
    """Surface answer-token + citation-union checks, NOT semantic correctness."""
    diag = source["diagnostic"]
    ref = diag["reference"]
    if diag["suite"] != "synthetic_controls" or ref["status"] != "sufficient":
        return None
    facts = payload.get("supported_facts", [])
    if not isinstance(facts, list):
        return {"answer_token_and_citation_path": False}
    required = [set(ids) for ids in ref["all_valid_citation_pairs"]]
    joined_fact = any(
        isinstance(fact, dict)
        and ref["answer"] in str(fact.get("fact", ""))
        and isinstance(fact.get("evidence_ids"), list)
        and any(pair <= set(map(str, fact["evidence_ids"])) for pair in required)
        for fact in facts)
    cited_union = set()
    fact_texts = []
    for fact in facts:
        if isinstance(fact, dict):
            fact_texts.append(str(fact.get("fact", "")))
            if isinstance(fact.get("evidence_ids"), list):
                cited_union.update(map(str, fact["evidence_ids"]))
    passed = ref["answer"] in "\n".join(fact_texts) and any(pair <= cited_union for pair in required)
    return {"answer_token_and_citation_path": passed,
            "single_fact_contains_complete_path": joined_fact,
            "limitation": "Citation union may span separate facts. Does not prove relation direction, per-claim binding, completeness, or absence of extra false claims."}


def paired_records(details):
    paired = []
    for reader in {row["reader"] for row in details}:
        records = [row for row in details if row["reader"] == reader]
        for factor in ("round", "hypothesis", "eid_order"):
            grouped = collections.defaultdict(list)
            for row in records:
                factors = row["factors"]
                if factor not in factors:
                    continue
                fixed = {k: v for k, v in factors.items() if k != factor}
                grouped[(row["suite"], row["case"], sha(fixed))].append(row)
            for group in grouped.values():
                if len(group) != 2:
                    continue
                left, right = sorted(group, key=lambda x: str(x["factors"][factor]))
                if left["factors"][factor] == right["factors"][factor]:
                    continue
                paired.append({
                    "reader": reader, "suite": left["suite"], "case": left["case"],
                    "factor": factor, "left_id": left["id"], "right_id": right["id"],
                    "left_value": left["factors"][factor], "right_value": right["factors"][factor],
                    "left_action": left["raw_action"], "right_action": right["raw_action"],
                    "left_action_correct": left["raw_action_correct"],
                    "right_action_correct": right["raw_action_correct"],
                    "raw_output_same": left["raw_output_sha256"] == right["raw_output_sha256"],
                })
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["full_base_v1", "full_clean_v1"])
    parser.add_argument("--audit-subdir", default="audits")
    parser.add_argument("--input", type=Path, default=ROOT / "input/eval_matrix.json")
    args = parser.parse_args()
    matrix_path = args.input
    rows = load(matrix_path)
    source_index = {row["id"]: row for row in rows}
    matrix_hash = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    details, config_hashes, scope_groups = [], {}, collections.defaultdict(list)
    for run in args.runs:
        directory = ROOT / "runs" / run
        completion = load(directory / "completion.json")
        config = load(directory / "config.json")
        if config["input_sha256"] != matrix_hash or not config["nonthinking"]:
            raise ValueError("input or thinking mismatch")
        predictions = load(directory / "predictions.json")
        if len(predictions) != len(rows) or {p["id"] for p in predictions} != source_index.keys():
            raise ValueError("prediction coverage mismatch")
        if completion["predictions_sha256"] != hashlib.sha256((directory / "predictions.json").read_bytes()).hexdigest():
            raise ValueError("completion output hash mismatch")
        rendered = [json.loads(line) for line in (directory / "rendered_prompts.jsonl").read_text().splitlines()]
        rendered_index = {r["id"]: r for r in rendered}
        if len(rendered) != len(rows) or len(rendered_index) != len(rows):
            raise ValueError("rendered coverage mismatch")
        config_hashes[run] = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
        for pred in predictions:
            source = source_index[pred["id"]]
            diag = source["diagnostic"]
            ref = diag["reference"]
            payload, parse_error = checked_prediction(source, pred)
            rendered_row = rendered_index[pred["id"]]
            if sha(rendered_row["prompt"]) != pred["rendered_prompt_sha256"] or "<think>\n\n</think>" not in rendered_row["prompt"]:
                raise ValueError("rendered prompt mismatch or missing nonthinking prefix")
            action = payload.get("next_action") if payload else None
            expected = ref["action"]
            visible = set(diag["input_fields"]["evidence_ids"])
            try:
                problems = protocol_problems(payload, visible)
            except (TypeError, ValueError) as exc:
                problems = ["malformed_payload:" + str(exc)]
            runtime = normalize_result(source, payload) if payload else {"accepted": False, "error": parse_error}
            item = {
                "reader": pred["reader"], "id": pred["id"], "suite": diag["suite"],
                "case": diag["case"], "family": diag.get("family", diag["case"]),
                "factors": diag["factors"], "status": ref["status"], "expected_action": expected,
                "raw_action": action, "raw_action_correct": action == expected if expected is not None else None,
                "parsed": payload is not None, "parse_error": parse_error,
                "old_strict_protocol_ok": not problems, "old_strict_protocol_issues": problems,
                "runtime": runtime,
                "normalized_action_correct": runtime.get("action") == expected if expected is not None else None,
                "truncated": pred["hit_max_new_tokens"], "finish_reason": pred["finish_reason"],
                "prompt_tokens": pred["prompt_token_count"], "output_tokens": pred["generated_token_count"],
                "raw_output_sha256": pred["raw_output_sha256"],
                "symbolic_proxy": symbolic_proxy(source, payload) if payload else None,
                "semantic_correct": None,
            }
            details.append(item)
            if diag["suite"] == "scope_controlled":
                world = int(pred["id"].split("_")[1])
                normalized_text = pred["raw_output"]
                for name, placeholder in zip(NAMES[world], ("甲", "乙", "丙"), strict=True):
                    normalized_text = normalized_text.replace(name, placeholder)
                group_key = (pred["reader"], diag["family"], diag["factors"]["state"], normalized_text)
                scope_groups[group_key].append(pred["id"])

    stats = {}
    for reader, suite in sorted({(row["reader"], row["suite"]) for row in details}):
        subset = [row for row in details if row["reader"] == reader and row["suite"] == suite]
        count = collections.Counter()
        for item in subset:
            count["rows"] += 1
            count["parsed"] += item["parsed"]
            count["truncated"] += item["truncated"]
            count["old_strict_protocol_ok"] += item["old_strict_protocol_ok"]
            count["normalizer_accepted"] += item["runtime"]["accepted"]
            count[f"status_{item['status']}"] += 1
            count[f"action_{item['raw_action']}"] += 1
            if item["expected_action"] is not None:
                count["known_label_rows"] += 1
                count["raw_action_correct"] += item["raw_action_correct"]
                count["normalized_action_correct"] += item["normalized_action_correct"]
                expected = item["expected_action"]
                count[f"expected_{expected}"] += 1
                count[f"correct_when_expected_{expected}"] += item["raw_action_correct"]
            proxy = item["symbolic_proxy"]
            if proxy is not None:
                count["symbolic_proxy_rows"] += 1
                count["symbolic_proxy_pass"] += proxy["answer_token_and_citation_path"]
            if item["runtime"].get("action") == "answer_directly":
                count["normalized_answers"] += 1
                count["low_level_eid_no_hard_issue"] += item["runtime"]["low_level_eid_no_hard_issue"]
        stats[f"{reader}/{suite}"] = dict(count)
    pairs = paired_records(details)
    scope_review = [
        {"reader": reader, "family": family, "state": state, "anonymized_raw_output": text,
         "member_ids": ids, "count": len(ids), "review": None}
        for (reader, family, state, text), ids in scope_groups.items()
    ]
    out = ROOT / args.audit_subdir
    write(out / "details.json", details)
    write(out / "pairs.json", pairs)
    write(out / "scope_review_groups.json", scope_review)
    write(out / "summary.json", {
        "stats": stats, "matrix_sha256": matrix_hash, "run_config_hashes": config_hashes,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "semantic_accuracy": None, "scope_review_groups": len(scope_review),
        "metrics_note": "Action/format counts only. Synthetic answer-token+path is a necessary-condition proxy, not semantic accuracy.",
        "real_retrieval_calls": 0, "training": False,
    })
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
