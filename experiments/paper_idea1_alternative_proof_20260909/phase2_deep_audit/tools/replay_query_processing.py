#!/usr/bin/env python3
"""Replay local query normalization/merge/planning; does NOT retrieve or infer."""
from __future__ import annotations

import collections
from dataclasses import asdict
from pathlib import Path

from research_controls import ROOT, PILOT, REPO, digest, file_digest, load, write, load_payload, protocol_problems
from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
from asa_arknight_story_agent.inference.payload.conclusion_payload import normalize_conclusion_payload
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import merge_hypotheses
from asa_arknight_story_agent.inference.planning.follow_up_query_generation import build_follow_up_hypothesis_queries
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ModelOutputError


def make_hypothesis(value):
    return HypothesisDocument(**value)


def saturation_demo():
    """Constructed counterexample; not a measured natural-query failure rate."""
    base = HypothesisDocument(
        question="哪个中转站可以提供通行口令？", intent="plot_fact", query_type="factual",
        entities=[f"旧实体{i:02d}" for i in range(12)],
        keywords=[f"旧词{i:02d}" for i in range(20)], expected_answer_type="short_text")
    follow = HypothesisDocument(
        question=base.question, intent=base.intent, query_type=base.query_type,
        entities=["新桥接实体"], keywords=["新缺失关系"], expected_answer_type="short_text")
    merged = merge_hypotheses(base, follow)
    return {
        "synthetic_counterexample": True,
        "base": asdict(base), "follow_up": asdict(follow), "merged": asdict(merged),
        "new_entity_survives": "新桥接实体" in merged.entities,
        "new_keyword_survives": "新缺失关系" in merged.keywords,
        "conclusion": "base-first concatenation plus caps can discard all new repair terms",
    }


def main():
    cases = {r["source_id"]: r for r in load(PILOT / "input/frozen_cases.json")["cases"]}
    records, summary = [], {}
    for reader in ("qwen35_4b_base", "clean_sft"):
        counter = collections.Counter()
        for row in load(PILOT / f"runs/nonthinking_vllm/{reader}.predictions.json"):
            payload = load_payload(row)
            if payload["next_action"] != "retrieve_more":
                continue
            source = cases[row["diagnostic"]["source_id"]]
            base = make_hypothesis(source["hypothesis"])
            raw = payload["follow_up_hypothesis"]
            strict_problems = protocol_problems(payload, set(row["diagnostic"]["evidence_id_to_corpus_id"]))
            record = {
                "reader": reader, "id": row["id"], "raw_output_sha256": digest(row["raw_output"]),
                "old_strict_protocol_problems": strict_problems, "raw_follow_up": raw,
            }
            counter["retrieve_rows"] += 1
            counter["old_strict_protocol_pass"] += not strict_problems
            try:
                normalized = normalize_hypothesis_payload(
                    raw, question=source["question"], dialogue_context="", current_intent=base.intent)
                conclusion = normalize_conclusion_payload(
                    payload, question=source["question"], dialogue_context="",
                    current_intent=base.intent, current_hypothesis=base, max_round_reached=False)
                merged = merge_hypotheses(base, normalized)
                queries = build_follow_up_hypothesis_queries(source["question"], merged)
                without_merge = build_follow_up_hypothesis_queries(source["question"], normalized)
                record.update({
                    "runtime_normalizer_accepted": True,
                    "full_conclusion_normalizer_action": conclusion.next_action,
                    "normalized_follow_up": asdict(normalized), "merged_hypothesis": asdict(merged),
                    "planned_follow_up_queries": queries,
                    "planner_without_history_merge": without_merge,
                    "without_merge_is_diagnostic_only": True,
                    "queries_lost_after_history_merge": [q for q in without_merge if q not in queries],
                    "model_question_was_rewritten_by_normalizer": raw.get("question") != normalized.question,
                    "normalized_entity_count": len(normalized.entities),
                    "normalized_keyword_count": len(normalized.keywords),
                    "retrieval_executed": False,
                })
                counter["runtime_normalizer_accepted"] += 1
                counter["conclusion_normalizer_retrieve_more"] += conclusion.next_action == "retrieve_more"
                counter["nonempty_planned_queries"] += bool(queries)
                counter["rewritten_follow_up_question"] += raw.get("question") != normalized.question
            except ModelOutputError as exc:
                record.update({"runtime_normalizer_accepted": False, "error": str(exc)})
            records.append(record)
        summary[reader] = dict(counter)
    paths = [
        "inference/payload/hypothesis_payload.py",
        "inference/payload/conclusion_payload.py",
        "inference/payload/conclusion_followup_payload.py",
        "inference/planning/follow_up_hypothesis.py",
        "inference/planning/follow_up_query_generation.py",
        "inference/pipeline/query_flow.py",
    ]
    result = {
        "scope": "current local working-tree pure functions, not historical remote runtime or retrieval success",
        "summary": summary, "records": records,
        "saturated_merge_counterexample": saturation_demo(),
        "source_files": {p: file_digest(REPO / "src/asa_arknight_story_agent" / p) for p in paths},
        "script_sha256": file_digest(Path(__file__)),
    }
    query_groups = collections.defaultdict(list)
    for record in records:
        if record.get("runtime_normalizer_accepted"):
            query_groups[digest(record["planned_follow_up_queries"])].append({
                "reader": record["reader"], "id": record["id"],
                "raw_output_sha256": record["raw_output_sha256"],
            })
    result["identical_planned_query_groups"] = [
        {"query_list_sha256": key, "members": members}
        for key, members in query_groups.items() if len(members) > 1
    ]
    write(ROOT / "audits/query_processing_replay.json", result)
    print(summary)
    print("Synthetic merge counterexample:", result["saturated_merge_counterexample"]["conclusion"])


if __name__ == "__main__":
    main()
