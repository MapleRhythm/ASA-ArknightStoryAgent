#!/usr/bin/env python3
"""Isolated deterministic probes, not new LLM generations or retrieval results."""
from __future__ import annotations

import collections
import copy
from dataclasses import asdict
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PREVIOUS = REPO / "experiments/paper_idea1_alternative_proof_20260909"
sys.path.insert(0, str(REPO / "src"))

from asa_arknight_story_agent.inference.pipeline.query_flow import build_initial_queries, prepare_next_round
from asa_arknight_story_agent.inference.pipeline.scheduler import AdaptiveRoundScheduler
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument, ConclusionResult
from asa_arknight_story_agent.inference.payload.conclusion_payload import normalize_conclusion_payload
from asa_arknight_story_agent.inference.planning.follow_up_query_generation import build_follow_up_hypothesis_queries
from asa_arknight_story_agent.inference.retrieval.query_rendering import build_retrieval_query
from asa_arknight_story_agent.inference.grounding.evidence_id_validation import validate_evidence_id_grounding
from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes


def load(path):
    return json.loads(path.read_text())


def sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def replays():
    frozen = load(PREVIOUS / "input/frozen_cases.json")["cases"]
    cases = {r["source_id"]: r for r in frozen}
    records = []
    for reader in ("qwen35_4b_base", "clean_sft"):
        for row in load(PREVIOUS / f"runs/nonthinking_vllm/{reader}.predictions.json"):
            payload = json.loads(row["raw_output"])
            if payload["next_action"] != "retrieve_more":
                continue
            case = cases[row["diagnostic"]["source_id"]]
            base = HypothesisDocument(**case["hypothesis"])
            normalized = normalize_conclusion_payload(
                payload, question=case["question"], dialogue_context="",
                current_intent=base.intent, current_hypothesis=base, max_round_reached=False)
            step = {}
            hypothesis, pending = prepare_next_round(
                pipeline=None, question=case["question"], current_hypothesis=base,
                conclusion=normalized, state=PipelineRunState(), step_record=step,
                round_index=1, progress_callback=None)
            follow = normalized.follow_up_hypothesis
            if follow is None:
                raise ValueError("unexpected fallback")
            no_merge = [build_retrieval_query(follow),
                        *build_follow_up_hypothesis_queries(case["question"], follow)]
            scheduler = AdaptiveRoundScheduler()
            initial_issued, _ = scheduler.prepare_queries(build_initial_queries(case["question"], base))
            scheduled, duplicates = scheduler.prepare_queries(pending)
            follow_terms = list(dict.fromkeys(
                payload["follow_up_hypothesis"].get("entities", [])
                + payload["follow_up_hypothesis"].get("keywords", [])))
            new_terms = [t for t in follow_terms if t not in base.entities + base.keywords]
            records.append({
                "reader": reader, "id": row["id"], "case": row["diagnostic"]["short_name"],
                "raw_output_sha256": sha(row["raw_output"]), "source_case_sha256": sha(case),
                "raw_follow_up": payload["follow_up_hypothesis"],
                "normalized_follow_up": asdict(follow), "merged_hypothesis": asdict(hypothesis),
                "all_pending_queries": pending, "aggregate_query": pending[0],
                "followup_branch_queries": pending[1:],
                "hypothetical_initial_queries": initial_issued,
                "hypothetical_scheduled_queries": scheduled,
                "hypothetical_duplicate_count": duplicates,
                "no_merge_all_pending_queries": no_merge,
                "new_raw_terms": new_terms,
                "new_terms_absent_followup_branch": [t for t in new_terms if t not in "\n".join(pending[1:])],
                "new_terms_absent_all_pending": [t for t in new_terms if t not in "\n".join(pending)],
                "new_terms_absent_after_hypothetical_scheduler": [t for t in new_terms if t not in "\n".join(scheduled)],
                "no_retrieval_or_model_execution": True,
                "scheduler_note": "Reconstructed initial queries, not the historic issued trace; illustrative only",
            })
    groups = {}
    for field in ("raw_follow_up", "followup_branch_queries", "all_pending_queries", "hypothetical_scheduled_queries"):
        by_case = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in records:
            by_case[r["case"]][sha(r[field])].append(f"{r['reader']}::{r['id']}")
        groups[field] = [
            {"case": case, "hash": digest, "members": members}
            for case, hashed in by_case.items() for digest, members in hashed.items() if len(members) > 1]
    return {"scope": "all local pending-query branches; retrieval not run",
            "records": records, "collision_groups": groups,
            "input_hashes": {str(p.relative_to(REPO)): file_sha(p) for p in [
                PREVIOUS / "input/frozen_cases.json",
                PREVIOUS / "runs/nonthinking_vllm/qwen35_4b_base.predictions.json",
                PREVIOUS / "runs/nonthinking_vllm/clean_sft.predictions.json"]}}


def synthetic_scope_rows():
    names = [("林舟", "周宁", "陈禾"), ("顾川", "许岚", "沈青"),
             ("方予", "陆安", "江月"), ("叶秋", "苏禾", "唐宁")]
    rows = []
    for world, (a, b, c) in enumerate(names):
        situations = [
            ("reported_speech",
             f"记录：{a}转述了{b}的说法“{c}支持提案”，没有表明自己的立场。",
             [(f"{a}转述了{b}说{c}支持提案。", "supported"),
              (f"{a}已经表明自己的立场。", "contradicted"),
              (f"{c}支持提案。", "not_established")]),
            ("negation",
             f"记录：{a}没有同意{b}的申请，也没有说明{c}的意见。",
             [(f"{a}没有同意{b}的申请。", "supported"),
              (f"{a}同意了{b}的申请。", "contradicted"),
              (f"{c}同意了{b}的申请。", "not_established")]),
            ("argument_direction",
             f"记录：{a}这次说的是“昨晚{b}救了{c}”，不是“昨晚{c}救了{b}”。",
             [(f"{a}说昨晚{b}救了{c}。", "supported"),
              (f"{a}说昨晚{c}救了{b}。", "contradicted"),
              (f"{a}昨晚救了{b}。", "not_established")]),
            ("belief_vs_fact",
             f"记录：{a}仅怀疑{b}偷走档案，并没有证据确认；{c}没有发表看法。",
             [(f"{a}怀疑{b}偷走档案。", "supported"),
              (f"{a}有证据确认{b}偷走档案。", "contradicted"),
              (f"{b}偷走了档案。", "not_established")]),
            ("temporal_scope",
             f"旁白：昨日{a}担任队长，今日已经卸任。今日队长是{b}。",
             [(f"今日队长是{b}。", "supported"),
              (f"今日{a}仍担任队长。", "contradicted"),
              (f"明日队长是{b}。", "not_established")]),
            ("negated_reporting_vs_negative_fact",
             f"记录：{a}从未说过“{b}反对提案”，也没有提供{b}的立场信息。",
             [(f"{a}没有说过{b}反对提案。", "supported"),
              (f"{a}说过{b}反对提案。", "contradicted"),
              (f"{b}不反对提案。", "not_established")]),
        ]
        for kind, text, claims in situations:
            if len(text) > 80:
                raise ValueError("quote would be truncated")
            for i, (claim, label) in enumerate(claims):
                rows.append({
                    "id": f"synthetic_{world}_{kind}_{i}", "family": kind,
                    "kind": "synthetic", "source_text": text, "quote": text,
                    "claim": claim, "reference_label": label,
                    "source_sha256": sha(text), "exclude_from_training": True,
                    "label_authority": "agent-authored controlled semantics, not human gold",
                    "semantics_note": "Narrated record fixes one event/time. Nested reported claims do not entail unqualified world truth.",
                })
    return rows


def natural_scope_rows():
    docs = {d["corpus_id"]: d for c in load(PREVIOUS / "input/frozen_cases.json")["cases"] for d in c["documents"]}
    cases = [
        ("reported_speaker", "obt/main/level_main_13-09_beg#chunk-0003",
         "——但是汤姆那个混球沾酒就胡说，他骂你就是仗着魔族佬气焰的叛徒！",
         "汤姆骂芙蕾达是叛徒。", "威尔骂芙蕾达是叛徒。"),
        ("attitude_owner", "activities/act19mini/level_act19mini_st05#chunk-0020",
         "只是剑化身的伥，戾气难去，或许她自己都无法控制。",
         "麟青砚认为剑化身的伥戾气难去。", "剑客认为剑化身的伥戾气难去。"),
        ("gift_direction", "activities/act19mini/level_act19mini_st05#chunk-0016",
         "想想还真好笑，去年我们见面的时候，你甚至还想着把“渡魂剑”这个名字送给我......明明，我就是它。",
         "剑客曾想把渡魂剑这个名字送给女子。", "女子曾想把渡魂剑这个名字送给剑客。"),
        ("intention_vs_completion", "activities/act19mini/level_act19mini_st05#chunk-0016",
         "想想还真好笑，去年我们见面的时候，你甚至还想着把“渡魂剑”这个名字送给我......明明，我就是它。",
         "剑客曾想把渡魂剑这个名字送给女子。", "剑客已经把渡魂剑这个名字送给女子。"),
        ("negation", "obt/main/level_main_07-11_beg#chunk-0014",
         "如你所知，阿米娅从来都不曾想要与人为敌。即使她被逼迫去这么做，即使她流着泪努力去这么做。",
         "凯尔希说阿米娅从来都不曾想要与人为敌。", "凯尔希说阿米娅想要与人为敌。"),
        ("search_direction", "activities/act47side/level_act47side_08_end#chunk-0023",
         "一路辗转，寻着娜斯提的踪迹来到哥伦比亚，已经超过半年。",
         "拂哀菈寻着娜斯提的踪迹来到哥伦比亚。", "娜斯提寻着拂哀菈的踪迹来到哥伦比亚。"),
    ]
    rows = []
    for kind, doc_id, quote, good, bad in cases:
        doc = docs[doc_id]
        if quote not in doc["text"] or len(quote) > 80 or sha(doc["text"]) != doc["text_sha256"]:
            raise ValueError("source/quote mismatch")
        for i, (claim, label) in enumerate(((good, "supported"), (bad, "not_established"))):
            rows.append({
                "id": f"natural_{kind}_{i}", "family": kind, "kind": "natural_development",
                "source_text": doc["text"], "source_id": doc_id, "quote": quote,
                "claim": claim, "reference_label": label, "source_sha256": sha(doc["text"]),
                "exclude_from_training": True, "label_authority": "Codex source review, not independent human review",
                "quote_scope_note": "Gold support is judged from the full visible source; a short quote may omit speaker context.",
            })
    return rows


def lexical_probe(rows):
    details = []
    for row in rows:
        for mode, validator in (("evidence_id", validate_evidence_id_grounding), ("quote", validate_grounded_quotes)):
            refs = [{"evidence_id": "E1"}]
            if mode == "quote":
                refs[0]["quote"] = row["quote"]
            conclusion = ConclusionResult(
                next_action="answer_directly", answer=row["claim"], missing_slots=[],
                clarification_question="", follow_up_hypothesis=None,
                supported_facts=[{"fact": row["claim"], "evidence_refs": refs}])
            issues, warnings = validator(conclusion=conclusion, evidence=[],
                                         question="当前原文支持哪些陈述？",
                                         evidence_prompt_text=f"[E1]\n{row['source_text']}")
            details.append({
                "id": row["id"], "kind": row["kind"], "family": row["family"],
                "reference_label": row["reference_label"], "mode": mode,
                "claim_sha256": sha(row["claim"]), "source_sha256": row["source_sha256"],
                "no_hard_issue": not issues, "issues": issues, "warnings": warnings,
                "interpretation": "Low-level lexical check only; not whole serving outcome or model correctness",
            })
    stats = {}
    for kind in ("synthetic", "natural_development"):
        for mode in ("evidence_id", "quote"):
            subset = [r for r in details if r["kind"] == kind and r["mode"] == mode]
            counter = collections.Counter()
            for r in subset:
                label = r["reference_label"]
                counter[f"{label}_count"] += 1
                counter[f"{label}_no_hard_issue"] += r["no_hard_issue"]
            stats[f"{kind}/{mode}"] = dict(counter)
    return {"scope": "controlled local lexical validator probe, not estimated hallucination rate",
            "stats": stats, "details": details}


def main():
    full = replays()
    write(ROOT / "audits/full_query_pipeline_replay.json", full)
    rows = synthetic_scope_rows() + natural_scope_rows()
    write(ROOT / "data/attribution_scope_probes.json", rows)
    result = lexical_probe(rows)
    write(ROOT / "audits/lexical_scope_probe.json", result)
    sources = [
        "inference/pipeline/query_flow.py", "inference/pipeline/scheduler.py",
        "inference/planning/follow_up_hypothesis.py",
        "inference/planning/follow_up_query_generation.py",
        "inference/retrieval/query_rendering.py",
        "inference/payload/hypothesis_payload.py",
        "inference/grounding/evidence_id_validation.py", "inference/grounding/quote_validation.py",
        "data/story_text.py",
    ]
    source_tree = REPO / "src/asa_arknight_story_agent"
    artifacts = [
        ROOT / "audits/full_query_pipeline_replay.json",
        ROOT / "data/attribution_scope_probes.json",
        ROOT / "audits/lexical_scope_probe.json",
    ]
    write(ROOT / "RUN_MANIFEST.json", {
        "script_sha256": file_sha(Path(__file__)),
        "source_hashes": {p: file_sha(source_tree / p) for p in sources},
        "source_tree_hashes": {str(p.relative_to(source_tree)): file_sha(p)
                               for p in sorted(source_tree.rglob("*.py"))},
        "artifact_hashes": {str(p.relative_to(ROOT)): file_sha(p) for p in artifacts},
        "query_replay_rows": len(full["records"]), "scope_probe_rows": len(rows),
        "scope_probe_kinds": dict(collections.Counter(r["kind"] for r in rows)),
        "model_calls": 0, "retrieval_calls": 0, "training": False,
        "production_modified": False,
        "production_modified_note": "This isolated experiment does not edit production files; existing dirty changes are preserved.",
    })
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
    for row in full["records"]:
        if row["case"] == "sword_identity":
            print(row["reader"], row["id"])
            print("All-query missing new terms:", row["new_terms_absent_all_pending"])
            print("Aggregate:", row["aggregate_query"])


if __name__ == "__main__":
    main()
