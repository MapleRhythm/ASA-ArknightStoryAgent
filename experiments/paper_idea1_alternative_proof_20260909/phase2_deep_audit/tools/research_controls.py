#!/usr/bin/env python3
"""Audit frozen outputs and build evaluation-only single-factor controls.

No model calls, training, network requests, or production code changes.
Natural semantic judgments remain explicitly agent-reviewed and provisional.
Synthetic gold is independently computed from graph reachability.
"""
from __future__ import annotations

import collections
import copy
import difflib
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT.parent
REPO = PILOT.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PILOT / "tools"))
from asa_arknight_story_agent.inference.generation.exx_prompt import (
    EXX_SYSTEM_PROMPT, render_exx_user_prompt,
)
from analyze_pilot import protocol_problems


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    text = value if isinstance(value, str) else canonical(value)
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(path.read_text(), object_pairs_hook=no_duplicate_keys)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def index_unique(rows):
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError(f"duplicate ID: {key}")
        result[key] = row
    return result


def messages(row):
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["conversations"][0]["value"]},
    ]


def verify_prediction(source, prediction):
    if source["id"] != prediction["id"] or messages(source) != messages(prediction):
        raise ValueError("prediction prompt/system/ID mismatch")
    if source["diagnostic"] != prediction["diagnostic"]:
        raise ValueError("prediction diagnostic mismatch")
    if prediction["raw_output"] != prediction["conversations"][-1]["value"]:
        raise ValueError("raw output mismatch")
    return json.loads(prediction["raw_output"], object_pairs_hook=no_duplicate_keys)


def normalize_round(prompt):
    # Require exactly one anchored round line; do not replace digits in evidence.
    result, count = re.subn(r"^round: [12]/2$", "round: <MASK>", prompt, flags=re.M)
    if count != 1:
        raise ValueError("missing/duplicate round line")
    return result


def reviewed_status(short_name, condition, spec):
    case = spec["cases"][short_name]
    key = "broken_status" if condition.startswith("broken") else "positive_status"
    return case[key]


def expected_action(status, round_value):
    if status == "unknown":
        return None
    if status == "sufficient":
        return "answer_directly"
    if status != "insufficient":
        raise ValueError(status)
    return "retrieve_more" if round_value == "1/2" else "abstain"


def audit_existing(spec):
    hashes = {}
    for relative, expected in spec["source_hashes"].items():
        actual = file_digest(PILOT / relative)
        if expected is not None and actual != expected:
            raise ValueError(f"source hash mismatch: {relative}")
        hashes[relative] = actual
    sources = index_unique(load(PILOT / "data/pilot_matrix.json"))
    readers = {}
    for name in ("qwen35_4b_base", "clean_sft"):
        predictions = index_unique(load(PILOT / f"runs/nonthinking_vllm/{name}.predictions.json"))
        if sources.keys() != predictions.keys():
            raise ValueError("prediction coverage mismatch")
        readers[name] = predictions
    old_reviews = load(PILOT / "audits/semantic_review.json")
    old_index = {(r["reader"], r["id"]): r for r in old_reviews}
    if len(old_index) != len(old_reviews) or set(old_index) != {
        (name, row_id) for name in readers for row_id in sources
    }:
        raise ValueError("review coverage/uniqueness failure")
    details, stats, round_pairs = [], {}, []
    for name, predictions in readers.items():
        counter = collections.Counter()
        for row_id, prediction in predictions.items():
            source = sources[row_id]
            payload = verify_prediction(source, prediction)
            diag = source["diagnostic"]
            status = reviewed_status(diag["short_name"], diag["condition"], spec)
            action = expected_action(status, diag["round"])
            visible_ids = set(diag["evidence_id_to_corpus_id"])
            problems = protocol_problems(payload, visible_ids)
            counter["rows"] += 1
            counter["strict_json"] += 1
            counter["old_protocol_pass"] += not problems
            counter[f"{status}_rows"] += 1
            if action is not None:
                counter["known_action_rows"] += 1
                counter["known_action_correct"] += payload["next_action"] == action
                counter[f"{status}_action_correct"] += payload["next_action"] == action
            details.append({
                "reader": name, "id": row_id, "case": diag["short_name"],
                "condition": diag["condition"], "messages_sha256": digest(messages(source)),
                "raw_output_sha256": digest(prediction["raw_output"]),
                "old_review_sha256": digest(old_index[(name, row_id)]),
                "v2_status": status, "expected_action_v2": action,
                "actual_action": payload["next_action"],
                "old_protocol_problems": problems,
                "new_semantic_score": None,
                "reason_for_no_new_score": "No independent complete atom/citation re-review",
            })
        stats[name] = dict(counter)
        by_case = collections.defaultdict(dict)
        for row in predictions.values():
            by_case[row["diagnostic"]["short_name"]][row["diagnostic"]["condition"]] = row
        for short_name, rows in by_case.items():
            for left_name, right_name, kind in (
                ("renumbered_b_with_distractor", "restored_round2", "positive_context_round"),
                ("broken_round1", "broken_round2", "broken_context_round"),
            ):
                left, right = rows[left_name], rows[right_name]
                lp, rp = messages(left)[1]["content"], messages(right)[1]["content"]
                exact = left["system"] == right["system"] and normalize_round(lp) == normalize_round(rp)
                la, ra = load_payload(left)["next_action"], load_payload(right)["next_action"]
                round_pairs.append({
                    "reader": name, "case": short_name, "pair_kind": kind,
                    "left_id": left["id"], "right_id": right["id"],
                    "round_only_prompt_difference": exact,
                    "left_action": la, "right_action": ra,
                    "action_changed": la != ra,
                    "raw_output_equal": left["raw_output"] == right["raw_output"],
                    "left_messages_sha256": digest(messages(left)),
                    "right_messages_sha256": digest(messages(right)),
                    "left_output_sha256": digest(left["raw_output"]),
                    "right_output_sha256": digest(right["raw_output"]),
                    "no_previous_model_response_in_prompt": True,
                    "diff": list(difflib.unified_diff(lp.splitlines(), rp.splitlines(), n=0)),
                })
    bound_errata = []
    for item in spec["semantic_errata"]:
        matches = [row for row in readers[item["reader"]].values()
                   if row["diagnostic"]["short_name"] == item["case"]
                   and row["diagnostic"]["condition"] == item["condition"]]
        if len(matches) != 1:
            raise ValueError("erratum does not resolve uniquely")
        row = matches[0]
        bound_errata.append({
            **item, "id": row["id"], "messages_sha256": digest(messages(row)),
            "raw_output_sha256": digest(row["raw_output"]), "raw_output": row["raw_output"],
            "previous_review": old_index[(item["reader"], row["id"])],
        })
    return {
        "source_hashes": hashes, "audit_spec_sha256": file_digest(ROOT / "audit_spec.json"),
        "all_56_prediction_prompts_exactly_bound": True,
        "scoring_warning": "Known-action subset only, not QA accuracy or independently verified gold.",
        "stats": stats, "details": details, "round_pairs": round_pairs,
        "content_bound_semantic_errata": bound_errata,
    }


def load_payload(row):
    return json.loads(row["raw_output"], object_pairs_hook=no_duplicate_keys)


def question_only(question):
    return {"question": question, "query_type": "reasoning", "entities": [],
            "keywords": [], "expected_answer_type": "short_text"}


def make_row(short_name, question, hypothesis, round_value, documents, eid_values, reference, factors):
    if len(documents) != len(eid_values) or len(set(eid_values)) != len(eid_values):
        raise ValueError("invalid evidence mapping")
    evidence = "\n".join(f"[{eid}]\n{doc['text']}" for eid, doc in zip(eid_values, documents))
    user = render_exx_user_prompt(question=question, hypothesis=hypothesis,
                                 round_value=round_value, evidence_text=evidence)
    fields = {"question": question, "hypothesis": hypothesis, "round": round_value,
              "documents": documents, "evidence_ids": eid_values}
    msg = [{"role": "system", "content": EXX_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    reference = {**reference, "action": expected_action(reference["status"], round_value)}
    return {
        "id": f"{short_name}::{digest(msg)[:20]}",
        "task_type": "grounded_action_generation", "system": EXX_SYSTEM_PROMPT,
        "conversations": [{"from": "human", "value": user}],
        "diagnostic": {
            "evaluation_only": True, "generation_status": "not_run",
            "prompt_messages_sha256": digest(msg), "reference_sha256": digest(reference),
            "reference": reference, "factors": factors, "input_fields": fields,
            "case": short_name, "evidence_chars": len(evidence),
        },
    }


def changed_fields(a, b):
    af = a["diagnostic"]["input_fields"]
    bf = b["diagnostic"]["input_fields"]
    return [key for key in af if af[key] != bf[key]]


def pair(a, b, factor, allowed):
    changed = changed_fields(a, b)
    if set(changed) != set(allowed):
        raise ValueError(f"{factor}: uncontrolled field differences {changed}")
    return {"left": a["id"], "right": b["id"], "factor": factor, "changed_fields": changed}


def build_natural(spec):
    frozen = load(PILOT / "input/frozen_cases.json")["cases"]
    old = load(PILOT / "data/pilot_matrix.json")
    names = {r["diagnostic"]["source_id"]: r["diagnostic"]["short_name"] for r in old}
    rows, pairs = [], []
    for case in frozen:
        short = names[case["source_id"]]
        docs = {d["corpus_id"]: {"corpus_id": d["corpus_id"], "text": d["text"],
                               "text_sha256": d["text_sha256"]} for d in case["documents"]}
        for d in docs.values():
            if digest(d["text"]) != d["text_sha256"]:
                raise ValueError("frozen evidence text/hash mismatch")
        existing = {r["diagnostic"]["condition"]: r for r in old if r["diagnostic"]["source_id"] == case["source_id"]}
        contexts = {key: existing[key]["diagnostic"]["visible_corpus_ids"]
                    for key in ("proof_a", "proof_b", "dual_proof", "broken_round1")}
        # Append context AFTER B so B retains its original E-IDs. This is not length-matched.
        contexts["proof_b_plus_context"] = list(dict.fromkeys([
            *contexts["proof_b"], *case["distractor_evidence_ids"]]))
        lookup = {}
        for context, ids in contexts.items():
            documents = [docs[i] for i in ids]
            orders = ["canonical", "permuted"] if len(ids) > 1 else ["canonical"]
            for round_value, h_mode, order in itertools.product(
                ("1/2", "2/2"), ("inherited", "question_only"), orders
            ):
                hypothesis = case["hypothesis"] if h_mode == "inherited" else question_only(case["question"])
                eids = [f"E{i+1}" for i in range(len(ids))]
                if order == "permuted":
                    eids.reverse()  # Text and presentation order remain identical.
                status = reviewed_status(short, context, spec)
                reference = {
                    "status": status, "rubric": spec["cases"][short],
                    "independent_human_review": False, "exclude_from_training": True,
                    "source_case_sha256": digest(case),
                }
                row = make_row(short, case["question"], hypothesis, round_value, documents, eids,
                               reference, {"context": context, "round": round_value,
                                           "hypothesis": h_mode, "eid_order": order})
                lookup[(context, round_value, h_mode, order)] = row
                rows.append(row)
        for (context, rnd, hypo, order), row in lookup.items():
            if rnd == "1/2":
                pairs.append(pair(row, lookup[(context, "2/2", hypo, order)], "round", ["round"]))
            if hypo == "inherited":
                pairs.append(pair(row, lookup[(context, rnd, "question_only", order)],
                                  "whole_hypothesis_package", ["hypothesis"]))
            if order == "canonical" and (context, rnd, hypo, "permuted") in lookup:
                pairs.append(pair(row, lookup[(context, rnd, hypo, "permuted")],
                                  "eid_permutation", ["evidence_ids"]))
            if context == "proof_b" and order == "canonical":
                other = lookup[("proof_b_plus_context", rnd, hypo, order)]
                a = row["diagnostic"]["input_fields"]
                b = other["diagnostic"]["input_fields"]
                if b["documents"][:len(a["documents"])] != a["documents"] or b["evidence_ids"][:len(a["evidence_ids"])] != a["evidence_ids"]:
                    raise ValueError("context addition altered existing evidence")
                pairs.append(pair(row, other, "context_addition_not_length_matched",
                                  ["documents", "evidence_ids"]))
    index_unique(rows)
    return rows, pairs


def graph_paths(edges, root):
    """Two-hop proofs independently found from visible typed relations."""
    paths = []
    for a in edges:
        if a["relation"] != "routes_to" or a["subject"] != root:
            continue
        for b in edges:
            if b["relation"] == "has_code" and b["subject"] == a["object"]:
                paths.append({"answer": b["object"], "proof": [a["id"], b["id"]]})
    return paths


def truth(mask):
    return (mask & 3 == 3) or (mask & 12 == 12)


def minimum_repairs(mask):
    if truth(mask):
        return [[]]
    missing = [i for i in range(4) if not mask & (1 << i)]
    for size in range(1, len(missing) + 1):
        candidates = [list(s) for s in itertools.combinations(missing, size)
                      if truth(mask | sum(1 << i for i in s))]
        if candidates:
            return candidates
    raise AssertionError("no repair")


def synthetic_world(world_id, mask, tokens):
    root, other_a, other_b, sa, sb, off_a, off_b, da, db, answer, wrong_a, wrong_b = tokens
    # Nonces have equal character length. Keep both answer and decoy strings in EVERY state.
    edges = [
        {"id": "d1", "subject": root if mask & 1 else other_a, "relation": "routes_to", "object": sa},
        {"id": "d2", "subject": sa if mask & 2 else off_a, "relation": "has_code", "object": answer},
        {"id": "d3", "subject": root if mask & 4 else other_b, "relation": "routes_to", "object": sb},
        {"id": "d4", "subject": sb if mask & 8 else off_b, "relation": "has_code", "object": answer},
        {"id": "d5", "subject": da, "relation": "has_code", "object": wrong_a},
        {"id": "d6", "subject": db, "relation": "has_code", "object": wrong_b},
    ]
    documents = []
    for edge in edges:
        if edge["relation"] == "routes_to":
            text = f"箱号{edge['subject']}指向中转站{edge['object']}。"
        else:
            text = f"中转站{edge['subject']}的通行口令为{edge['object']}。"
        documents.append({"corpus_id": f"{world_id}/{edge['id']}", "text": text, "text_sha256": digest(text)})
    paths = graph_paths(edges, root)
    if bool(paths) != truth(mask) or any(p["answer"] != answer for p in paths):
        raise ValueError("formula/reachability oracle disagreement")
    question = (f"以下是虚构封闭登记簿。箱号{root}可经任一它指向的中转站取得通行口令。"
                "请给出该箱号的通行口令；必须同时有该箱号到中转站、中转站到口令的两条可见登记，"
                "不能仅凭候选口令猜测；不使用登记簿以外的知识。")
    return question, documents, paths, edges, answer


def build_synthetic():
    rng = random.Random(20260909)
    vocabulary = rng.sample(range(16**6), 8 * 12)
    rows, pairs, truth_table = [], [], []
    for w in range(8):
        tokens = [f"{n:06X}" for n in vocabulary[w*12:(w+1)*12]]
        lengths, lookup = set(), {}
        for mask in range(16):
            name = f"synthetic_development_{w:02d}"
            question, docs, paths, edges, answer = synthetic_world(name, mask, tokens)
            lengths.add(tuple(len(d["text"]) for d in docs))
            for rnd, order in itertools.product(("1/2", "2/2"), ("canonical", "permuted")):
                eids = [f"E{i}" for i in range(1, 7)]
                if order == "permuted":
                    eids.reverse()
                id_map = {f"d{i+1}": eid for i, eid in enumerate(eids)}
                proofs = [[id_map[d] for d in p["proof"]] for p in paths]
                reference = {
                    "status": "sufficient" if paths else "insufficient",
                    "synthetic_closed_world": True, "exclude_from_training": True,
                    "answer": answer if paths else None,
                    "candidate_answers": tokens[-3:],
                    "all_valid_citation_pairs": proofs, "visible_graph": edges,
                    "minimum_edge_repairs": minimum_repairs(mask),
                }
                row = make_row(name, question, question_only(question), rnd, docs, eids, reference,
                               {"mask": mask, "round": rnd, "eid_order": order})
                rows.append(row)
                lookup[(mask, rnd, order)] = row
        if len(lengths) != 1:
            raise ValueError("synthetic evidence length changed")
        for (mask, rnd, order), row in lookup.items():
            if rnd == "1/2":
                pairs.append(pair(row, lookup[(mask, "2/2", order)], "round", ["round"]))
            if order == "canonical":
                pairs.append(pair(row, lookup[(mask, rnd, "permuted")], "eid_permutation", ["evidence_ids"]))
            for bit in range(4):
                if not mask & (1 << bit):
                    other = lookup[(mask | (1 << bit), rnd, order)]
                    pairs.append(pair(row, other, f"repair_relation_{bit+1}", ["documents"]))
        if w == 0:
            truth_table = [{"mask": mask, "edge_bits_d1_to_d4": [bool(mask & 1 << i) for i in range(4)],
                            "answerable": truth(mask), "minimum_repairs": minimum_repairs(mask)}
                           for mask in range(16)]
    index_unique(rows)
    return rows, pairs, truth_table


def overlap_audit():
    result = []
    for case in load(PILOT / "input/frozen_cases.json")["cases"]:
        groups = case["answer_facts"][0]["support_groups"]
        if len(groups) < 2:
            continue
        docs = {d["corpus_id"]: d["text"] for d in case["documents"]}
        a, b = docs[groups[0][0]], docs[groups[1][0]]
        match = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match()
        result.append({
            "source_id": case["source_id"], "a_chars": len(a), "b_chars": len(b),
            "longest_exact_contiguous_match_chars": match.size,
            "match_fraction_of_shorter": round(match.size / min(len(a), len(b)), 6),
            "note": "Character overlap, not a semantic independence score",
        })
    return result


def main():
    spec = load(ROOT / "audit_spec.json")
    audit = audit_existing(spec)
    write(ROOT / "audits/previous_run_audit.json", audit)
    write(ROOT / "audits/evidence_overlap.json", overlap_audit())
    natural, natural_pairs = build_natural(spec)
    synth, synth_pairs, truth_table = build_synthetic()
    write(ROOT / "data/natural_controls.json", natural)
    write(ROOT / "data/natural_pairs.json", natural_pairs)
    write(ROOT / "data/synthetic_controls.json", synth)
    write(ROOT / "data/synthetic_pairs.json", synth_pairs)
    write(ROOT / "data/synthetic_truth_table.json", truth_table)
    manifest = {
        "generation_status": "not_run", "thinking": False, "training_started": False,
        "natural_rows": len(natural), "natural_pairs": len(natural_pairs),
        "natural_status_counts": dict(collections.Counter(r["diagnostic"]["reference"]["status"] for r in natural)),
        "natural_pair_types": dict(collections.Counter(p["factor"] for p in natural_pairs)),
        "synthetic_rows": len(synth), "synthetic_pairs": len(synth_pairs), "synthetic_worlds": 8,
        "synthetic_status_counts": dict(collections.Counter(r["diagnostic"]["reference"]["status"] for r in synth)),
        "synthetic_pair_types": dict(collections.Counter(p["factor"] for p in synth_pairs)),
        "all_controls_evaluation_only": True,
        "all_pairs_field_difference_checked": True,
        "synthetic_char_length_and_document_count_matched": True,
        "synthetic_token_length_matched": "not measured; character equality does not ensure token equality",
        "files": {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT / "data").glob("*.json"))},
        "builder_sha256": file_digest(Path(__file__)),
        "prompt_renderer_sha256": file_digest(REPO / "src/asa_arknight_story_agent/inference/generation/exx_prompt.py"),
    }
    write(ROOT / "CONTROL_MANIFEST.json", manifest)
    print(json.dumps({"old_audit": audit["stats"], "controls": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
