#!/usr/bin/env python3
"""Validate binding-target pair candidates (BINDING_TARGET_SPEC.md); never edits or merges data.

Input pair files contain story text and stay private. The aggregate printed or
written here holds counts and pair IDs only. A record with
ready_for_training=true is refused: promotion needs explicit user authorization
outside this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HAZARDS = ("unlabelled_speaker", "pronoun_referent", "shared_predicate",
           "alias_or_namesake", "cross_stage_entity", "allegation_vs_fact")
# Substitution edits: the wrong clause is the correct clause with one entity replaced.
SUBSTITUTIONS = ("speaker_swap", "addressee_swap", "participant_substitution", "identity_merge")
PERTURBATIONS = SUBSTITUTIONS + ("actor_patient_reversal", "phase_splice", "scene_transfer",
                                 "attribution_flip")
ITEM_KINDS = ("uncued_binding", "contrast_twin", "probe_only")
MAX_SHARE = 0.10
MAX_PER_FAMILY = 3


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def family(document_id):
    return document_id.split("#chunk-")[0]


def exclusions(task_paths):
    """Passages and source families of evaluation sets; training-side pairs may touch neither."""
    texts, families = set(), set()
    for path in task_paths:
        for task in rows(path):
            for doc in task["documents"]:
                texts.add(sha(doc["text"]))
                families.add(family(doc["document_id"]))
    return texts, families


def check_target(item, pair, errors):
    pid = pair["pair_id"]
    try:
        target = json.loads(item["target"])
    except (TypeError, ValueError):
        errors.append((pid, "target is not native JSON"))
        return []
    facts = target.get("supported_facts", [])
    if target.get("next_action") != "answer_directly" or not 1 <= len(facts) <= 8:
        errors.append((pid, "target must be answer_directly with 1-8 facts"))
    ids = {f"E{i}" for i in range(1, 21)}
    if not all(1 <= len(f.get("evidence_ids", [])) <= 2 and set(f["evidence_ids"]) <= ids for f in facts):
        errors.append((pid, "each fact needs 1-2 valid evidence IDs"))
    texts = [f.get("fact") for f in facts]
    if pair["correct_clause"] not in texts:
        errors.append((pid, "correct clause absent from target"))
    if pair["wrong_clause"] in texts:
        errors.append((pid, "wrong clause used as a training target"))
    if not item.get("necessary_slots"):
        errors.append((pid, "necessary slots must be frozen"))
    return texts


def validate(pairs, excluded_texts, excluded_families, rubric_sha, base_rows):
    errors = []
    by_id = {}
    for pair in pairs:
        pid = pair.get("pair_id")
        if pid in by_id:
            errors.append((pid, "duplicate pair_id"))
        by_id[pid] = pair
    wrong_clauses = {p["wrong_clause"] for p in pairs}
    family_use = Counter()
    for pair in pairs:
        pid = pair["pair_id"]
        bad = lambda message: errors.append((pid, message))
        if pair.get("ready_for_training") is not False:
            bad("ready_for_training must be false; merging needs user authorization")
        review = pair.get("source_review") or {}
        if not (review.get("label_supported") is True and review.get("decision_appropriate") is True
                and review.get("issues") == []):
            bad("source row lacks a passing recorded review")
        docs = pair.get("documents", [])
        if len(docs) != 20 or len({d["evidence_id"] for d in docs}) != 20:
            bad("must keep the source row's 20 passages")
        if any(sha(d["text"]) != d["sha256"] for d in docs):
            bad("passage hash mismatch")
        if any(sha(d["text"]) in excluded_texts for d in docs):
            bad("passage shared with an evaluation set")
        families = {family(d["document_id"]) for d in docs}
        if families & excluded_families:
            bad("source family shared with an evaluation set")
        family_use.update(families)
        text_by_eid = {d["evidence_id"]: d["text"] for d in docs}
        anchors = pair.get("anchors") or []
        if not anchors or not all(a["evidence_id"] in text_by_eid and
                                  0 <= a["char_range"][0] < a["char_range"][1] <= len(text_by_eid[a["evidence_id"]])
                                  for a in anchors):
            bad("role cue anchors must point inside the passages")
        if pair.get("hazard") not in HAZARDS:
            bad("unknown hazard")
        kind = pair.get("perturbation")
        correct, wrong = pair.get("correct_clause", ""), pair.get("wrong_clause", "")
        if kind not in PERTURBATIONS:
            bad("unknown perturbation")
        elif not correct or correct == wrong:
            bad("wrong clause must differ from correct clause")
        elif kind in SUBSTITUTIONS:
            source, replacement = pair.get("swap_from", ""), pair.get("swap_to", "")
            if not source or correct.replace(source, replacement, 1) != wrong:
                bad("substitution must be a single entity replacement")
            if not any(replacement in text for text in text_by_eid.values()):
                bad("swapped-in entity must be a confusable present in the same passages")
        elif kind == "actor_patient_reversal":
            if not all(name in clause for name in (pair.get("swap_from", ""), pair.get("swap_to", ""))
                       for clause in (correct, wrong)):
                bad("reversal keeps both parties in both clauses")
        reviewers = {r["reviewer_id"] for r in pair.get("reviews", [])
                     if r.get("all20_checked") is True and r.get("correct_supported") is True
                     and r.get("wrong_false") is True and r.get("reviewer_id")}
        if len(reviewers) < 2:
            bad("two distinct passing reviews required")
        item_kind = pair.get("item_kind")
        if item_kind not in ITEM_KINDS:
            bad("unknown item kind")
        elif item_kind != "probe_only":
            item = pair.get("training_item") or {}
            if item.get("slot_rubric_sha256") != rubric_sha:
                bad("training item slots not written under the frozen SLOT_RUBRIC")
            # Fact-level equality: an attribution flip's wrong clause is often a substring of the right one.
            if set(check_target(item, pair, errors)) & wrong_clauses:
                bad("a wrong clause is a fact of a training target")
        if item_kind == "contrast_twin":
            twin = by_id.get(pair.get("twin_of"))
            if not twin or [d["sha256"] for d in twin.get("documents", [])] != [d["sha256"] for d in docs]:
                bad("contrast twin must reference a pair on the identical passages")
    trainable = [p for p in pairs if p.get("item_kind") in ("uncued_binding", "contrast_twin")]
    if len(trainable) > MAX_SHARE * base_rows:
        errors.append((None, f"trainable items exceed {MAX_SHARE:.0%} of base rows"))
    for name, count in family_use.items():
        if count > MAX_PER_FAMILY:
            errors.append((None, f"source family used by {count} pairs (max {MAX_PER_FAMILY})"))
    return dict(
        kind="binding_pair_validation", semantic_accuracy_measured=False, ready_for_training=False,
        pairs=len(pairs), trainable_items=len(trainable), probe_only=len(pairs) - len(trainable),
        by_hazard=dict(Counter(p.get("hazard") for p in pairs)),
        by_perturbation=dict(Counter(p.get("perturbation") for p in pairs)),
        by_item_kind=dict(Counter(p.get("item_kind") for p in pairs)),
        valid=not errors, errors=[{"pair_id": pid, "error": message} for pid, message in errors])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="private candidate JSONL")
    parser.add_argument("--exclude-tasks", action="append", required=True,
                        help="evaluation task JSONL (dev46, held-out); repeatable")
    parser.add_argument("--slot-rubric", required=True, help="frozen SLOT_RUBRIC.md")
    parser.add_argument("--base-rows", type=int, default=1117)
    parser.add_argument("--aggregate-out")
    args = parser.parse_args()
    texts, families = exclusions(args.exclude_tasks)
    rubric_sha = hashlib.sha256(Path(args.slot_rubric).read_bytes()).hexdigest()
    result = validate(rows(args.pairs), texts, families, rubric_sha, args.base_rows)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.aggregate_out:
        Path(args.aggregate_out).write_text(text)
    print(text, end="")
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
