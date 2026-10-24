"""Auditable positive-fact pilot from actual cached GLM API receipts.

Original 6/12-document evidence and labels are preserved. Neighbor evidence is
appended to reach 20; labels are NONEXHAUSTIVE positive facts, not fresh
20-document sufficiency/coverage annotations. Never reuse abstain/retrieve.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from tokenizers import Tokenizer
from fresh_glm_data import SYSTEM, digest, excluded, family, validate_target
from prepare_pilot import RULES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=40)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=13041)
    ap.add_argument("--review-exclusions")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(args.out)
    out.mkdir(exist_ok=False)
    source = Path(args.source)
    review_exclusions = json.loads(Path(args.review_exclusions).read_text()) if args.review_exclusions else {}
    rejected_ids = set(review_exclusions.get("excluded_ids", []))
    tasks = {t["task_id"]: t for t in json.loads((source / "tasks.json").read_text())}
    index_hash = hashlib.sha256()
    docs, groups = {}, {}
    with open(args.documents, "rb") as stream:
        for line in stream:
            index_hash.update(line)
            doc = json.loads(line)
            docs[doc["id"]] = doc
            groups.setdefault(family(doc["id"]), []).append(doc)
    tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
    rejects, candidates, seen_questions = Counter(), [], set()
    for line in (source / "labels.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["task_id"] in rejected_ids:
            rejects["assistant_semantic_review_rejected_whole_row"] += 1
            continue
        if record["api"].get("model") != "glm-5.3" or not record.get("semantic_verifier", {}).get("valid"):
            rejects["unverified_provider_or_label"] += 1
            continue
        target = record["label"]
        if target.get("next_action") != "answer_directly":
            rejects["not_positive_fact_label"] += 1
            continue
        try:
            validate_target(target)
        except ValueError:
            rejects["bad_target_schema"] += 1
            continue
        task = tasks[record["task_id"]]
        evidence = task["evidence"]
        if not all(e["doc_id"] in docs and e["text"] == docs[e["doc_id"]]["clean_text"] for e in evidence):
            rejects["original_document_not_full_exact"] += 1
            continue
        if len({e["doc_id"] for e in evidence}) != len(evidence):
            rejects["duplicate_original_document"] += 1
            continue
        families = {family(e["doc_id"]) for e in evidence}
        if len(families) != 1 or any(excluded(e["doc_id"]) for e in evidence):
            rejects["multifamily_or_excluded"] += 1
            continue
        group = next(iter(families))
        original_ids = {e["doc_id"] for e in evidence}
        original_labels = {e["label"] for e in evidence}
        if original_labels != {f"E{i}" for i in range(1, len(evidence) + 1)}:
            rejects["noncontiguous_old_labels"] += 1
            continue
        if any(ref not in original_labels for fact in target["supported_facts"] for ref in fact["evidence_ids"]):
            rejects["original_label_reference_missing"] += 1
            continue
        pool = [d for d in groups[group] if d["id"] not in original_ids and not excluded(d["id"])]
        def distance(doc):
            chapter, _, chunk = doc["id"].partition("#chunk-")
            same = [e["doc_id"] for e in evidence if e["doc_id"].partition("#chunk-")[0] == chapter]
            delta = min((abs(int(chunk) - int(i.partition("#chunk-")[2]))
                         for i in same if chunk.isdigit() and i.partition("#chunk-")[2].isdigit()),
                        default=999999)
            return not bool(same), delta, doc["id"]
        pool.sort(key=distance)
        selected = [{"evidence_id": e["label"], "document_id": e["doc_id"],
                     "text": e["text"], "sha256": digest(e["text"])} for e in evidence]
        for doc in pool[:20 - len(selected)]:
            selected.append({"evidence_id": f"E{len(selected)+1}", "document_id": doc["id"],
                             "text": doc["clean_text"], "sha256": digest(doc["clean_text"])})
        if len(selected) != 20:
            rejects["not_enough_documents"] += 1
            continue
        question = task["question"]
        if question in seen_questions:
            rejects["duplicate_question"] += 1
            continue
        body = (f"task: grounded_action_generation\nquestion: {question}\n"
                'hypothesis: {"query_type":"reasoning","expected_answer_type":"剧情解释"}\n'
                f"round: {task['round']}\nevidence:\n" +
                "\n".join(f"[{d['evidence_id']}]\n{d['text']}" for d in selected) +
                "\noutput_schema: grounded_action_exx_v1\nrules: " + RULES)
        prompt = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n" +
                  body + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        target_text = json.dumps(target, ensure_ascii=False)
        length = (len(tokenizer.encode(prompt, add_special_tokens=False).ids) +
                  len(tokenizer.encode(target_text + "<|im_end|>", add_special_tokens=False).ids) + 16)
        if length > args.max_tokens:
            rejects["whole_sample_over_tested_memory_budget_not_truncated"] += 1
            continue
        assert all(d["text"] == docs[d["document_id"]]["clean_text"] for d in selected)
        candidates.append({
            "id": task["task_id"], "question": question, "prompt": prompt, "target": target_text,
            "documents": selected, "evidence_count": 20, "truncated": False,
            "source_family": group, "source_label": "cached_GLM_API_positive_facts",
            "source_api": record["api"], "label_source_record_sha256": digest(line),
            "semantic_verifier": record["semantic_verifier"],
            "label_conditioned_document_count": len(evidence), "evidence_expanded_after_label": True,
            "nonexhaustive_positive_fact_supervision": True, "encoded_tokens_including_prompt": length,
        })
        seen_questions.add(question)
    by_family = {}
    for row in candidates:
        by_family.setdefault(row["source_family"], []).append(row)
    names = sorted(by_family)
    random.Random(20261013).shuffle(names)
    val_families, count = [], 0
    for name in names:
        if count < args.val:
            val_families.append(name)
            count += len(by_family[name])
    # Round-robin across activities so a large family cannot fill the pilot.
    def sample(group_names, count):
        rows = []
        position = 0
        while len(rows) < count:
            added = False
            for name in group_names:
                if position < len(by_family[name]):
                    rows.append(by_family[name][position])
                    added = True
                    if len(rows) == count:
                        return rows
            if not added:
                raise RuntimeError(f"insufficient safe cached rows: {len(candidates)}")
            position += 1
        return rows
    splits = {"train": sample([n for n in names if n not in val_families], args.train),
              "val": sample(val_families, args.val)}
    family_sets = {s: {r["source_family"] for r in rows} for s, rows in splits.items()}
    doc_sets = {s: {d["document_id"] for r in rows for d in r["documents"]} for s, rows in splits.items()}
    assert not family_sets["train"] & family_sets["val"]
    assert not doc_sets["train"] & doc_sets["val"]
    for split, rows in splits.items():
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "kind": "cached_GLM_positive_facts_context_expansion_pilot",
        "source": str(source), "source_index": args.documents,
        "source_index_sha256": index_hash.hexdigest(),
        "source_files_sha256": {n: hashlib.sha256((source / n).read_bytes()).hexdigest()
                                for n in ("tasks.json", "labels.jsonl")},
        "train": args.train, "val": args.val, "candidate_count": len(candidates),
        "rejected": dict(rejects), "evidence_count_per_example": 20,
        "truncated_documents": 0, "train_val_story_overlap": [],
        "train_val_source_family_overlap": [], "train_val_document_overlap": [],
        "families": {s: sorted(v) for s, v in family_sets.items()},
        "new_api_calls": 0, "labels_provider": "verified_cached_GLM_API",
        "teacher_model": "glm-5.3", "complete": True,
        "positive_only": True, "label_conditioned_document_counts": [6, 12],
        "evidence_expanded_after_label": True, "original_texts_and_targets_preserved": True,
        "nonexhaustive_positive_fact_supervision": True,
        "new_20_document_labels_not_available_due_to_API_balance": True,
        "not_human_gold": True, "not_unseen_benchmark": True,
        "not_full_chain_retrieval_evaluation": True, "no_coverage_accuracy_claim": True,
        "max_tokens_including_soft_prompt": max(r["encoded_tokens_including_prompt"] for rr in splits.values() for r in rr),
        "sample_length_budget": args.max_tokens,
        "review_required_before_training": True,
        "assistant_review_exclusions": review_exclusions,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
