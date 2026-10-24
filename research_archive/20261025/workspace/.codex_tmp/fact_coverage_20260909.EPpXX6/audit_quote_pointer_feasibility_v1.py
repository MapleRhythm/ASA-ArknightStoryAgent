"""CPU-only representability diagnostic, never a generation input or relabel.

Find how many existing copied quotes could be represented by whole annotated
chunks while retaining the original2-ref/160/400 character limits. This is
not an oracle for factual truth or a runtime repair algorithm.
"""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import re
import sys

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save
from quote_pointer_protocol_v1 import HEADER, annotate, validate_catalog


def options_for(quote, eid, evidence, catalog):
    if not isinstance(quote, str) or not quote.strip() or not isinstance(eid, str):
        return []
    headers = list(HEADER.finditer(evidence))
    block = next(((h.end(), headers[i + 1].start() if i + 1 < len(headers) else len(evidence))
                  for i, h in enumerate(headers) if "E" + h.group(1) == eid), None)
    if block is None:
        return []
    choices, start = {}, block[0]
    while True:
        position = evidence.find(quote, start, block[1])
        if position < 0:
            break
        matching = [r for r in catalog if r["evidence_id"] == eid
                    and r["start"] < position + len(quote) and r["end"] > position]
        if matching and matching[0]["start"] <= position and matching[-1]["end"] >= position + len(quote):
            key = tuple(r["quote_id"] for r in matching)
            choices[key] = sum(len(r["quote"]) for r in matching)
        start = position + 1
    return [{"quote_ids": list(key), "quote_chars": length} for key, length in sorted(
        choices.items(), key=lambda item: (len(item[0]), item[1], item[0]))]


def best_union(option_lists, catalog):
    if not option_lists or not all(option_lists):
        return None
    byid = {r["quote_id"]: r for r in catalog}
    # Original protocol permits at most2 refs per fact. Larger malformed
    # outputs are retained as not comparable, not silently shortened.
    if len(option_lists) > 2:
        return None
    best = None
    for choices in product(*option_lists):
        ids = sorted(set(qid for choice in choices for qid in choice["quote_ids"]))
        length = sum(len(byid[qid]["quote"]) for qid in ids)
        candidate = {"quote_ids": ids, "quote_chars": length, "fits_fact_limits": len(ids) <= 2 and length <= 160}
        key = (not candidate["fits_fact_limits"], len(ids), length, ids)
        if best is None or key < best[0]:
            best = key, candidate
    return best[1]


def main():
    destination = HERE / "quote_pointer_feasibility_v1.json"
    assert not destination.exists()
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.grounding.quote_match_utils import normalize_for_evidence_match
    files = {
        "original4b58": HERE / "question_echo_paired116_replay_v1.json",
        "native8b58": HERE / "question_echo_native8b58_replay_v1.json",
    }
    records, summaries = [], {}
    for arm, path in files.items():
        results = load(path)["results"]
        if arm == "original4b58":
            results = [r for r in results if r["renderer"] == "original_prompt_v1"]
        assert len(results) == 58
        counts = Counter(states=58)
        for index, row in enumerate(results):
            evidence = row["evidence_brief"]
            _, catalog = annotate(evidence)
            validate_catalog(evidence, catalog)
            refs_in_state, facts, total = [], [], 0
            all_eligible, fact_count = True, 0
            blocks = {}
            for r in catalog:
                blocks.setdefault(r["evidence_id"], []).append(r["quote"])
            for field in ("supported_facts", "inferred_facts"):
                for fi, fact in enumerate(row.get("parsed", {}).get(field, [])):
                    fact_count += 1
                    counts["facts"] += 1
                    refs = fact.get("evidence_refs", []) if isinstance(fact, dict) else []
                    choices = []
                    ref_records = []
                    for ri, ref in enumerate(refs):
                        counts["refs"] += 1
                        if not isinstance(ref, dict):
                            ref = {}
                        quote, eid = ref.get("quote", ""), ref.get("evidence_id", "")
                        options = options_for(quote, eid, evidence, catalog)
                        normalized = normalize_for_evidence_match(quote)
                        normalized_found = bool(normalized) and normalized in normalize_for_evidence_match(
                            "".join(blocks.get(eid, [])))
                        counts["exact_source_refs"] += bool(options)
                        counts["canonical_match_refs"] += normalized_found
                        counts["exact_source_but_canonical_not_matched_refs"] += bool(options) and not normalized_found
                        counts["canonical_only_not_exact_refs"] += normalized_found and not bool(options)
                        counts["exact_refs_representable_with2_chunks"] += bool(options) and any(
                            len(o["quote_ids"]) <= 2 and o["quote_chars"] <= 160 for o in options)
                        record = {"ref_index": ri, "quote": quote, "evidence_id": eid,
                                  "exact_source_match": bool(options), "canonical_match": normalized_found,
                                  "pointer_choices": options}
                        ref_records.append(record)
                        refs_in_state.append(record)
                        choices.append(options)
                    union = best_union(choices, catalog)
                    eligible = union is not None and union["fits_fact_limits"]
                    counts["facts_existing_quotes_representable"] += eligible
                    all_eligible = all_eligible and eligible
                    total += union["quote_chars"] if eligible else 0
                    facts.append({"field": field, "fact_index": fi,
                                  "fact": fact.get("fact") if isinstance(fact, dict) else fact,
                                  "original_ref_count": len(refs), "refs": ref_records,
                                  "best_pointer_union": union, "eligible_under_fact_limits": eligible})
            all_eligible = all_eligible and fact_count > 0
            counts["states_all_existing_fact_quotes_representable"] += all_eligible
            counts["states_also_fit_quote400"] += all_eligible and total <= 400
            records.append({
                "arm": arm, "index": index, "id": row["id"], "question": row["question"],
                "source_sha256": digest(evidence), "old_action": row.get("grounded", {}).get("next_action", "error"),
                "facts": facts, "all_existing_quotes_fit_fact_limits": all_eligible,
                "expanded_quote_chars_if_all_fit": total if all_eligible else None,
                "all_existing_quotes_fit_answer_limits": all_eligible and total <= 400,
            })
        summaries[arm] = dict(counts)
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "source_artifacts_sha256": {str(p): sha(p) for p in files.values()},
        "protocol_sha256": sha(HERE / "quote_pointer_protocol_v1.py"),
        "script_sha256": sha(Path(__file__)), "summary": summaries, "records": records,
        "new_generations": 0, "sent_to_generation": False, "old_labels_changed": False,
        "limits": [
            "Existing exact quotes measure pointer representability, not truth or future acceptance/accuracy.",
            "Whitespace/ellipsis-normalized-only matches counted separately, never silently rewritten.",
            "Invalid or more-than2 original refs are not repaired, deleted or treated as representable.",
            "Whole80-character chunks may consume extra160/400 budget, even for a short original quote.",
            "Does not predict whether the model will select appropriate pointers or write supported facts.",
        ],
    })
    print(json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
