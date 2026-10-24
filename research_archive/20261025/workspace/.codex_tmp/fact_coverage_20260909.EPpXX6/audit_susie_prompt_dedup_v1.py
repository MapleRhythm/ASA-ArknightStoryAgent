"""Exact shadow trace of canonical greedy similarity dedup and prompt ranking."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot


def main():
    destination = HERE / "susie_prompt_dedup_trace_v1.json"
    assert not destination.exists()
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.evidence.texts import (
        best_prompt_text, evidence_identity, prompt_evidence_score,
    )
    from asa_arknight_story_agent.inference.evidence.prompt_similarity import (
        text_similarity_tokens, jaccard_similarity, dedupe_prompt_evidence_candidates,
    )
    path = HERE / "susie_prompt_handoff_v1.json"
    artifact = load(path)
    source_before = snapshot(SOURCE)
    assert source_before == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
    phrase = "她依然没有放手"
    conditions = []
    for condition in artifact["conditions"]:
        candidates = condition["final32"]
        accepted, token_sets, identities, trace = [], [], set(), []
        for index, item in enumerate(candidates):
            identity = evidence_identity(item)
            text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
            tokens = text_similarity_tokens(text)
            near = next(((j, jaccard_similarity(tokens, known))
                         for j, known in enumerate(token_sets)
                         if tokens and jaccard_similarity(tokens, known) >= 0.82), None)
            entry = {
                "index": index, "document_id": item["document"]["id"],
                "prompt_score": prompt_evidence_score(item),
                "text_sha256": digest(text), "target_phrase_present": phrase in text,
            }
            if identity in identities:
                entry["status"] = "duplicate_identity"
            elif near is not None:
                j, similarity = near
                winner = accepted[j]
                winner_text = best_prompt_text(winner, prefer_direct=bool(winner.get("prompt_prefer_clean_text")))
                entry.update(
                    status="near_duplicate", earlier_document_id=winner["document"]["id"],
                    jaccard=similarity, earlier_has_target_phrase=phrase in winner_text,
                    earlier_prompt_score=prompt_evidence_score(winner),
                )
            else:
                assert tokens, "this trace requires nonempty source tokens"
                identities.add(identity)
                accepted.append(item)
                token_sets.append(tokens)
                entry["status"] = "kept"
            trace.append(entry)
        assert accepted == dedupe_prompt_evidence_candidates(candidates)
        ranked = sorted(enumerate(accepted), key=lambda p: (prompt_evidence_score(p[1]), -p[0]), reverse=True)
        selected = [r for _, r in ranked[:10]]
        assert selected == condition["selected"], "shadow differs from canonical prompt preparation"
        conditions.append({
            "scoring_condition": condition["scoring_condition"],
            "order_only_reserve6": condition["order_only_reserve6"],
            "trace": trace, "unique_after_dedup": len(accepted),
            "post_dedup_prompt_ranking": [
                {"rank": i, "document_id": r["document"]["id"], "score": prompt_evidence_score(r),
                 "target_phrase_present": phrase in best_prompt_text(r)}
                for i, (_, r) in enumerate(ranked, 1)],
            "critical_source_removed_by_near_dup": [
                r for r in trace if r["status"] == "near_duplicate"
                and r["target_phrase_present"] and not r["earlier_has_target_phrase"]],
            "target_visible_after_packing": condition["target_phrase_in_visible_prompt"],
        })
    assert snapshot(SOURCE) == source_before
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "conditions": conditions,
        "canonical_dedup_and_selection_reproduced": True, "target_phrase_used_only_for_diagnosis": True,
        "source_handoff_sha256": sha(path), "script_sha256": sha(Path(__file__)),
        "runtime_source_sha256": source_before, "production_dedup_changed": False,
        "new_answer_accuracy": None,
    })
    print(json.dumps([{
        "condition": c["scoring_condition"], "order_reserve6": c["order_only_reserve6"],
        "after_dedup": c["unique_after_dedup"],
        "dropped_critical": c["critical_source_removed_by_near_dup"],
        "critical_visible": c["target_visible_after_packing"],
    } for c in conditions], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
