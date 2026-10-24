"""Predeclared exact/containment alternatives, measured before any promotion."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot


def main():
    prepath = HERE / "prompt_dedup_policies_preflight_v1.json"
    output = HERE / "prompt_dedup_policies_v1.json"
    assert not prepath.exists() and not output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.inference.evidence.texts import (
        best_prompt_text, prompt_evidence_score, evidence_identity)
    from asa_arknight_story_agent.inference.evidence.prompt_similarity import (
        text_similarity_tokens, jaccard_similarity)
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    handoffpath = HERE / "susie_prompt_handoff_v1.json"
    querypre = HERE / "susie_prompt_handoff_preflight_v1.json"
    pairpath = HERE / "fact_coverage_pairs116_v1.matrix.json"
    cfgpath = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    cfg = load(cfgpath)
    assert cfg == load(querypre)["config"]
    source = snapshot(SOURCE)
    assert source == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
    artifacts = {str(p): sha(p) for p in (handoffpath, querypre, pairpath, cfgpath, Path(__file__))}
    save(prepath, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_source_sha256": source, "artifacts_sha256": artifacts,
        "policies": ["score_priority_near082", "score_priority_exact", "score_priority_containment"],
        "top_k": cfg["inference"]["prompt_evidence_top_k"],
        "evidence_budget_chars": cfg["inference"]["prompt_conclusion_evidence_max_total_chars"],
        "protocol": [
            "All six existing final32 states; fixed canonical top10/whole-block12000 packing.",
            "Preserve original score formula and complete evidence text; no changes to source/generator.",
            "Exact drops only byte-identical best_prompt_text; containment only drops literal substring.",
            "Common identity remains a duplicate under all policies; differing-text collisions are reported.",
            "Diagnostic phrase never enters ranking/dedup/packing decisions.",
            "No policy promotion or accuracy claim based on this one development question.",
        ],
        "training_ready": False, "evaluation_only": True, "new_model_calls": 0,
    })
    query = load(querypre)["question"]
    row = next(r for r in load(pairpath)["rows"]
               if r["question"] == query and r["renderer"] == "original_prompt_v1")
    hypothesis = HypothesisDocument(**row["hypothesis"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "model/qwen3.5-4b"), local_files_only=True, trust_remote_code=False)
    phrase, results = "她依然没有放手", []
    for condition in load(handoffpath)["conditions"]:
        original = condition["final32"]
        for policy in load(prepath)["policies"]:
            ranked = sorted(enumerate(original), key=lambda p: (-prompt_evidence_score(p[1]), p[0]))
            accepted, trace = [], []
            for original_index, item in ranked:
                text = best_prompt_text(item, prefer_direct=bool(item.get("prompt_prefer_clean_text")))
                tokens = text_similarity_tokens(text)
                blocker, reason, similarity = None, None, None
                for existing in accepted:
                    seen = best_prompt_text(existing, prefer_direct=bool(existing.get("prompt_prefer_clean_text")))
                    if evidence_identity(item) == evidence_identity(existing):
                        blocker, reason = existing, "same_identity"
                        break
                    if policy == "score_priority_near082":
                        similarity = jaccard_similarity(tokens, text_similarity_tokens(seen))
                        duplicate = bool(tokens) and similarity >= 0.82
                    elif policy == "score_priority_exact":
                        duplicate = bool(text) and text == seen
                    else:
                        duplicate = bool(text) and text in seen
                    if duplicate:
                        blocker, reason = existing, policy
                        break
                entry = {
                    "document_id": item["document"]["id"], "original_index": original_index,
                    "score": prompt_evidence_score(item), "has_diagnostic_phrase": phrase in text,
                    "text_sha256": digest(text), "status": "kept" if blocker is None else "dropped",
                    "reason": reason,
                }
                if blocker is None:
                    accepted.append(item)
                else:
                    seen = best_prompt_text(blocker, prefer_direct=bool(blocker.get("prompt_prefer_clean_text")))
                    entry.update(
                        blocker_id=blocker["document"]["id"], blocker_score=prompt_evidence_score(blocker),
                        blocker_has_diagnostic_phrase=phrase in seen,
                        identical_text=text == seen, literal_containment=text in seen,
                        jaccard=jaccard_similarity(tokens, text_similarity_tokens(seen)),
                    )
                trace.append(entry)
            selected = accepted[:cfg["inference"]["prompt_evidence_top_k"]]
            packing = []
            prompt, brief = build_conclusion_prompt(
                question=query, current_hypothesis=hypothesis,
                evidence=deepcopy(original), prompt_evidence=deepcopy(selected), retrieval_trace=[],
                current_round=row["current_round"], max_retrieval_rounds=row["max_retrieval_rounds"],
                prompt_evidence_top_k=cfg["inference"]["prompt_evidence_top_k"],
                evidence_max_chars_per_doc=cfg["inference"]["prompt_evidence_max_chars_per_doc"],
                evidence_max_total_chars=cfg["inference"]["prompt_conclusion_evidence_max_total_chars"],
                prompt_mode=cfg["inference"]["conclusion_prompt_mode"],
                grounding_mode=cfg["inference"]["answer_grounding_mode"], evidence_packing_trace=packing)
            input_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            results.append({
                "scoring_condition": condition["scoring_condition"],
                "order_only_reserve6": condition["order_only_reserve6"], "policy": policy,
                "trace": trace, "unique_after_dedup": len(accepted),
                "unique_ranking": [
                    {"document_id": r["document"]["id"], "score": prompt_evidence_score(r),
                     "has_diagnostic_phrase": phrase in best_prompt_text(r)}
                    for r in accepted],
                "selected": selected, "selected_ids": [r["document"]["id"] for r in selected],
                "target_selected": any(phrase in best_prompt_text(r) for r in selected),
                "target_visible": phrase in brief, "previous_target_visible": condition["target_phrase_in_visible_prompt"],
                "packing_trace": packing, "runtime_prompt": prompt, "evidence_brief": brief,
                "input_tokens": input_tokens,
                "fits_configured_budget": input_tokens + cfg["generator"]["max_tokens"] <= cfg["generator"]["ctx_size"],
                "prompt_sha256": digest(prompt), "new_answer": None,
            })
    assert snapshot(SOURCE) == source and all(sha(Path(p)) == h for p, h in artifacts.items())
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "preflight_sha256": sha(prepath),
        "conditions": results, "canonical_source_unchanged": True,
        "new_model_calls": 0, "independent_questions": 1, "evaluation_only": True,
        "training_ready": False, "new_full_rag_recall": None, "new_answer_accuracy": None,
    })
    print(json.dumps([{k: r[k] for k in [
        "scoring_condition", "order_only_reserve6", "policy", "unique_after_dedup",
        "target_selected", "target_visible", "input_tokens", "fits_configured_budget"]}
        for r in results], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
