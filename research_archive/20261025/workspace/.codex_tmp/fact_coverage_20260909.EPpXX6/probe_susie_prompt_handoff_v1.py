"""Execute canonical rerank/rescue/prompt suffix with exact saved CPU scores.

Fixed sparse120, no dense/fusion/front-end hypothesis generation or answer call.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot
from compare_chain_seed_preservation_v1 import build_retriever, fresh_hits
from explicit_speaker_reserve_v1 import reserve_ranked

PREFLIGHT = HERE / "susie_prompt_handoff_preflight_v1.json"
DESTINATION = HERE / "susie_prompt_handoff_v1.json"
CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
TARGET = "activities/act10mini/level_act10mini_st05#chunk-0025"


def main():
    assert not PREFLIGHT.exists() and not DESTINATION.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.config import QueryConfig
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.retrieval.finalization import (
        build_final_rerank_query, rerank_and_rescue_hits,
    )
    from asa_arknight_story_agent.inference.evidence.prompt_preparation import prepare_prompt_evidence
    from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
    from asa_arknight_story_agent.inference.planning.query_understanding import extract_content_tokens
    from asa_arknight_story_agent.inference.anchors.terms import extract_action_targets

    cfg = load(CONFIG)
    inf, rcfg = cfg["inference"], cfg["retrieval"]
    assert all(inf[name] is False for name in [
        "enable_evidence_pinning", "enable_mmr", "enable_crag_refinement", "enable_pyramid_order"])
    assert inf["web_context"]["enabled"] is False
    pipeline = SimpleNamespace(**{name: inf[name] for name in [
        "enable_evidence_pinning", "enable_mmr", "enable_crag_refinement", "enable_pyramid_order",
        "prompt_evidence_top_k", "mmr_lambda"]},
        web_context_config=SimpleNamespace(force_prompt_evidence=inf["web_context"]["force_prompt_evidence"]))
    # This suffix reads only these two QueryConfig fields; no front-end
    # retrieval is run or implicitly substituted by QueryConfig defaults.
    query_config = QueryConfig(rerank_top_k=rcfg["rerank_top_k"], rerank_batch_size=rcfg["rerank_batch_size"])
    control_pre = load(HERE / "susie_reranker120_preflight_v1.json")
    candidates, query = control_pre["candidates"], control_pre["query"]
    doc_scores = load(HERE / "susie_reranker120_v1.json")["scores_in_sparse_order"]
    source_row = next(r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
                      if r["question"] == query and r["renderer"] == "original_prompt_v1")
    hypothesis = HypothesisDocument(**source_row["hypothesis"])
    resolved, rerank_query, related = build_final_rerank_query(query, hypothesis)
    assert rerank_query == query
    old = load(HERE / "susie_chain_inputs_v1.json")
    pair = load(HERE / "susie_chain_complete_pair4_v1.json")
    old_maps = {}
    for arm in ("current_prefix1024", "complete3072"):
        old_maps[arm] = {
            c["text"]: next(s["score"] for s in pair["results"] if s["index"] == c["index"] and s["arm"] == arm)
            for c in old["chains"]
        }
    seed_pre = load(HERE / "susie_seed_retention_preflight_v1.json")
    seed = load(HERE / "susie_seed_retention_v1.json")
    seed_map = {c["text"]: s["score"] for c, s in zip(seed_pre["candidate_chains"], seed["candidate_chains"])}
    assert all(digest(c["text"]) == s["text_sha256"]
               for c, s in zip(seed_pre["candidate_chains"], seed["candidate_chains"]))
    anchors = load(HERE / "explicit_speaker_reserve_v1.json")["single_question_explicit_speakers"]
    tokenizer_path = ROOT / "model/qwen3.5-4b"
    tokenizer_files = [p for p in tokenizer_path.iterdir()
                       if p.is_file() and (p.suffix in {".json", ".jinja", ".model", ".txt"})]
    frozen_paths = [
        CONFIG, Path(__file__), HERE / "compare_chain_seed_preservation_v1.py",
        HERE / "chain_seed_preservation_candidate.py", HERE / "explicit_speaker_reserve_v1.py",
        HERE / "explicit_speaker_reserve_v1.json", HERE / "susie_reranker120_preflight_v1.json",
        HERE / "susie_reranker120_v1.json", HERE / "susie_chain_inputs_v1.json",
        HERE / "susie_chain_complete_pair4_v1.json", HERE / "susie_seed_retention_preflight_v1.json",
        HERE / "susie_seed_retention_v1.json", HERE / "fact_coverage_pairs116_v1.matrix.json",
    ] + tokenizer_files
    frozen = {str(p): sha(p) for p in frozen_paths}
    source = snapshot(SOURCE)
    assert source == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "artifacts_sha256": frozen, "runtime_source_sha256": source,
        "config": cfg, "question": query, "hypothesis": source_row["hypothesis"],
        "runtime_query": rerank_query, "runtime_safe_related_terms": related,
        "rescue_term_components": {
            "action_targets": extract_action_targets(resolved + "\n" + query),
            "content_tokens": extract_content_tokens(resolved),
        },
        "limits": [
            "Exact saved document/chain scores only; reject any unseen chain instead of fabricating logits.",
            "Run canonical final rerank/rescue32, prompt preparation10 and quote whole-block packing12000.",
            "After each canonical suffix, also test order-only reserve6 before unchanged prompt preparation.",
            "Current1024 arm replays earlier measured scores, not another truncated model execution.",
            "No new sparse/dense/fusion, alias resolution, full orchestration, generation or online service.",
            "Input source_row comes from a development historical state; not a newly generated hypothesis.",
        ],
    })
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    conditions = []
    for label, patched, scoremap in [
        ("original_builder_prefix1024", False, old_maps["current_prefix1024"]),
        ("original_builder_complete3072", False, old_maps["complete3072"]),
        ("seed_retained_complete3072", True, seed_map),
    ]:
        calls = []
        class Replay:
            def score(self, query, documents, batch_size):
                assert query == rerank_query
                calls.append({"query": query, "batch_size": batch_size, "input_sha256": [digest(t) for t in documents]})
                if len(calls) == 1:
                    assert documents == [c["document"]["search_text"] for c in candidates]
                    return doc_scores
                assert all(text in scoremap for text in documents), "unseen chain requires an actual new model score"
                return [scoremap[text] for text in documents]
        retriever = build_retriever(Replay(), candidates, candidate=patched)
        hits = fresh_hits(candidates)
        final = rerank_and_rescue_hits(
            retriever, question=query, resolved_question=resolved, rerank_query=rerank_query,
            hypothesis=hypothesis, fused_hits=hits, safe_related_terms=related, query_config=query_config)
        assert len(calls) == 2 and len(final) == 32
        for order_only in (False, True):
            ordered = reserve_ranked(
                final, anchors, top_k=12, reserve=6, id_of=lambda r: r["document"]["id"],
                text_of=lambda r: r["document"]["clean_text"]) if order_only else list(final)
            selected = prepare_prompt_evidence(
                pipeline=pipeline, question=query, hypothesis=hypothesis, evidence=deepcopy(ordered))
            packing = []
            prompt, brief = build_conclusion_prompt(
                question=query, current_hypothesis=hypothesis, evidence=ordered, retrieval_trace=[],
                current_round=source_row["current_round"], max_retrieval_rounds=source_row["max_retrieval_rounds"],
                prompt_evidence_top_k=inf["prompt_evidence_top_k"], prompt_evidence=selected,
                evidence_max_chars_per_doc=inf["prompt_evidence_max_chars_per_doc"],
                evidence_max_total_chars=inf["prompt_conclusion_evidence_max_total_chars"],
                prompt_mode=inf["conclusion_prompt_mode"], grounding_mode=inf["answer_grounding_mode"],
                evidence_packing_trace=packing)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            conditions.append({
                "scoring_condition": label, "order_only_reserve6": order_only,
                "actual_score_replay_calls": calls, "final32": ordered, "selected": selected,
                "selected_ids": [r["document"]["id"] for r in selected], "packing_trace": packing,
                "evidence_brief": brief, "runtime_prompt": prompt, "input_tokens": len(ids),
                "fits_configured_output_budget": len(ids) + cfg["generator"]["max_tokens"] <= cfg["generator"]["ctx_size"],
                "input_tokens_sha256": digest(json.dumps(ids)),
                "target_in_final32": any(r["document"]["id"] == TARGET for r in ordered),
                "target_selected_as_document": any(r["document"]["id"] == TARGET for r in selected),
                "target_phrase_in_selected_text": any("她依然没有放手" in best_prompt_text(r) for r in selected),
                "target_phrase_in_visible_prompt": "她依然没有放手" in brief,
                "new_generated_answer": None,
            })
    assert all(sha(Path(p)) == h for p, h in frozen.items()) and snapshot(SOURCE) == source
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "preflight_sha256": sha(PREFLIGHT),
        "conditions": conditions, "source_and_inputs_unchanged": True,
        "new_model_scores": 0, "new_generations": 0, "new_full_rag_recall": None,
        "new_certified_answer_accuracy": None,
    })
    print(json.dumps([{k: r[k] for k in [
        "scoring_condition", "order_only_reserve6", "selected_ids", "input_tokens",
        "fits_configured_output_budget", "target_in_final32", "target_selected_as_document",
        "target_phrase_in_visible_prompt",
    ]} for r in conditions], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
