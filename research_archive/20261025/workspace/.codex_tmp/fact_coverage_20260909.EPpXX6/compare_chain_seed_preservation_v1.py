"""Fixed sparse120/current logits, original vs seed-retaining full-chain scoring.

Research-only candidate; no release edits, training or generated answers.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot, counts

PREFLIGHT = HERE / "susie_seed_retention_preflight_v1.json"
DESTINATION = HERE / "susie_seed_retention_v1.json"
JOURNAL = HERE / "susie_seed_retention_scores_v1.progress.jsonl"
PATCH = HERE / "chain_seed_preservation_candidate.py"


class Captured(Exception):
    pass


def build_retriever(reranker, candidates, candidate):
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever
    result = ArknightsHybridRetriever(
        documents=[r["document"] for r in candidates], index=None, bm25=None,
        embedding_model=None, reranker=reranker, minirag_index=None)
    if candidate:
        spec = importlib.util.spec_from_file_location("asa_seed_research_candidate", PATCH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        method = module.HybridEvidenceChainBuildingMixin._build_evidence_chains
        result._build_evidence_chains = types.MethodType(method, result)
    return result


def fresh_hits(candidates):
    return [{"doc_index": r["doc_index"], "document": deepcopy(r["document"]), "fusion_score": 0.0}
            for r in candidates]


def main():
    assert not any(p.exists() for p in (PREFLIGHT, DESTINATION, JOURNAL))
    assert counts(HERE / "chain_seed_preservation_tests_v1.xml") == {
        "tests": 9, "failures": 0, "errors": 0, "skipped": 0}
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    import torch
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    control_pre = load(HERE / "susie_reranker120_preflight_v1.json")
    control = load(HERE / "susie_reranker120_v1.json")
    old_chains = load(HERE / "susie_chain_inputs_v1.json")
    old_pair = load(HERE / "susie_chain_complete_pair4_v1.json")
    query, candidates = control_pre["query"], control_pre["candidates"]
    mode = old_chains["query_mode"]
    assert old_chains["runtime_final_query_equals_probe_query"] and query == old_chains["query"]
    old_scores = control["scores_in_sparse_order"]
    captured = []
    class Capture:
        def score(self, query, documents, batch_size):
            if not captured:
                assert documents == [c["document"]["search_text"] for c in candidates]
                captured.append({"kind": "document", "query": query})
                return old_scores
            captured.append({"kind": "chains", "query": query, "texts": documents})
            raise Captured()
    probe = build_retriever(Capture(), candidates, candidate=True)
    try:
        probe.rerank_with_evidence_chains(query, fresh_hits(candidates), top_k=120, batch_size=1, query_mode=mode)
    except Captured:
        pass
    else:
        raise AssertionError("candidate produced no model chain dispatch")
    assert len(captured) == 2 and captured[1]["query"] == query
    texts = captured[1]["texts"]
    assert len(texts) == len(set(texts)) and 0 < len(texts) <= 80
    modelpath = Path(control_pre["model_path"])
    assert all(sha(Path(p)) == h for p, h in control_pre["model_assets_sha256"].items())
    tokenizer = AutoTokenizer.from_pretrained(str(modelpath), trust_remote_code=False, local_files_only=True)
    ids = tokenizer([query] * len(texts), texts, truncation=False, padding=False)["input_ids"]
    assert max(map(len, ids)) <= 3072, "do not truncate/drop a candidate chain to fit"
    assert load(modelpath / "config.json")["max_position_embeddings"] >= 3074
    old_full_by_text = {}
    old_prefix_by_text = {}
    for chain in old_chains["chains"]:
        full = next(r["score"] for r in old_pair["results"]
                    if r["index"] == chain["index"] and r["arm"] == "complete3072")
        prefix = next(r["score"] for r in old_pair["results"]
                      if r["index"] == chain["index"] and r["arm"] == "current_prefix1024")
        old_full_by_text[chain["text"]] = full
        old_prefix_by_text[chain["text"]] = prefix
    fresh = [(text, tok) for text, tok in zip(texts, ids) if text not in old_full_by_text]
    source_before = snapshot(SOURCE)
    frozen = {str(p): sha(p) for p in [
        HERE / "susie_reranker120_preflight_v1.json", HERE / "susie_reranker120_v1.json",
        HERE / "susie_chain_inputs_v1.json", HERE / "susie_chain_complete_pair4_v1.json",
        HERE / "chain_seed_preservation_tests_v1.xml", PATCH, Path(__file__),
    ]}
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "query": query, "query_mode": mode, "source_artifacts_sha256": frozen,
        "runtime_source_hashes": source_before, "model_assets_sha256": control_pre["model_assets_sha256"],
        "candidate_chains": [{"text": text, "tokens": len(tok), "token_ids_sha256": digest(json.dumps(tok)),
                              "reuses_exact_prior_full_score": text in old_full_by_text}
                             for text, tok in zip(texts, ids)],
        "new_chain_scores": len(fresh), "complete_max_length": 3072,
        "limits": [
            "Only process-local chain builder adds seed-first sort key; role/seed/member caps retained.",
            "Same automatically retrieved120 docs and current actual document logits; no gold/reference supplied.",
            "Every new chain is scored without truncation at3072, not just selected good chains.",
            "Only byte-identical prior full-chain inputs are reused, never prefix score as full score.",
            "One diagnosed development question; no recall/answer accuracy or default promotion.",
        ],
    })
    scored = dict(old_full_by_text)
    actual = []
    runner = CrossEncoderReranker(modelpath, device="cpu", max_length=3072) if fresh else None
    if runner:
        assert all(p.device.type == "cpu" for p in runner.model.parameters()) and not runner.model.training
        real = runner.model.forward
        def forward(*args, **kwargs):
            assert not args
            actual_ids = kwargs["input_ids"][0][kwargs["attention_mask"][0].bool()].tolist()
            assert actual_ids == fresh[len(actual)][1]
            actual.append(digest(json.dumps(actual_ids)))
            return real(**kwargs)
        runner.model.forward = forward
    try:
        with JOURNAL.open("x", encoding="utf-8") as journal:
            for index, (text, tok) in enumerate(fresh):
                begin = time.monotonic()
                score = runner.score(query, [text], batch_size=1)[0]
                assert math.isfinite(score)
                scored[text] = score
                record = {"text_sha256": digest(text), "input_ids_sha256": digest(json.dumps(tok)),
                          "full_input_tokens": len(tok), "score": score,
                          "elapsed_seconds": time.monotonic() - begin, "input_truncated": False}
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"new_chain_scored": index + 1, "total_new": len(fresh),
                                  "score": score, "seconds": record["elapsed_seconds"]}), flush=True)
    finally:
        if runner:
            runner.model.forward = real
    assert len(actual) == len(fresh)
    conditions = []
    for name, patched, chain_scores in [
        ("original_builder_prefix1024", False, old_prefix_by_text),
        ("original_builder_complete3072", False, old_full_by_text),
        ("seed_retained_complete3072", True, scored),
    ]:
        calls = []
        class Replay:
            def score(self, query, documents, batch_size):
                calls.append(documents)
                if len(calls) == 1:
                    assert documents == [c["document"]["search_text"] for c in candidates]
                    return old_scores
                assert set(documents) == (set(texts) if patched else set(old_full_by_text))
                return [chain_scores[text] for text in documents]
        retriever = build_retriever(Replay(), candidates, candidate=patched)
        ranked = retriever.rerank_with_evidence_chains(
            query, fresh_hits(candidates), top_k=120, batch_size=1, query_mode=mode)
        assert len(calls) == 2 and len(ranked) == 120
        conditions.append({
            "condition": name,
            "ranked": [{"rank": rank, "document_id": item["document"]["id"],
                        "document": item["document"], "final_score": retriever._final_rerank_score(item),
                        **{k: item[k] for k in ("rerank_score", "evidence_chain_score",
                                               "evidence_chain_model_score", "evidence_chain_roles",
                                               "evidence_chain_text") if k in item}}
                       for rank, item in enumerate(ranked, 1)],
        })
    assert snapshot(SOURCE) == source_before
    assert all(sha(Path(p)) == h for p, h in {**frozen, **control_pre["model_assets_sha256"]}.items())
    target = "activities/act10mini/level_act10mini_st05#chunk-0025"
    target_ranks = {c["condition"]: next(r["rank"] for r in c["ranked"] if r["document_id"] == target)
                    for c in conditions}
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "preflight_sha256": sha(PREFLIGHT), "new_chain_scores": len(fresh),
        "candidate_chain_count": len(texts),
        "candidate_chains": [{"text_sha256": digest(text), "score": scored[text]} for text in texts],
        "actual_token_ids_sha256": actual, "conditions": conditions,
        "selected_rescue_segment_ranks": target_ranks, "source_and_model_unchanged": True,
        "default_chain_builder_changed": False, "new_recall_or_answer_accuracy": None,
    })
    print(json.dumps({"candidate_chains": len(texts), "new_scored": len(fresh),
                      "selected_rescue_segment_ranks": target_ranks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
