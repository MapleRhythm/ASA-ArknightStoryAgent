"""Capture real chain-score inputs on the fixed sparse120, without scoring.

Reuse current-model document scores; interrupt at chain-model dispatch.
Never manufacture chain logits or claim the captured candidates are answers.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

DESTINATION = HERE / "susie_chain_inputs_v1.json"


class CapturedChainInputs(Exception):
    pass


def main():
    assert not DESTINATION.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.retrieval.finalization import build_final_rerank_query
    from asa_arknight_story_agent.inference.retrieval.reranking import classify_retrieval_query_mode
    prepath, resultpath = HERE / "susie_reranker120_preflight_v1.json", HERE / "susie_reranker120_v1.json"
    pre, result = load(prepath), load(resultpath)
    assert result["preflight_sha256"] == sha(prepath) and result["all120_inputs_untruncated"]
    assert all(sha(Path(p)) == h for p, h in pre["model_assets_sha256"].items())
    candidates = pre["candidates"]
    query = pre["query"]
    matrixpath = HERE / "fact_coverage_pairs116_v1.matrix.json"
    row = next(r for r in load(matrixpath)["rows"]
               if r["question"] == query and r["renderer"] == "original_prompt_v1")
    hypothesis = HypothesisDocument(**row["hypothesis"])
    resolved, runtime_query, related = build_final_rerank_query(query, hypothesis)
    mode = classify_retrieval_query_mode(hypothesis)
    captured = []
    class CaptureReranker:
        def score(self, query, documents, batch_size):
            if not captured:
                assert documents == [r["document"]["search_text"] for r in candidates]
                captured.append({"kind": "reused_document_scores", "query": query,
                                 "scores": result["scores_in_sparse_order"]})
                return result["scores_in_sparse_order"]
            captured.append({"kind": "unscored_chain_inputs", "query": query, "documents": documents,
                             "batch_size": batch_size})
            raise CapturedChainInputs()
    docs = [r["document"] for r in candidates]
    retriever = ArknightsHybridRetriever(
        documents=docs, index=None, bm25=None, embedding_model=None,
        reranker=CaptureReranker(), minirag_index=None)
    hits = [{"doc_index": r["doc_index"], "document": deepcopy(r["document"]), "fusion_score": 0.0}
            for r in candidates]
    source_before = snapshot(SOURCE)
    try:
        retriever.rerank_with_evidence_chains(query, hits, top_k=120, batch_size=1, query_mode=mode)
    except CapturedChainInputs:
        pass
    else:
        raise AssertionError("expected a real chain-model dispatch")
    assert len(captured) == 2 and captured[1]["kind"] == "unscored_chain_inputs"
    tokenizer = AutoTokenizer.from_pretrained(pre["model_path"], trust_remote_code=False, local_files_only=True)
    texts = captured[1]["documents"]
    full = tokenizer([query] * len(texts), texts, truncation=False, padding=False)["input_ids"]
    capped = tokenizer([query] * len(texts), texts, truncation=True, padding=False, max_length=1024)["input_ids"]
    payloads = [{
        "index": i, "text": text, "text_sha256": digest(text),
        "full_input_tokens": len(ids), "configured_input_tokens": len(cap),
        "would_truncate_at_current1024": ids != cap,
        "full_input_ids_sha256": digest(json.dumps(ids)),
        "configured_input_ids_sha256": digest(json.dumps(cap)),
        "source_target_name_present": "苏茜" in text,
        "critical_rescue_phrase_present": "她依然没有放手" in text,
    } for i, (text, ids, cap) in enumerate(zip(texts, full, capped))]
    assert snapshot(SOURCE) == source_before
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "model_loaded": False, "new_model_scores": 0, "query": query, "query_mode": mode,
        "runtime_final_query_for_comparison_only": runtime_query,
        "runtime_related_terms_for_comparison_only": related,
        "runtime_final_query_equals_probe_query": runtime_query == query,
        "chains": payloads, "chain_count": len(payloads),
        "would_truncate_count": sum(r["would_truncate_at_current1024"] for r in payloads),
        "source_artifacts_sha256": {str(p): sha(p) for p in (prepath, resultpath, matrixpath, Path(__file__))},
        "runtime_source_hashes": source_before,
        "limits": [
            "Current canonical chain builder on fixed sparse120 and actual saved document logits.",
            "Bare question used consistently for all scores; runtime final-query expansion merely recorded.",
            "Capture stops before model scoring; configured1024 truncation is inspected, not used to generate an answer.",
            "Chains may have different events or wrong role labels; not certified answer evidence.",
            "Neither this capture nor the direct-score120 probe is the production full retrieval pipeline.",
        ],
    })
    print(json.dumps({"chains": len(payloads),
                      "would_truncate": sum(r["would_truncate_at_current1024"] for r in payloads),
                      "max_full_tokens": max(r["full_input_tokens"] for r in payloads),
                      "runtime_final_query": runtime_query}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
