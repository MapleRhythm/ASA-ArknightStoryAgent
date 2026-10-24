"""Fresh CPU sparse-only diagnostics, not full retrieval or recall evaluation."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot

INDEX = ROOT / "indexes/arknights_story"
DESTINATION = HERE / "canonical_sparse58_probe_v1.json"
PREFLIGHT = HERE / "canonical_sparse58_probe_preflight_v1.json"
MATRIX = HERE / "fact_coverage_pairs116_v1.matrix.json"


def main():
    assert not DESTINATION.exists() and not PREFLIGHT.exists()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(SOURCE))
    from rank_bm25 import BM25Okapi
    import numpy as np
    from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_base_search import HybridBaseSearchMixin
    from asa_arknight_story_agent.retrieval.hybrid_components.hybrid_utils import tokenize_for_bm25
    rows = [r for r in load(MATRIX)["rows"] if r["renderer"] == "original_prompt_v1"]
    assert len(rows) == len({r["question"] for r in rows}) == 58
    assert rows[56]["question"] == "夏栎最终是如何救出苏茜的？"
    assets = {str(p): sha(p) for p in [
        INDEX / "documents.jsonl", INDEX / "bm25_tokens.pkl", INDEX / "index_meta.json",
        MATRIX, Path(__file__),
    ]}
    source_before = snapshot(SOURCE)
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "source_artifacts_sha256": assets, "runtime_source_hashes": source_before,
        "queries": [{"index": i, "id": r["id"], "question": r["question"]} for i, r in enumerate(rows)],
        "top_k": 120, "target_index": 56, "joint_entity_terms": ["夏栎", "苏茜"],
        "protocol": [
            "Same58 questions, unchanged bare questions, current canonical BM25 corpus/implementation.",
            "No dense, reranker, MiniRAG, generated hypothesis, neighbor expansion or production orchestrator.",
            "Joint literal presence is a candidate locator, never an answer-bearing gold label.",
            "All58 sparse outputs and targeted full joint-match documents saved; no default retrieval change.",
            "New candidate evidence stays separate from all earlier frozen generation inputs.",
        ],
    })
    started = time.monotonic()
    with (INDEX / "documents.jsonl").open() as stream:
        documents = [json.loads(line) for line in stream if line.strip()]
    # Existing canonical pickle already consumed by the production loader;
    # hash is bound before loading. No downloaded or arbitrary pickle.
    with (INDEX / "bm25_tokens.pkl").open("rb") as stream:
        tokens = pickle.load(stream)
    assert len(documents) == len(tokens) and len({d["id"] for d in documents}) == len(documents)
    class SparseOnly(HybridBaseSearchMixin):
        pass
    retriever = SparseOnly()
    retriever.documents = documents
    retriever.bm25 = BM25Okapi(tokens)
    retriever.storyline_doc_indices = {}
    results = []
    joint = [i for i, d in enumerate(documents) if all(t in d.get("clean_text", "") for t in ("夏栎", "苏茜"))]
    target_ranks = {}
    for index, row in enumerate(rows):
        begin = time.monotonic()
        hits = retriever.sparse_search(row["question"], 120)
        if index == 56:
            scores = retriever.bm25.get_scores(tokenize_for_bm25(row["question"]))
            order = np.argsort(scores)[::-1].tolist()
            target_ranks = {di: rank for rank, di in enumerate(order, 1) if di in joint}
        saved = [{
            "rank": rank, "doc_index": hit["doc_index"], "score": hit["score"],
            "document_id": hit["document"]["id"], "activity_name": hit["document"].get("activity_name"),
            "story_name": hit["document"].get("story_name"),
            "joint_target_entities": hit["doc_index"] in joint,
        } for rank, hit in enumerate(hits, 1)]
        results.append({"index": index, "id": row["id"], "question": row["question"],
                        "tokens": tokenize_for_bm25(row["question"]), "hits": saved,
                        "elapsed_seconds": time.monotonic() - begin})
        print(json.dumps({"sparse_completed": index + 1, "total": 58, "hits": len(saved)}), flush=True)
    assert all(sha(Path(p)) == h for p, h in assets.items())
    assert snapshot(SOURCE) == source_before
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "preflight_sha256": sha(PREFLIGHT), "source_artifacts_sha256": assets,
        "documents": len(documents), "queries": 58, "elapsed_seconds": time.monotonic() - started,
        "results": results,
        "target_joint_entity_candidates": [{
            "doc_index": i, "bare_question_sparse_rank": target_ranks[i], "document": documents[i],
        } for i in joint],
        "target_joint_candidates": len(joint),
        "target_joint_sparse_top120": sum(rank <= 120 for rank in target_ranks.values()),
        "target_old_evidence_contains_susie": "苏茜" in rows[56]["evidence_brief"],
        "default_retriever_changed": False, "models_loaded": False,
        "new_retrieval_recall": None, "new_answer_accuracy": None,
        "limits": load(PREFLIGHT)["protocol"],
    })
    print(json.dumps({"documents": len(documents), "queries": 58, "joint_candidates": len(joint),
                      "joint_sparse_top120": sum(rank <= 120 for rank in target_ranks.values()),
                      "models_loaded": False, "recall": None}), flush=True)


if __name__ == "__main__":
    main()
