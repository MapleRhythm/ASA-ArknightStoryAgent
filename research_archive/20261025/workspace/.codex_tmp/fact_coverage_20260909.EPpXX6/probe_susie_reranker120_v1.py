"""One question, untouched sparse120 candidates, current cross-encoder on CPU.

This isolates the model's document scores; it is not the production hybrid
retriever, its feature-adjusted/chain scores, or an E2E accuracy measurement.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

CONFIG = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
PROBE = HERE / "canonical_sparse58_probe_v1.json"
PREFLIGHT = HERE / "susie_reranker120_preflight_v1.json"
DESTINATION = HERE / "susie_reranker120_v1.json"
STORY = HERE / "susie_rescue_canonical_story_v1.json"


def main():
    assert not any(p.exists() for p in (PREFLIGHT, DESTINATION, STORY))
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    import torch
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    probe = load(PROBE)
    assert probe["queries"] == 58 and probe["cpu_only"]
    assert all(sha(Path(p)) == h for p, h in probe["source_artifacts_sha256"].items())
    query = probe["results"][56]["question"]
    assert query == "夏栎最终是如何救出苏茜的？"
    hits = probe["results"][56]["hits"]
    assert len(hits) == len({h["doc_index"] for h in hits}) == 120
    with (ROOT / "indexes/arknights_story/documents.jsonl").open() as stream:
        documents = [json.loads(line) for line in stream if line.strip()]
    assert len(documents) == probe["documents"]
    candidates = [documents[h["doc_index"]] for h in hits]
    assert all(d["id"] == h["document_id"] for d, h in zip(candidates, hits))
    # Only pick the configured path/length, never serialize secrets or the
    # rest of the configuration.
    configuration = load(CONFIG)
    matches = [v for v in configuration.values() if isinstance(v, dict) and "reranker_model_path" in v]
    assert len(matches) == 1
    cfg = matches[0]
    model = ROOT / "release/ASA-ArknightStoryAgent" / cfg["reranker_model_path"]
    model = model.resolve()
    assert cfg["enable_reranker"] and cfg["reranker_max_length"] == 1024
    assert model.is_dir() and not load(model / "config.json").get("auto_map")
    assert not load(model / "tokenizer_config.json").get("auto_map")
    assets = {str(p): sha(p) for p in sorted(model.iterdir()) if p.is_file()}
    source_before = snapshot(SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False, local_files_only=True)
    texts = [d["search_text"] for d in candidates]
    inputs = tokenizer([query] * 120, texts, truncation=False, padding=False)
    ids = inputs["input_ids"]
    assert max(map(len, ids)) <= 1024, "current model cap would truncate; do not run or drop a candidate"
    story = sorted([d for d in documents if d.get("story_id") == "activities/act10mini/level_act10mini_st05"],
                   key=lambda d: (d.get("chunk_index", -1), d["id"]))
    save(STORY, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "source_documents_sha256": sha(ROOT / "indexes/arknights_story/documents.jsonl"),
        "story_id": "activities/act10mini/level_act10mini_st05",
        "documents": story,
        "selection": "Full indexed story by exact story_id; all chunks/summary variants retained.",
        "sent_to_generator_or_reranker": False,
    })
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "query": query, "model_path": str(model), "max_length": 1024, "batch_size": 1,
        "threads": 2, "document_field": "search_text",
        "model_assets_sha256": assets, "runtime_source_hashes": source_before,
        "source_artifacts_sha256": {str(p): sha(p) for p in (CONFIG, PROBE, Path(__file__), STORY)},
        "candidates": [{"sparse_rank": h["rank"], "doc_index": h["doc_index"], "document": d,
                        "input_tokens": len(tok), "input_ids_sha256": digest(json.dumps(tok))}
                       for h, d, tok in zip(hits, candidates, ids)],
        "limits": [
            "Sparse120 candidates fixed before scoring; no positives supplied to the scorer.",
            "Current cross-encoder alone on search_text, no dense/fusion/neighbor/chain/feature adjustment.",
            "CPU batch1, eval mode; all full inputs must fit1024 and actual forward token IDs are checked.",
            "No generator or default setting changes; not a fresh production RAG recall/answer rate.",
        ],
    })
    begin = time.monotonic()
    reranker = CrossEncoderReranker(model, device="cpu", max_length=1024)
    assert all(p.device.type == "cpu" for p in reranker.model.parameters()) and not reranker.model.training
    actual, scores = [], []
    real_forward = reranker.model.forward
    def forward(*args, **kwargs):
        assert not args and len(actual) < 120
        actual_ids = kwargs["input_ids"][0][kwargs["attention_mask"][0].bool()].tolist()
        assert actual_ids == ids[len(actual)], "actual token sequence differs from full untruncated input"
        actual.append({"candidate_index": len(actual), "actual_token_ids_sha256": digest(json.dumps(actual_ids))})
        return real_forward(**kwargs)
    reranker.model.forward = forward
    try:
        for offset, text in enumerate(texts):
            values = reranker.score(query, [text], batch_size=1)
            assert len(values) == 1
            scores.extend(values)
            print(json.dumps({"reranked": offset + 1, "total": 120,
                              "elapsed": round(time.monotonic() - begin, 2)}), flush=True)
    finally:
        reranker.model.forward = real_forward
    assert len(scores) == len(actual) == 120
    assert all(sha(Path(p)) == h for p, h in assets.items())
    assert snapshot(SOURCE) == source_before
    import math
    assert all(math.isfinite(s) for s in scores)
    ordered = sorted(range(120), key=lambda i: (-scores[i], i))
    ranked = [{"rerank_position": rank, "sparse_rank": hits[i]["rank"], "doc_index": hits[i]["doc_index"],
               "document_id": candidates[i]["id"], "score": scores[i],
               "story_name": candidates[i].get("story_name")}
              for rank, i in enumerate(ordered, 1)]
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "preflight_sha256": sha(PREFLIGHT), "query": query, "ranked": ranked,
        "actual_inputs": actual, "scores_in_sparse_order": scores, "elapsed_seconds": time.monotonic() - begin,
        "all120_inputs_untruncated": True, "assets_and_source_unchanged": True,
        "default_reranker_changed": False, "full_rag_recall": None, "answer_accuracy": None,
        "limits": load(PREFLIGHT)["limits"],
    })
    print(json.dumps({"top12": ranked[:12], "all_inputs_untruncated": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
