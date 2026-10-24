"""CPU same120 comparison of an existing reranker; never replaces defaults."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

MODEL = ROOT / "model/reranker/bge-reranker-v2-m3-rank-v7-clean-online-stage2"
PREFLIGHT = HERE / "susie_v7stage2_same120_preflight_v1.json"
DESTINATION = HERE / "susie_v7stage2_same120_v1.json"
CURRENT = HERE / "susie_reranker120_v1.json"
CURRENT_PRE = HERE / "susie_reranker120_preflight_v1.json"


def data_lineage():
    manifests, files, overlap = {}, {}, {}
    path = MODEL
    query = load(CURRENT_PRE)["query"]
    for _ in range(8):
        manifest_path = path / "evidence_chain_reranker_manifest.json"
        if not manifest_path.is_file():
            break
        manifests[str(manifest_path)] = {"sha256": sha(manifest_path), "manifest": load(manifest_path)}
        manifest = load(manifest_path)
        for split in ("train_file", "eval_file"):
            if not manifest.get(split):
                continue
            dataset = Path(manifest[split])
            assert dataset.is_file()
            files[str(dataset)] = sha(dataset)
            matching = []
            with dataset.open() as stream:
                count = 0
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    count += 1
                    if row.get("query") == query:
                        matching.append({"line": line_number,
                                         "record_sha256": digest(json.dumps(row, ensure_ascii=False, sort_keys=True)),
                                         "query_type": row.get("query_type"), "negative_type": row.get("negative_type")})
            overlap[str(dataset)] = {"split": split, "records": count, "exact_question_matches": matching}
        parent = Path(manifest["base_model"])
        assert parent != path
        path = parent
    return {"manifests": manifests, "dataset_sha256": files, "exact_query_overlap": overlap,
            "limits": "Saved manifests/data only; not certification of historical training CLI/cache bytes."}


def main():
    assert not PREFLIGHT.exists() and not DESTINATION.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    import torch
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    control, control_pre = load(CURRENT), load(CURRENT_PRE)
    assert control["preflight_sha256"] == sha(CURRENT_PRE) and control["all120_inputs_untruncated"]
    assert all(sha(Path(p)) == h for p, h in control_pre["model_assets_sha256"].items())
    assert not load(MODEL / "config.json").get("auto_map")
    assert not load(MODEL / "tokenizer_config.json").get("auto_map")
    assets = {str(p): sha(p) for p in sorted(MODEL.iterdir()) if p.is_file()}
    lineage = data_lineage()
    source_before = snapshot(SOURCE)
    candidates = control_pre["candidates"]
    assert len(candidates) == 120
    query = control_pre["query"]
    texts = [r["document"]["search_text"] for r in candidates]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=False, local_files_only=True)
    ids = tokenizer([query] * 120, texts, padding=False, truncation=False)["input_ids"]
    assert max(map(len, ids)) <= 1024
    assert all(digest(json.dumps(tok)) == row["input_ids_sha256"] for tok, row in zip(ids, candidates))
    frozen = {str(p): sha(p) for p in (CURRENT, CURRENT_PRE, Path(__file__),
                                     HERE / "chain_score_binding_applied_verified_v1.json")}
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "model_path": str(MODEL), "model_assets_sha256": assets, "lineage": lineage,
        "source_artifacts_sha256": frozen, "runtime_source_hashes": source_before,
        "query": query, "candidates": candidates, "max_length": 1024, "batch_size": 1,
        "same_actual_input_tokens_as_current": True,
        "limits": [
            "One diagnosed development question, not holdout and not full RAG retrieval.",
            "Frozen sparse120 and query, search_text, actual token IDs and1024 cap unchanged.",
            "Only existing candidate reranker weights/assets change; no chain scorer or feature adjustment.",
            "All candidate outputs retained; no positive document used in scoring; no training or default change.",
            "Saved-manifest overlap is inspected; pretrained/earlier undeclared exposure is not excluded.",
        ],
    })
    model = CrossEncoderReranker(MODEL, device="cpu", max_length=1024)
    assert all(p.device.type == "cpu" for p in model.model.parameters()) and not model.model.training
    real, actual = model.model.forward, []
    def forward(*args, **kwargs):
        assert not args and len(actual) < 120
        values = kwargs["input_ids"][0][kwargs["attention_mask"][0].bool()].tolist()
        assert values == ids[len(actual)]
        actual.append(digest(json.dumps(values)))
        return real(**kwargs)
    model.model.forward = forward
    scores, start = [], time.monotonic()
    try:
        for index, text in enumerate(texts):
            values = model.score(query, [text], batch_size=1)
            assert len(values) == 1 and math.isfinite(values[0])
            scores.extend(values)
            if (index + 1) % 10 == 0:
                print(json.dumps({"candidate_scored": index + 1, "total": 120,
                                  "elapsed": round(time.monotonic() - start, 2)}), flush=True)
    finally:
        model.model.forward = real
    assert len(scores) == len(actual) == 120
    assert all(sha(Path(p)) == h for p, h in {**assets, **frozen, **lineage["dataset_sha256"]}.items())
    assert snapshot(SOURCE) == source_before
    old_rank = {r["document_id"]: r["rerank_position"] for r in control["ranked"]}
    order = sorted(range(120), key=lambda i: (-scores[i], i))
    ranked = [{
        "position": rank, "sparse_rank": candidates[i]["sparse_rank"],
        "current_model_position": old_rank[candidates[i]["document"]["id"]],
        "document_id": candidates[i]["document"]["id"], "score": scores[i],
        "story_name": candidates[i]["document"].get("story_name"),
    } for rank, i in enumerate(order, 1)]
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "preflight_sha256": sha(PREFLIGHT), "ranked": ranked, "scores_in_sparse_order": scores,
        "actual_token_ids_sha256": actual, "all120_inputs_untruncated": True,
        "same_input_tokens_as_control": True, "assets_and_source_unchanged": True,
        "elapsed_seconds": time.monotonic() - start, "default_model_changed": False,
        "new_recall_or_answer_accuracy": None,
    })
    print(json.dumps({"top12": ranked[:12], "default_model_changed": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
