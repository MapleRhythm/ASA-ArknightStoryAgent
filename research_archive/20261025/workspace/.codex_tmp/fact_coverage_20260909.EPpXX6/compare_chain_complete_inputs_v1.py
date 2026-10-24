"""Diagnose current chain-prefix scoring against complete-chain scoring on CPU.

The1024 arm faithfully reproduces the existing reranker, explicitly recorded
as truncated. The complete arm preserves every token; no live setting changes.
"""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

INPUTS = HERE / "susie_chain_inputs_v1.json"
PREFLIGHT = HERE / "susie_chain_complete_pair4_preflight_v1.json"
DESTINATION = HERE / "susie_chain_complete_pair4_v1.json"


def main():
    assert not PREFLIGHT.exists() and not DESTINATION.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(SOURCE))
    import torch
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.retrieval.reranker import CrossEncoderReranker
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    inputs = load(INPUTS)
    modelpre = load(HERE / "susie_reranker120_preflight_v1.json")
    modelpath = Path(modelpre["model_path"])
    assert inputs["chain_count"] == inputs["would_truncate_count"] == 2
    assert all(sha(Path(p)) == h for p, h in modelpre["model_assets_sha256"].items())
    assert snapshot(SOURCE) == inputs["runtime_source_hashes"]
    config = load(modelpath / "config.json")
    assert config["max_position_embeddings"] - config["pad_token_id"] - 1 >= 3072
    tokenizer = AutoTokenizer.from_pretrained(str(modelpath), trust_remote_code=False, local_files_only=True)
    rows = []
    for index, chain in enumerate(inputs["chains"]):
        full = tokenizer(inputs["query"], chain["text"], truncation=False)["input_ids"]
        capped = tokenizer(inputs["query"], chain["text"], truncation=True, max_length=1024)["input_ids"]
        assert digest(json.dumps(full)) == chain["full_input_ids_sha256"]
        assert digest(json.dumps(capped)) == chain["configured_input_ids_sha256"]
        assert len(full) <= 3072
        for arm in (["current_prefix1024", "complete3072"] if index == 0 else
                    ["complete3072", "current_prefix1024"]):
            ids = capped if arm == "current_prefix1024" else full
            rows.append({"index": index, "arm": arm, "query": inputs["query"], "text": chain["text"],
                         "ids": ids, "max_length": 1024 if arm == "current_prefix1024" else 3072,
                         "source_tokens": len(full), "actual_tokens": len(ids),
                         "input_truncated": ids != full})
    frozen = {str(p): sha(p) for p in (INPUTS, HERE / "susie_reranker120_preflight_v1.json", Path(__file__))}
    source_before = snapshot(SOURCE)
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "model_assets_sha256": modelpre["model_assets_sha256"],
        "runtime_source_hashes": source_before, "source_artifacts_sha256": frozen, "rows": rows,
        "limits": [
            "Two automatically assembled chains of one diagnosed question; not answer evidence or recall labels.",
            "Current1024 prefix arm explicitly reproduces known truncation, not a no-truncation claim.",
            "Complete3072 arm uses full chain; maximum local architecture capacity checked before running.",
            "Same model/eval mode/query/text; opposite per-chain order; forward token IDs verified.",
            "No chain edits, answer generation, training, default max_length or model changes.",
        ],
    })
    runner = CrossEncoderReranker(modelpath, device="cpu", max_length=1024)
    assert not runner.model.training and all(p.device.type == "cpu" for p in runner.model.parameters())
    real, actual = runner.model.forward, []
    def forward(*args, **kwargs):
        assert not args and len(actual) < len(rows)
        ids = kwargs["input_ids"][0][kwargs["attention_mask"][0].bool()].tolist()
        row = rows[len(actual)]
        assert ids == row["ids"]
        actual.append({"index": row["index"], "arm": row["arm"], "input_ids_sha256": digest(json.dumps(ids))})
        return real(**kwargs)
    runner.model.forward = forward
    results = []
    try:
        for row in rows:
            runner.max_length = row["max_length"]
            started = time.monotonic()
            score = runner.score(row["query"], [row["text"]], batch_size=1)[0]
            assert math.isfinite(score)
            result = {k: row[k] for k in ("index", "arm", "source_tokens", "actual_tokens", "input_truncated")}
            result.update(score=score, elapsed_seconds=time.monotonic() - started)
            results.append(result)
            print(json.dumps(result), flush=True)
    finally:
        runner.model.forward = real
    assert len(actual) == len(results) == 4
    assert snapshot(SOURCE) == source_before
    assert all(sha(Path(p)) == h for p, h in {**frozen, **modelpre["model_assets_sha256"]}.items())
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "preflight_sha256": sha(PREFLIGHT), "results": results, "actual_inputs": actual,
        "source_and_model_unchanged": True, "default_reranker_limit_changed": False,
        "new_answer_accuracy_or_recall": None,
    })


if __name__ == "__main__":
    main()
