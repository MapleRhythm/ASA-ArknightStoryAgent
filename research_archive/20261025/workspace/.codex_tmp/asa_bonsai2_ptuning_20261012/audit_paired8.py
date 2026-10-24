"""Read-only integrity checks and descriptive aggregation; no semantic rewriting."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    root = Path(__file__).resolve().parent
    run = root / "paired8_4b_sftkto_vs_bonsai_pt_20261014_attempt02"
    manifest = json.loads((run / "manifest.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    assert sha(run / "results.jsonl") == completion["results_sha256"]
    assert sha(run / "bonsai_reference.jsonl") == manifest["bonsai_results_sha256"]
    requests = rows(run / "requests.jsonl")
    generated = rows(run / "results.jsonl")
    bonsai = rows(run / "bonsai_reference.jsonl")
    selection = json.loads((root / "heldout8_selection_20261014.json").read_text())["ids"]
    indexed = {r["id"]: r for r in rows(root / "glm_reviewed_full20_data_v1_20261014/val.jsonl")}
    review = json.loads((root / "paired8_review_20261014.json").read_text())
    assert len(generated) == len(requests) == completion["generations"] == 16
    assert len(bonsai) == 24 and len(review["reviews"]) == 32
    for arm in manifest["arms"]:
        assert [r["id"] for r in generated if r["arm"] == arm] == selection
    for receipt in requests:
        row = indexed[receipt["id"]]
        assert receipt["documents"] == row["documents"]
        assert receipt["round"] == row["round"]
        assert receipt["evidence_count"] == 20 and not receipt["input_truncated"]
        assert hashlib.sha256(receipt["prompt"].encode()).hexdigest() == receipt["prompt_sha256"]
        normalize = lambda x: " ".join(x.split())
        assert all(normalize(d["text"]) in normalize(receipt["prompt"]) for d in row["documents"])
        if receipt["arm"] == "4b_shared_exx_prompt":
            assert receipt["prompt"] == row["prompt"]
            assert all(b["prompt_sha256"] == receipt["prompt_sha256"]
                       for b in bonsai if b["id"] == receipt["id"])
        assert row["target"] not in receipt["prompt"]
    all_outputs = generated + [{**r, "arm": r["stage"]} for r in bonsai
                               if r["stage"] in {"frozen_no_prompt", "trained_prompt"}]
    assert len(all_outputs) == 32
    summary = {}
    for arm in sorted({r["arm"] for r in all_outputs}):
        outputs = [r for r in all_outputs if r["arm"] == arm]
        decisions = [r for r in review["reviews"] if r["arm"] == arm]
        assert len(outputs) == len(decisions) == 8
        assert {r["q"] for r in decisions} == set(range(1, 9))
        valid = []
        for output in outputs:
            content = output.get("protocol_prefill", "") + output["raw_output"]
            for t in ("<|im_end|>", "<|endoftext|>"):
                content = content.removesuffix(t)
            try:
                parsed = json.loads(content)
            except ValueError:
                parsed = None
            valid.append(isinstance(parsed, dict))
        summary[arm] = {
            "n": 8,
            "semantic_complete": sum(d["semantic_complete"] for d in decisions),
            "strict_deliverable": sum(d["strict_deliverable"] for d in decisions),
            "valid_json": sum(valid),
            "output_length_limits": sum(r["output_reached_length_limit"] for r in outputs),
            "mean_seconds_descriptive_not_engine_speed_comparison":
                statistics.mean(r["elapsed_seconds"] for r in outputs),
        }
    print(json.dumps({
        "integrity_passed": True,
        "shared_prompt_exact_matches": 8,
        "full20_evidence_verified_requests": 16,
        "results_sha256": completion["results_sha256"],
        "review_sha256": sha(root / "paired8_review_20261014.json"),
        "semantic_review_is_assistant_not_human_gold": True,
        "summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
