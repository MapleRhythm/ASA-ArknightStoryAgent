"""Bind assistant observations to exact visible blocks without relabeling data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    assert not (HERE / "checkpoint_v40.json").exists()
    pre, results, post, review = [
        load(HERE / name) for name in
        ("preflight.json", "retrieval.json", "postflight_v1.json", "source_review_v40.json")
    ]
    assert sha(HERE / "preflight.json") == results["preflight_sha256"] == post["preflight_sha256"]
    assert sha(HERE / "retrieval.json") == post["retrieval_sha256"]
    assert post["status_counts"] == {"ok": 12}
    assert results["llm_generation_calls"] == 0
    indexed = {(row["id"], row["condition"]): row for row in results["conditions"]}
    assert len(indexed) == 12
    assert {case["id"] for case in review["cases"]} == {r["id"] for r in pre["rows"]}
    bindings = []
    for case in review["cases"]:
        for binding in case["bindings"]:
            for kind in binding["conditions"]:
                record = indexed[(case["id"], kind)]
                brief = record["evidence_brief"]
                parts = re.split(r"(?m)^\[(E\d+)\]\s*", brief)
                blocks = dict(zip(parts[1::2], parts[2::2], strict=True))
                assert binding["quote"] in blocks[binding["eid"]], binding
                bound = {
                    "id": case["id"], "condition": kind, **binding,
                    "visible_sha256": digest(brief),
                    "block_sha256": digest(blocks[binding["eid"]]),
                }
                if "document_id" in binding:
                    matches = [
                        obj for obj in record["prompt_evidence"]
                        if obj["document"]["id"] == binding["document_id"]
                    ]
                    assert len(matches) == 1
                    assert binding["quote"] in json.dumps(matches[0]["document"], ensure_ascii=False)
                    bound["document_sha256"] = digest(json.dumps(matches[0]["document"], ensure_ascii=False, sort_keys=True))
                bindings.append(bound)
            for kind in binding.get("absent_from_visible", []):
                assert binding["quote"] not in indexed[(case["id"], kind)]["evidence_brief"]
    query_counts = {}
    for row in pre["rows"]:
        pair = [indexed[(row["id"], kind)] for kind in ("control", "candidate")]
        query_counts[row["id"]] = [len(item["actual_queries"]) for item in pair]
        if row["no_intervention_control"]:
            assert pair[0]["actual_hypothesis"] == pair[1]["actual_hypothesis"]
            assert pair[0]["evidence"] == pair[1]["evidence"]
            assert pair[0]["evidence_brief"] == pair[1]["evidence_brief"]
            assert pair[0]["preview_prompt"] == pair[1]["preview_prompt"]
    assert query_counts == {
        "fixed58-007": [16, 16], "fixed58-010": [16, 16], "fixed58-017": [13, 13],
        "fixed58-023": [16, 16], "fixed58-046": [16, 16], "fixed58-049": [15, 16],
    }
    assert indexed[("fixed58-049", "control")]["evidence_brief"] == indexed[("fixed58-049", "candidate")]["evidence_brief"]
    canonical = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
    if canonical.exists():
        assert all(sha(Path(p)) == h for p, h in pre["sha256"].items())
    else:
        raise RuntimeError("Close only on canonical 241 after source/index verification")
    files = [HERE / name for name in (
        "preflight.json", "retrieval.json", "postflight_v1.json", "run_v1.log",
        "idle_samples.json", "retrieval.progress.jsonl", "source_review_v40.json",
        "run_retrieval_probe.py", "close_retrieval_probe.py",
    )]
    result = {
        "version": "routing_frozen_first_retrieval_closed_v40",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False,
        "status_counts_not_accuracy": post["status_counts"],
        "llm_generation_calls": 0,
        "assistant_visible_source_review_complete": True,
        "reviewer": review["reviewer"], "source_bindings": bindings,
        "query_counts_control_candidate": query_counts,
        "comparisons_not_accuracy": post["comparisons_not_accuracy"],
        "promotion_decision": review["promotion_decision"],
        "basis_sha256": {str(p): sha(p) for p in files},
        "limitations": review["limitations"],
    }
    with (HERE / "checkpoint_v40.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print({"conditions": 12, "bindings": len(bindings), "generation_calls": 0,
           "production_changed": False, "decision": review["promotion_decision"]})


if __name__ == "__main__":
    main()
