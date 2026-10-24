"""Bind semantic notes to completed final text and the actual frozen evidence."""
import json
from pathlib import Path
from protocol_helpers import digest, sha

HERE = Path(__file__).resolve().parent


def main():
    results = [json.loads(x) for x in (HERE / "attempt1/results.jsonl").read_text().splitlines()]
    review = json.loads((HERE / "semantic_review.json").read_text())
    inputs = json.loads((HERE / "inputs.json").read_text())
    lookup = {(x["question_id"], x["condition"]): x for x in results}
    source = {x["question_id"]: x for x in inputs["rows"]}
    assert len(lookup) == len(review["rows"]) == 40 and review["complete"]
    rows = []
    for item in review["rows"]:
        result = lookup[(item["question_id"], item["condition"])]
        documents = source[item["question_id"]]["documents"]
        if item.get("raw_final_answer"):
            payload = json.loads(result["final_text"])
            # Read a completed final_answer solely for content audit. Do not fix runtime outputs.
            assert payload["answer_directly"]["final_answer"] == item["raw_final_answer"]
            assert result["status"] == "protocol_error" and result["finish_reason"] == "stop"
        evidence_bindings = []
        for ref in item["reference_evidence"]:
            assert ref.startswith("E")
            idx = int(ref[1:]) - 1
            assert 0 <= idx < 20
            doc = documents[idx]
            evidence_bindings.append({
                "ref": ref, "id": doc["id"], "clean_text_sha256": digest(doc["clean_text"]),
            })
        original_source = item.get("reference_source")
        original_source_binding = None
        if original_source:
            name, line = original_source.rsplit(":", 1)
            path = (HERE / name).resolve()
            assert path.exists() and len(path.read_text().splitlines()) >= int(line)
            original_source_binding = {"path": str(path), "line": int(line), "sha256": sha(path)}
        rows.append({
            "question_id": item["question_id"], "condition": item["condition"],
            "final_text_sha256": digest(result.get("final_text", "")),
            "raw_answer_sha256": digest(result.get("normalized", {}).get(
                "answer", item.get("raw_final_answer", ""))),
            "delivered_answer_sha256": digest(result.get("grounded", {}).get("answer", "")),
            "raw_grade": item["raw_grade"], "delivered_grade": item["delivered_grade"],
            "evidence_bindings": evidence_bindings,
            "reference_source_binding": original_source_binding,
        })
    report = {
        "bound_reviews": len(rows), "results_sha256": sha(HERE / "attempt1/results.jsonl"),
        "review_sha256": sha(HERE / "semantic_review.json"), "rows": rows,
        "warning": "Hashes bind review to source/output; they do not independently prove semantic labels.",
    }
    with (HERE / "review_bindings.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"bound_reviews": len(rows), "hash": sha(HERE / "review_bindings.json")}))


if __name__ == "__main__":
    main()
