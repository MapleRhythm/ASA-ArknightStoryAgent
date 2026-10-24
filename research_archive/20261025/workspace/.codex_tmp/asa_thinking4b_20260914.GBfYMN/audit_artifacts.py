"""Local cross-file audit; does not rerun inference or change frozen inputs."""
import json
from pathlib import Path
from protocol_helpers import sha, digest, answer_after_thinking

HERE = Path(__file__).resolve().parent


def main():
    pre = json.loads((HERE / "preflight.json").read_text())
    out = HERE / "attempt1"
    rows = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    completion = json.loads((out / "completion.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(rows) == completion["generations"] == 40
    assert completion["results_sha256"] == sha(out / "results.jsonl")
    assert completion["preflight_sha256"] == manifest["preflight_sha256"] == sha(HERE / "preflight.json")
    expected_sequence = [(r["question_id"], c) for r in pre["rows"] for c in r["order"]]
    assert [(r["question_id"], r["condition"]) for r in rows] == expected_sequence
    lookup = {r["question_id"]: r for r in pre["rows"]}
    checks = 0
    for index, row in enumerate(rows, 1):
        source = lookup[row["question_id"]]
        kind = "on" if row["condition"].endswith("_on") else "off"
        prompt = source["prompts"][kind]
        assert row["request_index"] == index
        assert row["question"] == source["question"]
        assert row["seed"] == source["seed"]
        assert row["prompt_sha256"] == prompt["sha256"] == digest(prompt["text"])
        assert row["input_tokens"] == prompt["tokens"]
        assert row["input_ids_sha256"] == prompt["token_ids_sha256"]
        assert row["input_token_ids_exact_match"] is True
        assert row["requested_lora_id"] == row["returned_lora_id"] == (
            1 if row["condition"].startswith("lora_") else None)
        assert row["max_output_tokens"] == 32768
        assert row["input_tokens"] + 32768 <= 40960
        assert row["output_tokens"] <= 32768
        assert len(prompt["body_checks"]) == len(source["documents"]) == 20
        assert all(prompt["body_checks"])
        checks += 20
        if row["status"] == "budget_exhausted":
            assert row["finish_reason"] != "stop"
            assert "final_text" not in row
        if "final_text" in row:
            assert row["finish_reason"] == "stop"
            final, reasoning = answer_after_thinking(row["raw_output"], row["thinking"])
            assert final == row["final_text"]
            assert len(reasoning) == row["reasoning_chars"]
    report = {
        "questions": 10, "generations": 40,
        "full_evidence_checks": checks, "all_cross_file_checks_passed": True,
        "results_sha256": sha(out / "results.jsonl"),
        "preflight_sha256": sha(HERE / "preflight.json"),
        "semantic_accuracy_not_inferred_from_checks": True,
    }
    with (HERE / "artifact_audit.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
