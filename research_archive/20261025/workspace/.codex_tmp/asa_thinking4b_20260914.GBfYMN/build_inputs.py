"""Freeze ten first-round inputs from the completed audited 20-question run."""
from pathlib import Path
import json
from protocol_helpers import sha, digest, messages_from_prompt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "asa_e2e20_20260914.9KSGhY"
SELECTED = [1, 4, 6, 7, 8, 12, 14, 15, 19, 20]


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    calls = load_jsonl(SOURCE / "attempt1/calls.jsonl")
    audits = load_jsonl(SOURCE / "attempt1/pipeline_audit.jsonl")
    sources = {}
    rows = []
    for qid in SELECTED:
        call = next(x for x in calls if x["event"] == "generation_end"
                    and x["question_index"] == qid
                    and x["stage"] == "conclusion_generation")
        evidence = next(x for x in audits if x["event"] == "prompt_evidence"
                        and x["question_index"] == qid
                        and x["prompt_sha256"] == call["prompt_sha256"])
        assert call["status"] == "ok"
        assert evidence["evidence_count"] == len(evidence["documents"]) == 20
        assert all(x["full_body_present"] for x in evidence["body_checks"])
        messages = messages_from_prompt(call["prompt"])
        user = messages[1]["content"]
        assert "\nround: 1/2\n" in user
        hypothesis = json.loads(user.split("\nhypothesis: ", 1)[1].split("\nround:", 1)[0])
        assert hypothesis["question"] == call["question"]
        rows.append({
            "question_id": qid, "question": call["question"],
            "messages": messages, "hypothesis": hypothesis,
            "documents": evidence["documents"],
            "messages_sha256": digest(messages),
            "source_prompt_sha256": call["prompt_sha256"],
            "source_call_index": call["call_index"],
            "seed": 20260914 + qid,
        })
    for path in [SOURCE / "PROTOCOL.md", SOURCE / "attempt1/calls.jsonl",
                 SOURCE / "attempt1/pipeline_audit.jsonl", HERE / "PROTOCOL.md",
                 Path(__file__), HERE / "protocol_helpers.py"]:
        sources[str(path)] = sha(path)
    out = {"mode": "frozen_first_round_reader_factorial", "selected_ids": SELECTED,
           "source_sha256": sources, "rows": rows,
           "reference_answers_passed_to_model": False}
    with (HERE / "inputs.json").open("x") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"questions": len(rows), "evidence_per_question": 20,
                      "inputs_sha256": sha(HERE / "inputs.json")}))


if __name__ == "__main__":
    main()
