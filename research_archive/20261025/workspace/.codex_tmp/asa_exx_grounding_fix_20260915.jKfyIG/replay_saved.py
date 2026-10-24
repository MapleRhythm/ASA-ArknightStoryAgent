"""Revalidate saved outputs only. This is not fresh inference or a new score."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--calls", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    events = [json.loads(line) for line in args.calls.read_text().splitlines()]
    requests = {(r["question_index"], r["call"]): r for r in events if r["event"] == "request"}
    responses = {(r["question_index"], r["call"]): r for r in events if r["event"] == "response"}
    rows = []
    for q in range(1, 21):
        eligible = [
            r for (qid, _), r in requests.items()
            if qid == q and "\ntask: grounded_action_generation\n" in r["prompt"]
        ]
        for phase, request in (("first", eligible[0]), ("last", eligible[-1])):
            prompt = request["prompt"]
            brief = prompt.split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            mapping = evidence_id_text_map_from_prompt(brief)
            assert len(mapping) == 20
            evidence = [{"document": {"id": eid, "clean_text": body}} for eid, body in mapping.items()]
            hypothesis = HypothesisDocument(**json.loads(prompt.split("\nhypothesis: ", 1)[1].splitlines()[0]))
            round_text = prompt.split("\nround: ", 1)[1].splitlines()[0]
            final = round_text.split("/")[0] == round_text.split("/")[1]
            raw = responses[q, request["call"]]["text"]
            candidate = normalize_conclusion_payload(
                json.loads(raw), question=request["question"], dialogue_context="",
                current_intent=hypothesis.intent, current_hypothesis=hypothesis,
                max_round_reached=final,
            )
            candidate = validate_conclusion_grounding(
                question=request["question"], hypothesis=hypothesis, evidence=evidence,
                conclusion=candidate, max_round_reached=final,
                mode="evidence_id", evidence_prompt_text=brief,
            )
            rows.append({
                "q": q, "phase": phase, "call": request["call"], "question": request["question"],
                "raw": raw, "action": candidate.next_action, "answer": candidate.answer,
                "issues": candidate.generation_diagnostics.get("grounding_issues", []),
                "warnings": candidate.grounding_warnings,
                "partial_facts": [
                    item["fact"]["fact"]
                    for item in candidate.generation_diagnostics.get("grounding_partial_support", [])
                ],
            })
    with args.out.open("x") as f:
        json.dump({"kind": "saved_output_validator_replay", "rows": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"rows": len(rows), "accepted_last": [
        r["q"] for r in rows if r["phase"] == "last" and r["action"] == "answer_directly"
    ]}))


if __name__ == "__main__":
    main()
