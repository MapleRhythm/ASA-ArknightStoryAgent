"""Synthetic, read-only regression probes of structured quote boundaries."""
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys


run = Path(__file__).resolve().parent
sys.path.insert(0, str(run / "after/src"))
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.payload.normalization import normalize_conclusion_payload
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

question = "角色甲在做什么？"
visible = "[E1] 角色甲在窗边读书。角色乙在街上散步。"
evidence = [{"id": "synthetic_1", "text": "角色甲在窗边读书。角色乙在街上散步。"}]
hyp = HypothesisDocument(
    question=question, intent="plot_fact", query_type="fact",
    entities=["角色甲"], keywords=["读书"], expected_answer_type="行为", dialogue_context="",
)
supported = [{"fact": "角色甲在窗边读书。", "evidence_refs": [{"evidence_id": "E1", "quote": "角色甲在窗边读书。"}]}]
invented = "角色乙已经投降。"
cases = [
    ("supported_control", [], "角色甲在窗边读书。"),
    ("inferred_without_references", [{"fact": invented}], "角色甲在窗边读书。角色乙已经投降。"),
    ("inferred_with_fake_reference", [{"fact": invented, "evidence_refs": [{"evidence_id": "E99", "quote": invented}]}],
     "角色甲在窗边读书。角色乙已经投降。"),
]
results = []
for name, inferred, answer in cases:
    payload = {"next_action": "answer_directly", "supported_facts": copy.deepcopy(supported),
               "inferred_facts": copy.deepcopy(inferred), "final_answer": answer}
    parsed = normalize_conclusion_payload(
        payload, question=question, dialogue_context="", current_intent=hyp.intent,
        current_hypothesis=hyp, max_round_reached=False,
    )
    output = validate_conclusion_grounding(
        question=question, hypothesis=hyp, evidence=copy.deepcopy(evidence),
        conclusion=copy.deepcopy(parsed), max_round_reached=False,
        mode="quote", evidence_prompt_text=visible,
    )
    results.append({
        "name": name, "payload": payload, "parsed": dataclasses.asdict(parsed),
        "grounded": dataclasses.asdict(output),
        "unsupported_inferred_assertion_exposed": output.next_action == "answer_directly" and invented in output.answer,
    })
report = {
    "version": "inferred_fact_boundary_synthetic_v1", "evaluation_only": True, "training_ready": False,
    "synthetic_not_dataset_accuracy": True, "canonical_frozen_source": str(run / "after/src"),
    "question": question, "visible_evidence": visible, "results": results,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
with (run / "inferred_fact_boundary_synthetic_v1.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([
    {"name": r["name"], "action": r["grounded"]["next_action"],
     "answer": r["grounded"]["answer"], "unsupported_inference_exposed": r["unsupported_inferred_assertion_exposed"],
     "parsed_inferred": r["parsed"]["inferred_facts"]}
    for r in results
], ensure_ascii=False))
