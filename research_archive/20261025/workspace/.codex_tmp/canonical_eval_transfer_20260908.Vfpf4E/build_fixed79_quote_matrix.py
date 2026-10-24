"""Replay all 79 historical visible evidence states using 241's quote prompt.

No answers or IDs from gold influence packing. This does NOT restore text that
was already absent from historical evidence, and it does NOT run retrieval.
"""
import dataclasses
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


run = Path(__file__).resolve().parent
sys.path.insert(0, str(run / "after/src"))
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def field(prompt, name):
    match = re.search(rf"(?m)^{name}: (.*)$", prompt)
    if not match:
        raise ValueError(f"missing {name}")
    return match[1]


gold = json.loads((run / "fixed79_gold.json").read_text())
config = json.loads((run / "after/configs/runtime_gpu_reranker_qwen35_4b.json").read_text())["inference"]
output = []
for row in gold:
    original = row["conversations"][0]["value"]
    question = field(original, "question")
    hyp = json.loads(field(original, "hypothesis"))
    hyp["question"] = question
    hyp.setdefault("intent", "plot_reasoning")
    hyp.setdefault("dialogue_context", "")
    allowed = {f.name for f in dataclasses.fields(HypothesisDocument)}
    if set(hyp) - allowed:
        raise ValueError(f"unexpected hypothesis fields: {set(hyp) - allowed}")
    hypothesis = HypothesisDocument(**hyp)
    current_round, max_rounds = map(int, field(original, "round").split("/"))
    text = original.split("\nevidence:\n", 1)[1].split("\noutput_schema:", 1)[0]
    blocks = evidence_id_text_map_from_prompt(text)
    if not blocks:
        raise ValueError("historical row contains no parseable evidence")
    evidence = [{
        "document": {
            "id": f'fixed79:{row["id"]}:{eid}', "clean_text": body,
        },
    } for eid, body in blocks.items()]
    packing = []
    prompt, brief = build_minimal_conclusion_prompt(
        question=question, current_hypothesis=hypothesis,
        evidence=evidence, prompt_evidence=evidence,
        current_round=current_round, max_retrieval_rounds=max_rounds,
        prompt_evidence_top_k=config["prompt_evidence_top_k"],
        evidence_max_chars_per_doc=config["prompt_evidence_max_chars_per_doc"],
        evidence_max_total_chars=config["prompt_conclusion_evidence_max_total_chars"],
        grounding_mode="quote", evidence_packing_trace=packing,
    )
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*)",
        prompt, re.DOTALL,
    )
    if not match:
        raise ValueError("unknown runtime prompt syntax")
    output.append({
        "id": f'{row["id"]}::current_quote_fixed79', "source_id": row["id"],
        "question": question, "renderer": "after", "template": "runtime_raw",
        "question_family": "", "system": match[1], "user": match[2],
        "runtime_prompt": prompt, "runtime_prompt_sha256": sha(prompt),
        "evidence_brief": brief, "evidence_sha256": sha(brief),
        "assistant_prefix": match[3], "hypothesis": dataclasses.asdict(hypothesis),
        "current_round": current_round, "max_retrieval_rounds": max_rounds,
        "prompt_evidence": evidence,
        "evidence_id_to_corpus_id": {
            item["evidence_id"]: item["source_id"] for item in packing if item["status"] == "included"
        },
        "packing_trace": packing, "historical_visible_evidence_sha256": sha(text),
        "historical_prompt_sha256": sha(original), "original_evidence_ids": list(blocks),
        "evaluation_only": True, "training_ready": False,
    })
if len(output) != 79 or len({row["id"] for row in output}) != 79:
    raise ValueError("must retain all 79 rows")
questions = Counter(re.sub(r"\s+", "", row["question"]).strip("？?。！!，,；;：:\"'") for row in output)
report = {
    "version": "asa_fixed79_current_quote_v1", "evaluation_only": True,
    "training_ready": False, "question_count": len(questions), "rows": output,
    "source_gold_sha256": hashlib.sha256((run / "fixed79_gold.json").read_bytes()).hexdigest(),
    "source_manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "limitations": [
        "All 79 historical evidence states, 58 distinct normalized questions; not independent families.",
        "Current quote prompt over historical visible evidence, not fresh retrieval or full E2E.",
        "Reference outputs are not part of the generated matrix; historical gold may be flawed.",
        "No restoration of text absent from original frozen prompts.",
        "History's Exx output protocol differs from this quote protocol; no direct score equivalence.",
    ],
}
with (run / "fixed79.current_quote.matrix.json").open("x", encoding="utf-8") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({
    "rows": len(output), "distinct_normalized_questions": len(questions),
    "duplicate_question_groups": sum(n > 1 for n in questions.values()),
    "budget_skip_rows": sum(any(e["status"] == "budget_skip" for e in r["packing_trace"]) for r in output),
    "max_evidence_chars": max(len(r["evidence_brief"]) for r in output),
}))
