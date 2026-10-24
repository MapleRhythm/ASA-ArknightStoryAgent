"""Read-only current-runtime probe; no retrieval, model call or data rewrite."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "release/ASA-ArknightStoryAgent/src"))
sys.path.insert(0, str(ROOT / "scripts"))
from build_soda_api_verifier_dataset import extract_hypothesis, extract_question
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import (
    build_heuristic_follow_up_hypothesis, merge_hypotheses,
)
from asa_arknight_story_agent.inference.planning.missing_slot_queries import clean_missing_slots_for_retrieval


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


diagnostic = "answer_directly 缺少可校验 quote 支撑"
machine_diagnostic = "supported_fact_1_ref_1_quote_not_found"
question = "甲为什么离开城里？"
clean = HypothesisDocument(
    question=question, intent="plot_fact", query_type="reason",
    entities=["甲"], keywords=["离开", "城里"], expected_answer_type="原因",
)
polluted = HypothesisDocument(
    question=question, intent="plot_fact", query_type="reason",
    entities=["甲", "answer_directly"],
    keywords=["离开", "missing_supported_facts", diagnostic],
    expected_answer_type="原因",
)
cases = {
    "clean_hypothesis_diagnostic_missing_slots": (
        clean, [diagnostic, machine_diagnostic, "离开的具体原因"],
    ),
    "prepolluted_hypothesis_empty_slots": (polluted, []),
}
historical_path = HERE.parent / "verifier_contract_patch_v1.nKJjAQ/historical_protocol_conflict_example_v1.json"
historical = json.loads(historical_path.read_text())
source = extract_hypothesis(historical["prompt"])
historical_question = extract_question(historical["prompt"])
cases["historical_input_hypothesis_replayed_now"] = (
    HypothesisDocument(
        question=str(source.get("question") or historical_question),
        intent=str(source.get("intent") or "plot_fact"),
        query_type=str(source.get("query_type") or "reason"),
        entities=source.get("entities", []), keywords=source.get("keywords", []),
        expected_answer_type=str(source.get("expected_answer_type") or "原因"),
        dialogue_context=str(source.get("dialogue_context") or ""),
    ), [],
)
results = []
for name, (hypothesis, slots) in cases.items():
    follow_up = build_heuristic_follow_up_hypothesis(
        hypothesis.question, hypothesis, slots,
    )
    merged = merge_hypotheses(hypothesis, follow_up)
    results.append({
        "name": name, "input": asdict(hypothesis), "missing_slots": slots,
        "cleaned_missing_slots": clean_missing_slots_for_retrieval(slots),
        "follow_up": asdict(follow_up), "merged": asdict(merged),
    })
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "evaluation_only": True, "retrieval_was_run": False, "production_changed": False,
    "source_sha256": {
        relative: sha(ROOT / "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent" / relative)
        for relative in ("inference/planning/follow_up_hypothesis.py", "inference/planning/missing_slot_queries.py")
    },
    "historical_example_sha256": sha(historical_path), "results": results,
    "limitation": "Replaying an already-polluted historical input proves propagation, not fresh current-run creation.",
}
with (HERE / "hypothesis_diagnostic_probe_v1.json").open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
print(json.dumps([{
    "name": r["name"], "cleaned_missing_slots": r["cleaned_missing_slots"],
    "merged_entities": r["merged"]["entities"], "merged_keywords": r["merged"]["keywords"],
} for r in results], ensure_ascii=False, indent=2))
