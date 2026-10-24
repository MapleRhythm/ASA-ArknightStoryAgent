"""Preflight the exact process-local bypass without any inference."""
import argparse
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo / "src"))
spec = importlib.util.spec_from_file_location("experiment", Path(__file__).with_name("run_noguard50.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
events = []
passthrough = module.install_no_answer_validation(events.append)
from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult
from asa_arknight_story_agent.inference.generation import answer_conclusion_generation as ac
from asa_arknight_story_agent.inference.generation import direct_answer_generation as da

for action in ("answer_directly", "retrieve_more", "abstain"):
    # Deliberately unsupported content and invalid citation must pass unchanged.
    candidate = ConclusionResult(
        next_action=action, answer="This is an unsupported preflight fixture, not an evaluation answer.",
        missing_slots=[], clarification_question="", follow_up_hypothesis=None,
        supported_facts=[{"fact": "Entirely unsupported", "evidence_refs": [{"evidence_id": "E9999"}]}],
    )
    before = asdict(candidate)
    output = passthrough(conclusion=candidate, evidence=[], question="fixture", mode="evidence_id")
    assert output is candidate
    after = asdict(candidate)
    before.pop("generation_diagnostics")
    after.pop("generation_diagnostics")
    assert before == after
assert ac.validate_conclusion_grounding is passthrough
assert da.validate_conclusion_grounding is passthrough
assert len(events) == 3 and all(event["input_equals_output"] for event in events)
print("PASS: 3 actions including invalid citation and unsupported content are unchanged; no inference run.")
