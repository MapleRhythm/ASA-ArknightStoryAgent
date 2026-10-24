"""Frozen offline diagnostic: retain only entailed unchanged facts, then audit coverage."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
sys.path.insert(0, str(SOURCE_ROOT))
from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import answer_from_grounded_facts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "build"])
    args = parser.parse_args()
    plan_path = HERE / "semantic_filter_plan_v1.json"
    source = HERE / "canonical_fixed79_validation_v1.json"
    probe_source = HERE / "cited_claim_probe13_v2.json"
    renderer = SOURCE_ROOT / "asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py"
    if args.mode == "freeze":
        assert not probe_source.exists(), "Freeze the candidate rule before inspecting completed probe outcomes"
        plan = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True, "training_ready": False,
            "script_sha256": sha(Path(__file__)), "source_analysis_sha256": sha(source),
            "renderer_sha256": sha(renderer),
            "rule": [
                "Use only successful claim-judge results with relation=entailed.",
                "Never rewrite, shorten or add a fact, reference or answer from external knowledge.",
                "Use the existing canonical fact renderer for the retained unchanged facts.",
                "If any claim audit failed, or no facts survive, emit a diagnostic abstention, never fall back to the old answer.",
                "Judge resulting shown-answer content separately; surviving factual claims do not establish task completeness.",
            ],
            "limitations": [
                "Seven deliberately chosen development states, not a random or independent evaluation.",
                "An experimental filter using an imperfect model judge, not a verified production semantic gate.",
                "No production code, dataset labels, weights, default model or served answers are changed.",
            ],
        }
        write_new(plan_path, plan)
        print("Frozen offline filtering rule; no answer candidates written.")
        return

    plan = load(plan_path)
    assert sha(Path(__file__)) == plan["script_sha256"]
    assert sha(source) == plan["source_analysis_sha256"] and sha(renderer) == plan["renderer_sha256"]
    probe = load(probe_source)
    rows = {r["id"]: r for r in load(source)["results"]}
    judgments = {r["id"]: r for r in probe["results"]}
    cases = {}
    for case in probe["cases"]:
        cases.setdefault(case["parent_id"], []).append(case)
    assert len(cases) == 7 and sum(map(len, cases.values())) == 13
    candidates, traces = [], []
    for parent_id, facts in cases.items():
        original = rows[parent_id]
        assert not original["grounded"].get("inferred_facts")
        assert len(facts) == len(original["grounded"]["supported_facts"])
        kept, excluded = [], []
        errors = [c["id"] for c in facts if judgments[c["id"]]["status"] != "ok"]
        for case in sorted(facts, key=lambda c: c["fact_index"]):
            fact = original["grounded"]["supported_facts"][case["fact_index"] - 1]
            assert fact["fact"] == case["claim"] and fact["evidence_refs"] == case["original_refs"]
            judgment = judgments[case["id"]]
            if not errors and judgment["status"] == "ok" and judgment["judgement"]["relation"] == "entailed":
                kept.append(deepcopy(fact))
            else:
                excluded.append({"fact_index": case["fact_index"], "claim": case["claim"],
                                 "status": judgment["status"], "relation": (judgment.get("judgement") or {}).get("relation")})
        candidate = deepcopy(original)
        candidate["id"] = parent_id + "::semantic_filter_v1"
        candidate["grounded"]["supported_facts"] = kept
        candidate["grounded"]["inferred_facts"] = []
        if kept:
            answer = answer_from_grounded_facts(SimpleNamespace(supported_facts=kept, inferred_facts=[], answer=""))
            assert answer.strip()
            candidate["grounded"]["answer"] = answer
            candidate["grounded"]["next_action"] = "answer_directly"
        else:
            candidate["grounded"]["next_action"] = "abstain"
            candidate["grounded"]["answer"] = "当前引用未通过语义核验，暂不能给出可验证的答案。"
        # Judge messages use only question, original visible evidence and new
        # shown answer/action, never parsed output, previous verdict or trace.
        candidates.append(candidate)
        traces.append({
            "parent_id": parent_id, "candidate_id": candidate["id"],
            "kept_fact_count": len(kept), "excluded": excluded, "audit_failures": errors,
            "answer_changed": candidate["grounded"]["answer"] != original["grounded"]["answer"],
        })
    write_new(HERE / "semantic_filter7_candidate_v1.json", {
        "evaluation_only": True, "training_ready": False, "new_generation": False, "new_retrieval": False,
        "plan_sha256": sha(plan_path), "claim_audit_sha256": sha(probe_source),
        "results": candidates,
    })
    write_new(HERE / "semantic_filter7_trace_v1.json", {"traces": traces, "training_ready": False})
    print(json.dumps({"candidate_states": len(candidates), "traces": traces}, ensure_ascii=False))


if __name__ == "__main__":
    main()
