"""Isolated CPU replay: formatting may change, decisions and facts may not."""
import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
MODULE = SRC / "asa_arknight_story_agent/inference/grounding/grounded_fact_answers.py"
BASE_SHA = "52bf832e64093da50de4b974c235d4c7a661e7f810c96a0f07113c395fde9592"
sys.path.insert(0, str(SRC))


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def verify_only_layout_ast():
    before = ast.parse((HERE / "baseline_grounded_fact_answers.py").read_text())
    after = ast.parse((HERE / "grounded_fact_answers.py").read_text())
    restored = deepcopy(after)
    fn = next(n for n in restored.body if isinstance(n, ast.FunctionDef)
              and n.name == "answer_from_grounded_facts")
    for index in (0, 1):
        separator = fn.body[3].body[index].value.func.value
        assert isinstance(separator, ast.Constant) and separator.value == "\n"
        separator.value = "；"
    last = fn.body[-1]
    assert isinstance(last, ast.Return) and isinstance(last.value, ast.BinOp)
    separator = last.value.right.func.value
    assert isinstance(separator, ast.Constant) and separator.value == "\n"
    separator.value = "；"
    last.value = ast.BinOp(left=last.value, op=ast.Add(), right=ast.Constant(value="。"))
    assert ast.dump(before, include_attributes=False) == ast.dump(restored, include_attributes=False)


def cases():
    parent = HERE.parent
    inputs, records, errors = [], [], []
    for version, folder in (
        ("v29", "reader_chain_owner_20260910.v29"),
        ("v30", "reader_native_qwen35_20260910.v30"),
        ("v31", "reader_thinking_budget_20260910.v31"),
    ):
        directory = parent / folder
        pre_path = directory / "preflight.json"
        inputs.append(pre_path)
        pre = load(pre_path)
        paths = ([directory / kind / "generations.json" for kind in ("native4b", "native9b")]
                 if version == "v30" else [directory / "generations.json"])
        for path in paths:
            inputs.append(path)
            generated = load(path)
            for source, result in zip(pre["rows"], generated["rows"], strict=True):
                assert source["id"] == result["id"]
                kinds = ["quote"] if version == "v30" else list(result["conditions"])
                for kind in kinds:
                    item = result["conditions"][kind]
                    identity = f"{version}:{path.parent.name}:{result.get('pair_id', result['id'])}:{kind}"
                    if item["status"] != "ok":
                        errors.append({"id": identity, "error": item["error"]})
                        continue
                    evidence = (source["candidate_evidence"]
                                if version == "v29" and kind == "candidate" else source["evidence_brief"])
                    records.append({
                        "id": identity, "question": source["question"], "hypothesis": source["hypothesis"],
                        "text": item["answer_text"] if version == "v31" else item["raw_output"],
                        "evidence": evidence, "expected": item["grounded"],
                    })
    assert len(records) == 58 and len(errors) == 6
    assert len({r["id"] for r in records}) == 58
    return records, errors, inputs


def main():
    assert not (HERE / "isolated_replay_v1.json").exists()
    assert sha(MODULE) == sha(HERE / "baseline_grounded_fact_answers.py") == BASE_SHA
    verify_only_layout_ast()
    source_files = sorted(SRC.rglob("*.py"))
    initial_source = {str(p): sha(p) for p in source_files}
    records, errors, input_files = cases()
    input_hashes = {str(p): sha(p) for p in input_files}
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        normalize_minimal_conclusion_output, parse_conclusion_output,
    )
    from asa_arknight_story_agent.inference.grounding import quote_checks
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    spec = importlib.util.spec_from_file_location("layout_candidate", HERE / "grounded_fact_answers.py")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    original = quote_checks.answer_from_grounded_facts

    def replay(row):
        hypothesis = HypothesisDocument(**row["hypothesis"])
        parsed = parse_conclusion_output(
            normalize_minimal_conclusion_output(row["text"], "minimal"),
            question=row["question"], current_hypothesis=hypothesis, max_round_reached=True,
        )
        return asdict(validate_conclusion_grounding(
            question=row["question"], hypothesis=hypothesis, evidence=[], conclusion=parsed,
            max_round_reached=True, mode="quote", evidence_prompt_text=row["evidence"],
        ))

    results = []
    try:
        for row in records:
            quote_checks.answer_from_grounded_facts = original
            before = replay(row)
            assert before == row["expected"], ("baseline replay differs", row["id"])
            renderings = []

            def instrumented(conclusion):
                snapshot = deepcopy(asdict(conclusion))
                old_text = original(conclusion)
                new_text = candidate.answer_from_grounded_facts(conclusion)
                assert asdict(conclusion) == snapshot
                renderings.append((old_text, new_text))
                return new_text

            quote_checks.answer_from_grounded_facts = instrumented
            after = replay(row)
            assert {k: v for k, v in before.items() if k != "answer"} == {
                k: v for k, v in after.items() if k != "answer"
            }, ("non-presentation change", row["id"])
            # No words or existing punctuation in statements may be repaired.
            if before["answer"] != after["answer"]:
                assert before["next_action"] == after["next_action"] == "answer_directly"
                assert (before["answer"], after["answer"]) in renderings
            results.append({
                "id": row["id"], "action": after["next_action"],
                "answer_changed": before["answer"] != after["answer"],
                "baseline_answer": before["answer"], "candidate_answer": after["answer"],
                "all_non_answer_fields_identical": True,
            })
    finally:
        quote_checks.answer_from_grounded_facts = original
    assert initial_source == {str(p): sha(p) for p in source_files}
    assert input_hashes == {str(p): sha(p) for p in input_files}
    report = {
        "version": "grounded_fact_layout_isolated_replay_v33",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "production_changed": False,
        "gpu_used": False, "method": "CPU-only in-process renderer substitution, then restored",
        "only_three_join_literals_and_final_added_fullstop_changed_ast_verified": True,
        "replayed_records": len(results), "unique_development_questions": 8,
        "budget_errors_preserved_not_replayed": errors,
        "changed_answers": sum(r["answer_changed"] for r in results),
        "all_actions_facts_references_warnings_and_diagnostics_unchanged": True,
        "semantic_accuracy_gain_claimed": False, "results": results,
        "source_sha256": initial_source, "input_sha256": input_hashes,
        "candidate_sha256": sha(HERE / "grounded_fact_answers.py"),
        "verifier_sha256": sha(Path(__file__)),
    }
    save(HERE / "isolated_replay_v1.json", report)
    print(json.dumps({k: report[k] for k in (
        "replayed_records", "changed_answers", "production_changed", "semantic_accuracy_gain_claimed",
    )}), flush=True)


if __name__ == "__main__":
    main()
