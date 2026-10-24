"""Full325 regression and frozen79/116 replay for the refusal-only token fallback."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import counts, regression_files, save, snapshot

RELATIVE = "asa_arknight_story_agent/inference/grounding/token_checks.py"
CANDIDATE = HERE / "token_rejection_candidate/src"
REFUSAL = "现有检索证据不足以给出可校验的回答。"


def run_tests(source, output):
    assert not output.exists()
    sys.path.insert(0, str(source))
    import asa_arknight_story_agent.inference.grounding.token_checks as module
    assert Path(module.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    files = regression_files() + [
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_normalized_quote.py",
        ROOT / "release/ASA-ArknightStoryAgent/tests/test_empty_quote_diagnostic_terms.py",
        HERE / "test_token_rejection_display.py",
    ]
    code = pytest.main([
        "-q", "-p", "no:cacheprovider", "--tb=short",
        *(str(path) for path in files), "--junitxml=" + str(output),
    ])
    origins = {}
    for name, imported in sys.modules.copy().items():
        if name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent."):
            filename = getattr(imported, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                assert path.is_relative_to(source.resolve()), f"Mixed source: {name}: {path}"
                origins[name] = str(path)
    assert code == 0 and counts(output) == {"tests": 325, "failures": 0, "errors": 0, "skipped": 0}
    return origins


def replay(label, matrix, predictions, original, expected_rows):
    output = HERE / f"token_rejection_{label}_replay_v1.json"
    assert not output.exists()
    subprocess.run([
        sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
        "--validator", "after", "--source-root", str(CANDIDATE),
        "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(output),
    ], check=True)
    before, after = load(original), load(output)
    assert len(before["results"]) == len(after["results"]) == expected_rows
    assert [r["id"] for r in before["results"]] == [r["id"] for r in after["results"]]
    changes = []
    for a, b in zip(before["results"], after["results"]):
        assert a["status"] == b["status"] and a.get("parsed") == b.get("parsed")
        if a.get("grounded") == b.get("grounded"):
            continue
        old, new = a["grounded"], b["grounded"]
        assert old["next_action"] == new["next_action"] == "abstain"
        assert old["answer"].startswith("当前证据只能确认以下片段事实：") or old["answer"] == "现有检索证据不足以确认答案所需的关键表述。"
        assert new["answer"] == REFUSAL
        assert new["generation_diagnostics"]["grounding_status"] == "rejected_to_abstain"
        assert "token_grounding" in new["generation_diagnostics"]
        comparison = deepcopy(new)
        for key in ("answer", "generation_diagnostics", "grounding_warnings"):
            comparison[key] = old[key]
        assert comparison == old
        changes.append({
            "id": a["id"], "question": a["question"], "old_answer": old["answer"],
            "new_answer": new["answer"], "new_diagnostics": new["generation_diagnostics"],
            "action_unchanged": True,
        })
    return {
        "states": expected_rows, "old_analysis_sha256": sha(original), "replay_sha256": sha(output),
        "matrix_sha256": sha(matrix), "predictions_sha256": sha(predictions),
        "all_actions_unchanged": True, "all_direct_answers_unchanged": True,
        "changes": changes,
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "canonical_tests":
        origins = run_tests(SOURCE, HERE / "token_rejection_canonical_tests325_v1.xml")
        save(HERE / "token_rejection_canonical_test_origins_v1.json", origins)
        return
    destination = HERE / "token_rejection_candidate_verified_v1.json"
    assert not destination.exists()
    before, candidate = snapshot(SOURCE), snapshot(CANDIDATE)
    assert set(before) == set(candidate)
    assert [path for path in before if before[path] != candidate[path]] == [RELATIVE]
    assert sha(HERE / "token_checks.py") == candidate[RELATIVE]
    assert counts(HERE / "token_rejection_baseline_v1.xml") == {
        "tests": 13, "failures": 8, "errors": 0, "skipped": 0,
    }
    assert counts(HERE / "token_rejection_candidate13_v1.xml") == {
        "tests": 13, "failures": 0, "errors": 0, "skipped": 0,
    }
    origins = run_tests(CANDIDATE, HERE / "token_rejection_candidate_tests325_v1.xml")
    old79 = replay(
        "fixed79", RUN / "fixed79.current_quote.matrix.json",
        RUN / "fixed79_configured_lora_ctx12288/predictions.json",
        HERE / "empty_quote_fixed79_replay_v1.json", 79,
    )
    new116 = replay(
        "new116", HERE / "fact_coverage_pairs116_v1.matrix.json",
        HERE / "paired116_ctx12288_v1/predictions.json",
        HERE / "empty_quote_new116_replay_v1.json", 116,
    )
    assert any(
        r["id"].startswith("e9722207725f6eb8__") and r["id"].endswith("fact_coverage_prompt_v1")
        for r in new116["changes"]
    ), "The observed real fallback case must be covered"
    assert snapshot(SOURCE) == before
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "canonical_source_hashes": before, "candidate_source_hashes": candidate,
        "changed_source_path": RELATIVE,
        "candidate_tests": counts(HERE / "token_rejection_candidate_tests325_v1.xml"),
        "test_sha256": sha(HERE / "test_token_rejection_display.py"),
        "test_module_origins": origins, "fixed79": old79, "new116": new116,
        "limits": [
            "Refusal integrity fix: does not turn abstentions into correct answers or raise measured answer accuracy.",
            "Old13 failures include one refusal-wording test and diagnostic/wiring tests, not eight independent harmful answers.",
            "Token thresholds and pre-final retrieval behavior are unchanged; only final rejection display and diagnostics change.",
            "The old generated outputs, audits, evidence, model and runtime config were not replaced.",
        ],
    })
    print({
        "candidate_tests": 325, "fixed79_changed_refusals": len(old79["changes"]),
        "new116_changed_refusals": len(new116["changes"]), "all_actions_unchanged": True,
    }, flush=True)


if __name__ == "__main__":
    main()
