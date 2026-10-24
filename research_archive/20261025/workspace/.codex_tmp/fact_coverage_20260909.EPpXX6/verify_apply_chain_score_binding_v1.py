"""Isolate/test/replay/guard-apply coupled chain-score/text metadata repair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from verify_apply_question_echo_v1 import full_tests
from apply_token_rejection_patch_v1 import scripts, CONFIG

RELATIVE = "asa_arknight_story_agent/retrieval/hybrid_components/hybrid_chain_scoring.py"
CANDIDATE = HERE / "chain_score_binding_candidate/src"
PATCH = HERE / "chain_score_binding_candidate.py"
TEST = HERE / "test_chain_score_text_binding.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_chain_score_text_binding.py"
ECHO_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_question_echo_grounding.py"
PREPARED = HERE / "chain_score_binding_prepared_v1.json"
VERIFIED = HERE / "chain_score_binding_candidate_verified_v1.json"
PREAPPLY = HERE / "chain_score_binding_preapply_v1.json"
APPLIED = HERE / "chain_score_binding_applied_verified_v1.json"


def tests(source, xml, baseline=False, canonical=False):
    assert not xml.exists()
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.retrieval.hybrid_components import hybrid_chain_scoring as module
    assert Path(module.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    files = [TEST] if baseline else full_tests(ECHO_TEST) + [CANONICAL_TEST if canonical else TEST]
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                       *(str(p) for p in files), "--junitxml=" + str(xml)])
    expected = {"tests": 14, "failures": 9, "errors": 0, "skipped": 0} if baseline else {
        "tests": 362, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(xml) == expected and code == (1 if baseline else 0)
    origins = {}
    for name, imported in sys.modules.copy().items():
        filename = getattr(imported, "__file__", None)
        if filename and (name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent.")):
            actual = Path(filename).resolve()
            assert actual.is_relative_to(source.resolve())
            origins[name] = str(actual)
    return origins


def replay():
    inputs = {
        "fixed79": (RUN / "fixed79.current_quote.matrix.json", RUN / "fixed79_configured_lora_ctx12288/predictions.json",
                    HERE / "question_echo_fixed79_replay_v1.json", 79),
        "paired116": (HERE / "fact_coverage_pairs116_v1.matrix.json", HERE / "paired116_ctx12288_v1/predictions.json",
                      HERE / "question_echo_paired116_replay_v1.json", 116),
        "native8b58": (HERE / "qwen3_8b_native58_v1.matrix.json", HERE / "qwen3_8b_native58_ctx12288_v1/predictions.json",
                       HERE / "question_echo_native8b58_replay_v1.json", 58),
        "no_hints58": (HERE / "without_reasoning_hints58_v1.matrix.json", HERE / "without_reasoning_hints58_ctx12288_v1/predictions.json",
                       HERE / "question_echo_no_hints58_replay_v1.json", 58),
    }
    reports = {}
    for name, (matrix, predictions, old, expected) in inputs.items():
        destination = HERE / f"chain_score_binding_{name}_replay_v1.json"
        assert not destination.exists()
        subprocess.run([
            sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
            "--validator", "after", "--source-root", str(CANDIDATE),
            "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(destination),
        ], check=True)
        before, after = load(old), load(destination)
        assert len(before["results"]) == len(after["results"]) == expected
        assert before["results"] == after["results"], "unexpected change outside retrieval assembly"
        reports[name] = {"states": expected, "changed_results": 0, "replay_sha256": sha(destination),
                         "inputs_sha256": {str(p): sha(p) for p in (matrix, predictions, old)}}
    return reports


def frozen(pre, sourcekey):
    assert snapshot(SOURCE) == pre[sourcekey]
    assert sha(CONFIG) == pre["config_sha256"] and scripts() == pre["dataset_scripts_sha256"]
    assert sha(PATCH) == pre["patch_sha256"] and sha(TEST) == pre["test_sha256"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == pre["head"]


def main():
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not PREPARED.exists() and not CANDIDATE.exists() and not CANONICAL_TEST.exists()
        before = snapshot(SOURCE)
        assert before == load(HERE / "checkpoint_v19_state.json")["runtime_source_sha256"]
        for relative in before:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, target)
        assert snapshot(CANDIDATE) == before
        shutil.copyfile(PATCH, CANDIDATE / RELATIVE)
        candidate = snapshot(CANDIDATE)
        assert [k for k in before if before[k] != candidate[k]] == [RELATIVE]
        save(PREPARED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before, "candidate_source_hashes": candidate,
            "patch_sha256": sha(PATCH), "test_sha256": sha(TEST),
            "config_sha256": sha(CONFIG), "dataset_scripts_sha256": scripts(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        })
    elif mode == "baseline":
        frozen(load(PREPARED), "canonical_source_hashes")
        tests(SOURCE, HERE / "chain_score_binding_baseline14_v1.xml", baseline=True)
    elif mode == "candidate":
        assert not VERIFIED.exists()
        pre = load(PREPARED)
        frozen(pre, "canonical_source_hashes")
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        assert counts(HERE / "chain_score_binding_baseline14_v1.xml") == {
            "tests": 14, "failures": 9, "errors": 0, "skipped": 0}
        xml = HERE / "chain_score_binding_candidate362_v1.xml"
        origins = tests(CANDIDATE, xml)
        replays = replay()
        frozen(pre, "canonical_source_hashes")
        save(VERIFIED, {**pre, "prepared_sha256": sha(PREPARED),
                        "candidate_tests": counts(xml), "test_origins": origins, "replays": replays,
                        "evaluation_only": True, "training_ready": False,
                        "new_answer_accuracy": None, "new_full_rag_recall": None})
    elif mode == "apply":
        assert not PREAPPLY.exists() and not APPLIED.exists() and not CANONICAL_TEST.exists()
        verified = load(VERIFIED)
        frozen(verified, "canonical_source_hashes")
        assert verified["prepared_sha256"] == sha(PREPARED)
        assert verified["candidate_tests"] == {"tests": 362, "failures": 0, "errors": 0, "skipped": 0}
        assert all(r["changed_results"] == 0 for r in verified["replays"].values())
        backup = HERE / "chain_score_binding.original.py"
        assert not backup.exists()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == verified["canonical_source_hashes"][RELATIVE]
        save(PREAPPLY, {
            **load(PREPARED), "candidate_verified_sha256": sha(VERIFIED),
            "backup_path": str(backup), "old_target_sha256": sha(backup),
            "applied_sha256": {
                "release/ASA-ArknightStoryAgent/src/" + RELATIVE: sha(PATCH),
                "release/ASA-ArknightStoryAgent/tests/test_chain_score_text_binding.py": sha(TEST),
            },
        })
        shutil.copyfile(PATCH, SOURCE / RELATIVE)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
    else:
        assert not APPLIED.exists()
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes")
        assert all(sha(ROOT / p) == h for p, h in pre["applied_sha256"].items())
        xml = HERE / "chain_score_binding_canonical362_v1.xml"
        origins = tests(SOURCE, xml, canonical=True)
        frozen(pre, "candidate_source_hashes")
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY), "applied_sha256": pre["applied_sha256"],
            "canonical_tests": counts(xml), "origins": origins, "backup_path": pre["backup_path"],
            "only_chain_winner_binding_changed": True, "training_ready": False,
            "new_certified_correct_answers": 0, "models_changed": False, "services_restarted": False,
            "historical_generation_replays_unchanged": {"fixed79": 79, "paired116": 116,
                                                        "native8b58": 58, "no_hints58": 58},
            "pending_pointer_preflight_source_expired": True,
        })
    print(json.dumps({"stage": mode, "done": True, "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
