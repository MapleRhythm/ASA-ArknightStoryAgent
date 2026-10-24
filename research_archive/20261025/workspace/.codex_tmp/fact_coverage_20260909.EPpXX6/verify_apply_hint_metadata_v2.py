"""Guarded structural metadata cleanup, independent of answer/pointer experiments."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot, counts
from verify_apply_prompt_priority_v1 import previous_tests
from apply_token_rejection_patch_v1 import CONFIG, scripts

RELATIVE = "asa_arknight_story_agent/inference/retrieval/minirag_prompt_hints.py"
CANDIDATE = HERE / "hint_metadata_candidate_v2/src"
PATCH = HERE / "hint_metadata_candidate.py"
TEST = HERE / "test_hint_metadata_isolation_v2.py"
CANONICAL_TEST = ROOT / "release/ASA-ArknightStoryAgent/tests/test_hint_metadata_isolation.py"
PRE = HERE / "hint_metadata_prepared_v2.json"
VERIFIED = HERE / "hint_metadata_candidate_verified_v2.json"
PREAPPLY = HERE / "hint_metadata_preapply_v2.json"
APPLIED = HERE / "hint_metadata_applied_verified_v2.json"


def frozen(pre, source_key):
    assert snapshot(SOURCE) == pre[source_key]
    assert sha(PATCH) == pre["patch_sha256"] and sha(TEST) == pre["test_sha256"]
    assert sha(CONFIG) == pre["config_sha256"] and scripts() == pre["dataset_scripts_sha256"]
    assert all(sha(Path(p)) == h for p, h in pre["verification_inputs_sha256"].items())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == pre["head"]


def tests(source, mode):
    xml = HERE / f"hint_metadata_{mode}_v2.xml"
    assert not xml.exists()
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.retrieval import minirag_prompt_hints
    assert Path(minirag_prompt_hints.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    files = [TEST] if mode == "baseline" else previous_tests() + [CANONICAL_TEST if mode == "canonical" else TEST]
    result = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short",
                         *(str(p) for p in files), "--junitxml=" + str(xml)])
    totals = counts(xml)
    assert totals["errors"] == totals["skipped"] == 0
    if mode == "baseline":
        assert result == 1 and totals["tests"] == 13 and 0 < totals["failures"] < 13
    else:
        assert result == 0 and totals == {"tests": 375, "failures": 0, "errors": 0, "skipped": 0}
    origins = {}
    for name, module in sys.modules.copy().items():
        path = getattr(module, "__file__", None)
        if path and (name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent.")):
            assert Path(path).resolve().is_relative_to(source.resolve())
            origins[name] = path
    return {"counts": totals, "origins": origins, "xml_sha256": sha(xml)}


def replay():
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.retrieval.minirag_prompt_hints import render_minirag_hints_for_prompt
    old = load(HERE / "hint_metadata_leak_audit_v1.json")
    rows = [r for r in load(HERE / "fact_coverage_pairs116_v1.matrix.json")["rows"]
            if r["renderer"] == "original_prompt_v1"]
    rows += load(HERE / "susie_handoff4_v1.matrix.json")["rows"]
    assert len(rows) == len(old["results"]) == 62
    results = []
    for row, before in zip(rows, old["results"]):
        assert row["id"] == before["id"] and before["canonical_hint_reproduces_saved"]
        visible = [row["prompt_evidence"][r["input_index"]]
                   for r in row["packing_trace"] if r["status"] == "included"]
        hints = render_minirag_hints_for_prompt(visible, HypothesisDocument(**row["hypothesis"]))
        assert hints == before["sanitized_hint_candidate"]
        results.append({"id": row["id"], "hints": hints, "changed": hints != before["canonical_hints"]})
    assert not any(r["changed"] for r in results[:58]) and all(r["changed"] for r in results[58:])
    return {"states": 62, "historical58_hints_unchanged": True, "changed_real_handoff_states": 4,
            "results": results, "new_generation_calls": 0, "new_answer_accuracy": None}


def main():
    mode = sys.argv[1]
    assert mode in {"prepare", "baseline", "candidate", "apply", "canonical"}
    if mode == "prepare":
        assert not PRE.exists() and not CANDIDATE.exists() and not CANONICAL_TEST.exists()
        before = snapshot(SOURCE)
        assert before == load(HERE / "checkpoint_v21_state.json")["runtime_source_sha256"]
        postpath = HERE / "single_pointer_handoff4_postflight_v1.json"
        post = load(postpath)
        assert post["protocol_verified"]
        assert all(sha(Path(p)) == h for p, h in post["artifacts_sha256"].items())
        for relative in before:
            target = CANDIDATE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / relative, target)
        assert snapshot(CANDIDATE) == before
        shutil.copyfile(PATCH, CANDIDATE / RELATIVE)
        after = snapshot(CANDIDATE)
        assert set(before) == set(after) and [p for p in before if before[p] != after[p]] == [RELATIVE]
        inputs = previous_tests() + [
            Path(__file__), HERE / "verify_apply_prompt_priority_v1.py",
            HERE / "hint_metadata_leak_audit_v1.json", HERE / "fact_coverage_pairs116_v1.matrix.json",
            HERE / "susie_handoff4_v1.matrix.json", HERE / "checkpoint_v21_state.json", postpath,
        ]
        save(PRE, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_source_hashes": before, "candidate_source_hashes": after,
            "patch_sha256": sha(PATCH), "test_sha256": sha(TEST),
            "config_sha256": sha(CONFIG), "dataset_scripts_sha256": scripts(),
            "verification_inputs_sha256": {str(p): sha(p) for p in inputs},
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        })
    elif mode == "baseline":
        frozen(load(PRE), "canonical_source_hashes")
        result = tests(SOURCE, mode)
        frozen(load(PRE), "canonical_source_hashes")
        save(HERE / "hint_metadata_baseline_v2.json", result)
    elif mode == "candidate":
        assert not VERIFIED.exists()
        pre = load(PRE)
        frozen(pre, "canonical_source_hashes")
        assert snapshot(CANDIDATE) == pre["candidate_source_hashes"]
        assert load(HERE / "hint_metadata_baseline_v2.json")["counts"]["failures"] > 0
        result, replayed = tests(CANDIDATE, mode), replay()
        frozen(pre, "canonical_source_hashes")
        save(VERIFIED, {**pre, "prepared_sha256": sha(PRE), "candidate_tests": result,
                        "hint_replay": replayed, "training_ready": False})
    elif mode == "apply":
        assert not PREAPPLY.exists() and not APPLIED.exists() and not CANONICAL_TEST.exists()
        verified = load(VERIFIED)
        frozen(verified, "canonical_source_hashes")
        assert verified["prepared_sha256"] == sha(PRE)
        assert verified["candidate_tests"]["counts"] == {"tests": 375, "failures": 0, "errors": 0, "skipped": 0}
        assert verified["hint_replay"]["historical58_hints_unchanged"]
        assert verified["hint_replay"]["changed_real_handoff_states"] == 4
        backup = HERE / "minirag_prompt_hints.original.v21.py"
        assert not backup.exists() and not (SOURCE / RELATIVE).is_symlink()
        shutil.copyfile(SOURCE / RELATIVE, backup)
        assert sha(backup) == verified["canonical_source_hashes"][RELATIVE]
        save(PREAPPLY, {
            **load(PRE), "candidate_verified_sha256": sha(VERIFIED),
            "backup_path": str(backup), "old_target_sha256": sha(backup),
            "applied_sha256": {
                "release/ASA-ArknightStoryAgent/src/" + RELATIVE: sha(PATCH),
                "release/ASA-ArknightStoryAgent/tests/test_hint_metadata_isolation.py": sha(TEST)},
        })
        shutil.copyfile(PATCH, SOURCE / RELATIVE)
        shutil.copyfile(TEST, CANONICAL_TEST)
        assert snapshot(SOURCE) == verified["candidate_source_hashes"]
    else:
        assert not APPLIED.exists()
        pre = load(PREAPPLY)
        frozen(pre, "candidate_source_hashes")
        assert all(sha(ROOT / p) == h for p, h in pre["applied_sha256"].items())
        result, replayed = tests(SOURCE, mode), replay()
        assert replayed == load(VERIFIED)["hint_replay"]
        frozen(pre, "candidate_source_hashes")
        save(APPLIED, {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preapply_sha256": sha(PREAPPLY), "applied_sha256": pre["applied_sha256"],
            "canonical_tests": result, "hint_replay": replayed, "backup_path": pre["backup_path"],
            "only_hint_extractor_input_cleanup_changed": True,
            "evidence_text_and_grounding_gates_unchanged": True, "new_generation_calls": 0,
            "training_ready": False, "models_or_config_changed": False, "services_restarted": False,
            "new_certified_answer_accuracy": None,
        })
    print(json.dumps({"stage": mode, "done": True, "training_ready": False}), flush=True)


if __name__ == "__main__":
    main()
