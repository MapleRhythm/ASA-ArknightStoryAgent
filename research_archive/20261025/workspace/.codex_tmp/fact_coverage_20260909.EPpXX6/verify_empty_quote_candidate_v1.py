"""312 CPU tests, canonical-source isolation, and complete fixed79 replay."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from build_fact_coverage_pairs_v1 import HERE, ROOT, RUN, SOURCE, load, sha

RELATIVE = "asa_arknight_story_agent/inference/grounding/quote_validation.py"
CANDIDATE = HERE / "empty_quote_candidate/src"
OLD_XML = RUN / "builder_followup_integrity_v1.4mrTlm/canonical_combined_tests_v2.xml"


def counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def snapshot(source):
    return {str(p.relative_to(source)): sha(p) for p in source.rglob("*.py")}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def regression_files():
    root = ET.parse(OLD_XML).getroot()
    names = sorted({case.attrib["classname"] for case in root.iter("testcase")})
    paths = [ROOT / (name.replace(".", "/") + ".py") for name in names]
    assert len(paths) == 14 and all(path.is_file() for path in paths)
    return paths


def run_tests(source, destination):
    assert not destination.exists()
    sys.path.insert(0, str(source))
    import asa_arknight_story_agent.inference.grounding.quote_validation as module
    assert Path(module.__file__).resolve() == (source / RELATIVE).resolve()
    import pytest
    tests = regression_files() + [
        HERE / "test_empty_normalized_quote.py", HERE / "test_empty_quote_followup_v1.py",
    ]
    code = pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", *(str(path) for path in tests),
                        "--junitxml=" + str(destination)])
    origins = {}
    for name, imported in sys.modules.copy().items():
        if name == "asa_arknight_story_agent" or name.startswith("asa_arknight_story_agent."):
            filename = getattr(imported, "__file__", None)
            if filename:
                actual = Path(filename).resolve()
                assert actual.is_relative_to(source.resolve()), f"Wrong source loaded: {name}: {actual}"
                origins[name] = str(actual)
    assert code == 0
    assert counts(destination) == {"tests": 312, "failures": 0, "errors": 0, "skipped": 0}
    return origins


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "canonical_tests":
        origins = run_tests(SOURCE, HERE / "empty_quote_canonical_tests312_v1.xml")
        save(HERE / "empty_quote_canonical_test_origins_v1.json", origins)
        return
    output = HERE / "empty_quote_candidate_verified_v1.json"
    assert not output.exists()
    before, candidate = snapshot(SOURCE), snapshot(CANDIDATE)
    assert set(before) == set(candidate)
    assert [key for key in before if before[key] != candidate[key]] == [RELATIVE]
    assert sha(HERE / "quote_validation.py") == candidate[RELATIVE]
    assert counts(HERE / "empty_quote_baseline_v1.xml") == {
        "tests": 34, "failures": 30, "errors": 0, "skipped": 0,
    }
    assert counts(HERE / "empty_quote_candidate_v1.xml") == {
        "tests": 34, "failures": 0, "errors": 0, "skipped": 0,
    }
    assert counts(OLD_XML) == {"tests": 276, "failures": 0, "errors": 0, "skipped": 0}
    origins = run_tests(CANDIDATE, HERE / "empty_quote_candidate_tests312_v1.xml")
    replay = HERE / "empty_quote_fixed79_replay_v1.json"
    assert not replay.exists()
    matrix = RUN / "fixed79.current_quote.matrix.json"
    predictions = RUN / "fixed79_configured_lora_ctx12288/predictions.json"
    old = RUN / "semantic79_content_v2.fPkkhM/canonical_fixed79_validation_v1.json"
    frozen_sha = {str(p): sha(p) for p in (matrix, predictions, old)}
    subprocess.run([
        sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
        "--validator", "after", "--source-root", str(CANDIDATE),
        "--matrix", str(matrix), "--predictions", str(predictions), "--output", str(replay),
    ], check=True)
    baseline, result = load(old), load(replay)
    assert len(baseline["results"]) == len(result["results"]) == 79
    assert [r["id"] for r in result["results"]] == [r["id"] for r in baseline["results"]]
    changes = [
        b["id"] for a, b in zip(baseline["results"], result["results"])
        if a.get("grounded") != b.get("grounded") or a["status"] != b["status"]
    ]
    assert not changes, "Unexpected historical replay changes need source review"
    assert snapshot(SOURCE) == before
    assert all(sha(Path(path)) == value for path, value in frozen_sha.items())
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "changed_source_path": RELATIVE,
        "canonical_source_hashes": before, "candidate_source_hashes": candidate,
        "candidate_tests": counts(HERE / "empty_quote_candidate_tests312_v1.xml"),
        "baseline_new_tests": counts(HERE / "empty_quote_baseline_v1.xml"),
        "new_tests_sha256": {
            name: sha(HERE / name) for name in (
                "test_empty_normalized_quote.py", "test_empty_quote_followup_v1.py",
            )
        },
        "test_module_origins": origins, "frozen_inputs_sha256": frozen_sha,
        "replay_sha256": sha(replay), "all79_grounded_results_unchanged": True,
        "actions": dict(Counter(r["grounded"]["next_action"] for r in result["results"])),
        "limits": [
            "Fixes the empty-string substring bypass; this is not semantic entailment verification.",
            "Nonempty punctuation, wrong actors, unsupported causality and quote insufficiency are not solved here.",
            "34 new quote tests plus two diagnostic propagation tests;276 prior regressions are reused, not new tests.",
            "The canonical source and in-flight generation experiment remained unchanged.",
            "cp -a reported unsupported permission preservation; diff -qr verified copied bytes before the candidate overlay.",
        ],
    })
    print(json.dumps({"candidate_tests": 312, "replay_states": 79, "changed": len(changes)}))


if __name__ == "__main__":
    main()
