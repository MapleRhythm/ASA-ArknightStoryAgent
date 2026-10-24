"""Bind the human-readable checkpoint to actual tests, mirror receipts and progress."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import counts, save, snapshot

REPORT = ROOT / "docs/CANONICAL_FACT_COVERAGE_AND_GROUNDING_GUARDS_20260909_leU0LZ_v17.md"
TOKEN = "asa_arknight_story_agent/inference/grounding/token_checks.py"


def main():
    output = HERE / "checkpoint_v17_state.json"
    assert not output.exists() and REPORT.is_file()
    canonical = load(HERE / "token_rejection_applied_verified_v1.json")
    preapply = load(HERE / "token_rejection_preapply_v1.json")
    assert sha(ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json") == preapply["config_sha256"]
    assert canonical["canonical_tests"] == {"tests": 325, "failures": 0, "errors": 0, "skipped": 0}
    xml = HERE / "token_rejection_canonical_tests325_v1.xml"
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert len(cases) == len({(c.attrib["classname"], c.attrib["name"]) for c in cases}) == 325
    assert counts(xml) == canonical["canonical_tests"]
    mirror = HERE / "mirror248_token_delta"
    received = load(mirror / "mirror_applied_verified_v1.json")
    assert received["canonical_applied_receipt_sha256"] == sha(HERE / "token_rejection_applied_verified_v1.json")
    assert received["applied_sha256"] == canonical["applied_sha256"]
    assert received["postapply_test"]["counts"] == {"tests": 169, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(mirror / "mirror_tests169_v1.xml") == received["postapply_test"]["counts"]
    mirror_before = load(mirror / "mirror_preapply_v1.json")["mirror_before"]
    mirror_before["runtime"][TOKEN] = canonical["applied_sha256"]["release/ASA-ArknightStoryAgent/src/" + TOKEN]
    assert mirror_before["runtime"] == snapshot(SOURCE)
    assert mirror_before["dataset_scripts"] == preapply["dataset_scripts_sha256"]
    for path, digest in canonical["applied_sha256"].items():
        assert sha(ROOT / path) == digest
    native = load(HERE / "qwen3_8b_native_preflight_v1.json")
    assert native["all58_inputs_fit"] and native["cpu_only"] and native["new_generations"] == 0
    prefix = (HERE / "blind_audits_v1.progress.jsonl").read_bytes()
    prefix = prefix[:prefix.rfind(b"\n") + 1]
    progress = [json.loads(line) for line in prefix.splitlines()]
    audit_preflight = load(HERE / "blind_audits_preflight_v1.json")
    assert audit_preflight["unique_provider_inputs"] == 126
    assert len(progress) == len({r["id"] for r in progress}) <= 126
    assert {r["id"] for r in progress} <= {r["id"] for r in audit_preflight["cases"]}
    post = load(HERE / "generation_postflight_v1.json")
    assert post["protocol_verified"] and post["states"] == 116
    assert post["finish_reasons"] == {"stop": 116}
    artifacts = [
        HERE / name for name in (
            "fact_coverage_pairs116_v1.matrix.json", "generation_postflight_v1.json",
            "validation_v1.json", "blind_audits_preflight_v1.json",
            "empty_quote_applied_verified_v1.json", "token_rejection_applied_verified_v1.json",
            "partial_source_review_v1.json", "qwen3_8b_native_preflight_v1.json",
            "qwen3_8b_native58_v1.matrix.json",
        )
    ] + [mirror / "mirror_applied_verified_v1.json", xml]
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": False, "training_ready": False, "report": str(REPORT),
        "report_sha256": sha(REPORT), "canonical_unique_tests": 325,
        "mirror_cpu_tests": 169, "mirrored_runtime_python_hashes_match_canonical": True,
        "mechanical_summary": post["mechanical_summary"],
        "audit_finished_at_checkpoint": (HERE / "blind_audits_v1.json").exists(),
        "audit_progress_complete_lines": len(progress),
        "audit_progress_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "audit_progress_statuses": dict(Counter(row["status"] for row in progress)),
        "artifacts_sha256": {str(p): sha(p) for p in artifacts},
        "no_new_certified_answer_accuracy_or_retrieval_recall": True,
        "native8b_new_generations": 0,
    })
    print(json.dumps({
        "report": str(REPORT), "canonical_tests": 325, "mirror_tests": 169,
        "audit_progress": len(progress), "runtime_hashes_match": True,
        "goal_complete": False,
    }), flush=True)


if __name__ == "__main__":
    main()
