"""Verify and bind completed CPU retrieval diagnostics plus canonical repair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot, counts


def main():
    report = ROOT / "docs/CANONICAL_CHAIN_BINDING_AND_RETRIEVAL_FAILURE_20260909_leU0LZ_v20.md"
    destination = HERE / "checkpoint_v20_state.json"
    reviewpath = HERE / "susie_source_retrieval_review_v1.json"
    assert report.is_file() and not destination.exists() and not reviewpath.exists()
    applied = load(HERE / "chain_score_binding_applied_verified_v1.json")
    preapply = load(HERE / "chain_score_binding_preapply_v1.json")
    assert applied["preapply_sha256"] == sha(HERE / "chain_score_binding_preapply_v1.json")
    assert applied["canonical_tests"] == {"tests": 362, "failures": 0, "errors": 0, "skipped": 0}
    assert snapshot(SOURCE) == preapply["candidate_source_hashes"]
    assert all(sha(ROOT / p) == h for p, h in applied["applied_sha256"].items())
    mirror = load(HERE / "mirror248_chain_score_binding/mirror_applied_verified_v1.json")
    assert mirror["full_runtime_python_matches241"] and mirror["four_dataset_scripts_match241"]
    assert mirror["postapply_test"]["counts"] == {"tests": 206, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(HERE / "quote_pointer_combined_tests_v1.xml")["tests"] == 44
    assert counts(HERE / "quote_pointer_feasibility_tests_v1.xml") == {
        "tests": 7, "failures": 0, "errors": 0, "skipped": 0}
    assert counts(HERE / "quote_pointer_refresh_tests_v1.xml") == {
        "tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    sparse = load(HERE / "canonical_sparse58_probe_v1.json")
    assert sparse["documents"] == 54252 and sparse["queries"] == len(sparse["results"]) == 58
    assert sparse["target_joint_candidates"] == 43 and sparse["target_joint_sparse_top120"] == 40
    assert all(sha(Path(p)) == h for p, h in sparse["source_artifacts_sha256"].items())
    controls = load(HERE / "susie_reranker120_v1.json")
    candidate = load(HERE / "susie_v7stage2_same120_v1.json")
    cp = load(HERE / "susie_reranker120_preflight_v1.json")
    np = load(HERE / "susie_v7stage2_same120_preflight_v1.json")
    assert controls["preflight_sha256"] == sha(HERE / "susie_reranker120_preflight_v1.json")
    assert candidate["preflight_sha256"] == sha(HERE / "susie_v7stage2_same120_preflight_v1.json")
    assert cp["candidates"] == np["candidates"] and controls["all120_inputs_untruncated"] and candidate["all120_inputs_untruncated"]
    assert [r["actual_token_ids_sha256"] for r in controls["actual_inputs"]] == candidate["actual_token_ids_sha256"]
    target = "activities/act10mini/level_act10mini_st05#chunk-0025"
    old_target = next(r for r in controls["ranked"] if r["document_id"] == target)
    new_target = next(r for r in candidate["ranked"] if r["document_id"] == target)
    assert old_target["sparse_rank"] == 16 and old_target["rerank_position"] == 15 and new_target["position"] == 37
    for p in (cp, np):
        assert all(sha(Path(path)) == h for path, h in p["model_assets_sha256"].items())
    chains = load(HERE / "susie_chain_inputs_v1.json")
    assert chains["chain_count"] == chains["would_truncate_count"] == 2
    assert [c["full_input_tokens"] for c in chains["chains"]] == [2365, 2520]
    assert not any(c["critical_rescue_phrase_present"] for c in chains["chains"])
    scores = load(HERE / "susie_chain_complete_pair4_v1.json")
    assert len(scores["results"]) == 4 and scores["source_and_model_unchanged"]
    assert sum(r["input_truncated"] for r in scores["results"]) == 2
    storypath = HERE / "susie_rescue_canonical_story_v1.json"
    story = load(storypath)
    docs = {r["id"]: r for r in story["documents"]}
    assert "她依然没有放手" in docs[target]["clean_text"]
    prefix = "activities/act10mini/level_act10mini_st05#chunk-"
    reviewed_ids = [prefix + f"{i:04d}" for i in range(13, 27)]
    assert all(i in docs for i in reviewed_ids)
    save(reviewpath, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "reviewer": "independent_assistant_not_human_gold", "full_corpus_reviewed": False,
        "reviewed_story_chunk_ids": reviewed_ids,
        "reviewed_text_sha256": {i: digest(docs[i]["clean_text"]) for i in reviewed_ids},
        "story_artifact_sha256": sha(storypath),
        "findings": [
            "0014-0015 locate Susie in underground channels and divide the search with Skyfire.",
            "0019 shows Quercus using seeds/Originium Arts/vines to bind attackers and enter.",
            "0020-0022 separately show the explosion, Reid protecting Susie and throwing her over the broken bridge; do not attribute these all to Quercus.",
            "0025-0026 show vines/roots opening wreckage, Quercus embracing and calming Susie despite burns, then taking her back.",
            "These exact target-event passages exist in canonical corpus and the sparse120 pool; prior frozen E1-E10 is another episode.",
            "Current andv7 document scoring prefer wrong-event sources; the fixed two-chain construction also omits final rescue chunk0025.",
            "Neither keyword cooccurrence, score nor this selected assistant review is human gold or overall recall/accuracy.",
        ],
        "sent_to_generation_or_auditors": False, "old_labels_changed": False,
    })
    assert not (HERE / "quote_pointer8b58_ctx12288_v1/predictions.json").exists(), "report pending state needs a new version"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == "0d594a3c9fd641fd8d7f798c8ca6b07db6c7020b"
    names = [
        "checkpoint_v19_state.json", "chain_score_binding_applied_verified_v1.json",
        "chain_score_binding_candidate_verified_v1.json",
        "mirror248_chain_score_binding/mirror_applied_verified_v1.json",
        "canonical_sparse58_probe_v1.json", "susie_reranker120_v1.json",
        "susie_v7stage2_same120_v1.json", "susie_chain_inputs_v1.json", "susie_chain_complete_pair4_v1.json",
        "susie_source_retrieval_review_v1.json", "quote_pointer_feasibility_v1.json",
        "without_reasoning_hints58_partial_source_review_v2.json",
        "quote_pointer8b58_execution_preflight_v2.json", "quote_pointer8b58_launch_receipt_v1.json",
    ]
    save(destination, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "goal_complete": False, "training_ready": False, "head": head,
        "report": str(report), "report_sha256": sha(report), "runtime_source_sha256": snapshot(SOURCE),
        "canonical_tests": applied["canonical_tests"], "mirror_subset_tests": mirror["postapply_test"]["counts"],
        "new_sparse_queries": 58, "new_document_scores": 240, "new_chain_scores": 4,
        "selected_rescue_segment_ranks": {"sparse": 16, "current_reranker": 15, "candidate_v7": 37},
        "chain_complete_comparison": scores["results"],
        "pointer58_generation_complete": False, "pending_pointer_preflight_version": 2,
        "artifacts_sha256": {str(HERE / name): sha(HERE / name) for name in names},
        "checkpoint_script_sha256": sha(Path(__file__)),
        "default_model_or_window_changed": False, "old_labels_changed": False,
        "new_training_runs": 0, "services_restarted": False,
        "new_full_rag_recall": None, "new_certified_answer_accuracy": None,
    })
    print(json.dumps({"report": str(report), "canonical_tests": 362,
                      "mirror_subset_tests": 206, "goal_complete": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
