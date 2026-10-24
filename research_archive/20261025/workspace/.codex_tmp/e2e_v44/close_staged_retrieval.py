"""Close the retrieval/state protocol only; no answer-quality certification."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import run_second_round_retrieval as staged

HERE = Path(__file__).resolve().parent
shared = staged.shared


def main():
    assert not (HERE / "checkpoint_v44.json").exists()
    pre = shared.load(HERE / "preflight.json")
    result = shared.load(HERE / "retrieval.json")
    post = shared.load(HERE / "postflight_v1.json")
    staged.reader.verify(pre)
    assert shared.sha(HERE / "preflight.json") == result["preflight_sha256"] == post["preflight_sha256"]
    assert shared.sha(HERE / "retrieval.json") == post["retrieval_sha256"]
    assert post["first_round_state_replays"] == 24
    assert post["status_counts_not_accuracy"] == {
        "terminal_carried_unchanged": 10, "round2_retrieval_ok": 14,
    }
    assert result["new_llm_calls"] == post["new_llm_calls"] == 0
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    released = shared.gpu_idle()
    query_counts = []
    for item, saved in zip(pre["records"], result["records"], strict=True):
        assert (item["pair_id"], item["condition"]) == (saved["pair_id"], saved["condition"])
        query_counts.append({
            "pair_id": item["pair_id"], "condition": item["condition"],
            "terminal_carried": item["terminal_after_round1"],
            "round2_queries": len(item["round2_queries"]),
            "actual_search_calls": len(saved["actual_search_calls"]),
            "planned_visible_chars": len(saved.get("visible_evidence", "")),
        })
    log = (HERE / "run_v1.log").read_text()
    files = [HERE / name for name in (
        "preflight.json", "prepare.log", "run_v1.log", "check_v1.log",
        "idle_samples.json", "retrieval.json", "retrieval.progress.jsonl",
        "postflight_v1.json", "run_second_round_retrieval.py", "close_staged_retrieval.py",
    )]
    shared.save(HERE / "checkpoint_v44.json", {
        "version": "staged_round2_retrieval_closed_v44",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status_counts_not_accuracy": post["status_counts_not_accuracy"],
        "first_round_state_replays": 24, "terminal_answers_carried_unchanged": 10,
        "completed_round2_retrievals": 14, "new_llm_calls": 0,
        "query_and_prompt_protocol_checked": True, "query_counts": query_counts,
        "gpu2_released": released,
        "gpu_processes_at_close": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader",
        ], text=True),
        "log_error_lines": [line for line in log.splitlines() if "ERROR" in line or "Traceback" in line],
        "basis_sha256": {str(path): shared.sha(path) for path in files},
        "production_changed": False, "training_ready": False,
        "assistant_visible_source_review_complete": False,
        "promotion_decision": "no_promotion_round2_reader_pending",
        "limitations": [
            "Staged continuation of24 existing v41 attempts, not24 independent questions or fresh E2E.",
            "Retrieval/state/prompt replay checks do not establish answer correctness or semantic source support.",
            "Ten first-round terminal answers, including known errors, remain unchanged.",
            "No new LLM outputs in this stage; no final two-round comparison is available yet.",
        ],
    })
    print({"statuses": Counter(r["status"] for r in result["records"]), "new_llm_calls": 0,
           "gpu2_released": True, "second_round_reader_pending": True}, flush=True)


if __name__ == "__main__":
    main()
