from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    diagnosis = json.loads((HERE / "diagnosis_v1.json").read_text())
    trace = json.loads((HERE / "trace_provenance_v1.json").read_text())
    assert diagnosis["production_changed"] is trace["production_changed"] is False
    assert diagnosis["gpu_used"] is trace["gpu_used"] is False
    assert trace["questions"] == 58 and trace["trace_steps"] == 92
    assert len(diagnosis["observations"]) == 8
    assert diagnosis["isolated_synthetic_normalization"]["original_model_payload_not_claimed"]
    assert all(sha(Path(p)) == h for p, h in diagnosis["source_sha256"].items())
    assert all(sha(Path(p)) == h for p, h in trace["basis_sha256"].items())
    assert sha(Path(trace["results_path"])) == trace["results_sha256"]
    files = [HERE / name for name in (
        "diagnosis_v1.json", "trace_provenance_v1.json",
        "diagnose_entity_fragments.py", "audit_trace_provenance.py", "close_diagnosis.py",
    )]
    with (HERE / "checkpoint_v36.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "version": "entity_fragment_diagnosis_closed_v36",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "production_changed": False, "gpu_used": False, "training_ready": False,
            "status": "diagnosis_archived_not_fixed",
            "fragment_reproduction_questions": 8,
            "saved_trace_questions": 58, "saved_trace_steps": 92,
            "raw_planner_not_reconstructed": True,
            "source_and_saved_results_unchanged": True,
            "basis_sha256": {str(p): sha(p) for p in files},
        }, stream, ensure_ascii=False, indent=2)
    print({"status": "diagnosis_archived_not_fixed", "trace_questions": 58, "trace_steps": 92})


if __name__ == "__main__":
    main()
