"""Append source-refreshed preflight/attempt2; preserve failed busy attempt1."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import run_quote_pointer8b_v1 as pointer
from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot

OLD_PREFLIGHT = HERE / "quote_pointer8b58_execution_preflight_v1.json"
OLD_RECEIPT = HERE / "quote_pointer8b58_launch_receipt_v1.json"
NEW_PREFLIGHT = HERE / "quote_pointer8b58_execution_preflight_v2.json"
NEW_LOG = HERE / "quote_pointer8b58_generation_v2.log"
NEW_RECEIPT = HERE / "quote_pointer8b58_launch_receipt_v2.json"


def configure():
    dependency_paths = tuple(pointer.dependencies())
    def dependencies():
        return {**{p: sha(Path(p)) for p in dependency_paths}, str(Path(__file__)): sha(Path(__file__))}
    pointer.dependencies = dependencies
    pointer.PREFLIGHT, pointer.LOG, pointer.RECEIPT = NEW_PREFLIGHT, NEW_LOG, NEW_RECEIPT
    pointer.configure_base()


def prepare():
    assert not NEW_PREFLIGHT.exists() and not pointer.OUTPUT.exists()
    old = load(OLD_PREFLIGHT)
    failed = load(OLD_RECEIPT)
    assert failed["launched"] is False and failed["reason"] == "no_sustained_idle_window"
    assert all(not r["idle"] for r in failed["idle_observations"])
    applied = load(HERE / "chain_score_binding_applied_verified_v1.json")
    preapply = load(HERE / "chain_score_binding_preapply_v1.json")
    assert applied["preapply_sha256"] == sha(HERE / "chain_score_binding_preapply_v1.json")
    assert old["source_hashes"] == preapply["canonical_source_hashes"]
    current = snapshot(SOURCE)
    assert current == preapply["candidate_source_hashes"]
    assert [p for p in current if current[p] != old["source_hashes"][p]] == [
        "asa_arknight_story_agent/retrieval/hybrid_components/hybrid_chain_scoring.py"]
    assert all(sha(Path(p)) == h for p, h in old["dependencies_sha256"].items())
    assert all(sha(Path(p)) == h for p, h in old["source_artifacts_sha256"].items())
    assert sha(pointer.MATRIX) == old["matrix_sha256"]
    assert sha(pointer.base.CONFIG) == old["canonical_config_sha256"]
    assert pointer.base.assets() == old["model_assets"]
    fresh = deepcopy(old)
    fresh.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(), source_hashes=current,
        dependencies_sha256=pointer.dependencies(), previous_preflight_sha256=sha(OLD_PREFLIGHT),
        previous_no_launch_receipt_sha256=sha(OLD_RECEIPT),
        source_refresh_reason="Verified chain-score/text binding fix only; generator/evidence matrix unchanged.",
    )
    fresh["source_artifacts_sha256"].update({str(p): sha(p) for p in [
        HERE / "chain_score_binding_preapply_v1.json", HERE / "chain_score_binding_applied_verified_v1.json",
        OLD_PREFLIGHT, OLD_RECEIPT,
    ]})
    save(NEW_PREFLIGHT, fresh)
    pointer.ensure_frozen(fresh)
    print(json.dumps({"refreshed_rows": 58, "new_generations": 0, "previous_attempt_preserved": True}), flush=True)


def launch():
    # pointer.launch's child target is its original __file__. To make the
    # child use this explicit source-refreshed wrapper, replace only its local
    # module __file__; dependencies still include the frozen original script.
    original_file = pointer.__file__
    pointer.__file__ = str(Path(__file__))
    try:
        return pointer.launch()
    finally:
        pointer.__file__ = original_file


if __name__ == "__main__":
    assert len(sys.argv) == 2 and sys.argv[1] in {"prepare", "launch", "run", "analyze", "check"}
    configure()
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "launch":
        raise SystemExit(launch())
    elif sys.argv[1] == "run":
        pointer.ensure_frozen(load(NEW_PREFLIGHT))
        pointer.base.run()
        pointer.ensure_frozen(load(NEW_PREFLIGHT))
    elif sys.argv[1] == "analyze":
        import analyze_quote_pointer8b_v1 as analyzer
        analyzer.PREFLIGHT = NEW_PREFLIGHT
        # configure_base is idempotent for the same process-local paths.
        analyzer.main()
    else:
        import check_quote_pointer8b_v1 as checker
        checker.PREFLIGHT, checker.RECEIPT = NEW_PREFLIGHT, NEW_RECEIPT
        checker.main()
