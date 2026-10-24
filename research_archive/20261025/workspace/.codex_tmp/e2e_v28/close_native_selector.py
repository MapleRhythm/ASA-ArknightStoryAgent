"""Bind source-review annotations without editing any original generation or label."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reader_evidence_router_20260909.v27"
sys.path.insert(0, str(PREVIOUS))
from selector_protocol import CASE_IDS, blocks


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_pre = load(PREVIOUS / "preflight.json")
    old_gen = load(PREVIOUS / "generations.json")
    pre = load(HERE / "preflight.json")
    gen = load(HERE / "generations.json")
    post = load(HERE / "postflight_v1.json")
    review = load(HERE / "source_review_v28.json")
    assert [r["id"] for r in review["reviews"]] == CASE_IDS
    assert post["calls"] == gen["actual_call_count"] == 23
    assert post["generations_sha256"] == sha(HERE / "generations.json")
    assert post["preflight_sha256"] == sha(HERE / "preflight.json")
    assert pre["runtime"] == old_pre["runtime"]
    assert pre["base_model"] == old_pre["base_model"]
    assert pre["adapter"] == old_pre["adapter"]
    assert pre["tokenizer"] == old_pre["tokenizer"]
    comparisons, excerpt_count = [], 0
    for a, b, x, y, annotation in zip(
        old_pre["rows"], pre["rows"], old_gen["rows"], gen["rows"], review["reviews"], strict=True,
    ):
        assert a["id"] == b["id"] == x["id"] == y["id"] == annotation["id"]
        for key in ("control_prompt", "evidence_brief", "hypothesis", "sampling_seed", "reader_order"):
            assert a[key] == b[key]
        assert b["selector_prompt"] == a["selector_prompt"][:-1] + "<think>\n\n</think>\n\n{"
        sources = dict(blocks(b["evidence_brief"]))
        for check in annotation.get("evidence_checks", []):
            assert check["quote"] in sources[check["evidence_id"]], (b["id"], check)
            excerpt_count += 1
        comparisons.append({
            "id": b["id"], "control_prompt_and_seed_identical": True,
            "control_raw_identical": x["control"]["raw_output"] == y["control"]["raw_output"],
            "control_display_identical": x["control"]["grounded"]["answer"] == y["control"]["grounded"]["answer"],
            "control_action_identical": x["control"]["grounded"]["next_action"] == y["control"]["grounded"]["next_action"],
            "selected_ids": y["selector"].get("payload", {}).get("evidence_ids"),
            "routed_status": y["routed"]["status"],
        })
    report = {
        "version": "native_selector_closed_v28",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "promotion_decision": "reject_this_candidate",
        "basis_sha256": {
            str(p): sha(p) for p in (
                PREVIOUS / "preflight.json", PREVIOUS / "generations.json",
                HERE / "preflight.json", HERE / "generations.json",
                HERE / "postflight_v1.json", HERE / "source_review_v28.json", Path(__file__),
            )
        },
        "mechanical_summary": post["summary"],
        "calls": 23, "finish_reasons": post["finish_reasons"],
        "source_excerpt_checks_passed": excerpt_count,
        "comparisons": comparisons,
        "cross_run_controls": {
            "raw_identical": sum(r["control_raw_identical"] for r in comparisons),
            "display_identical": sum(r["control_display_identical"] for r in comparisons),
            "action_identical": sum(r["control_action_identical"] for r in comparisons),
            "total": 8,
        },
        "review_categories_not_certified_accuracy": {
            kind: Counter(row[kind] for row in review["reviews"]) for kind in ("control", "routed")
        },
        "production_changed": False,
        "limitations": review["limitations"],
    }
    with (HERE / "checkpoint_v28.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "calls", "source_excerpt_checks_passed", "cross_run_controls", "promotion_decision",
    )}), flush=True)


if __name__ == "__main__":
    main()
