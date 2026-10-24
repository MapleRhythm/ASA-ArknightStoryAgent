from datetime import datetime, timezone
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reader_evidence_router_20260909.v27"))
import run_reader_router as shared
from selector_protocol import CASE_IDS, blocks


def main():
    pre = shared.load(HERE / "preflight.json")
    review = shared.load(HERE / "source_review_v30.json")
    generated = {key: shared.load(HERE / key / "generations.json") for key in pre["models"]}
    posts = {key: shared.load(HERE / key / "postflight_v1.json") for key in pre["models"]}
    assert [row["id"] for row in review["reviews"]] == CASE_IDS
    sources = {row["id"]: row for row in pre["rows"]}
    outputs = {key: {row["id"]: row for row in value["rows"]} for key, value in generated.items()}
    for key, post in posts.items():
        assert post["calls"] == 16
        assert post["preflight_sha256"] == shared.sha(HERE / "preflight.json")
        assert post["generations_sha256"] == shared.sha(HERE / key / "generations.json")
    excerpt_count = output_count = 0
    for annotation in review["reviews"]:
        source = sources[annotation["id"]]
        evidence = dict(blocks(source["evidence_brief"]))
        for check in annotation.get("evidence_checks", []):
            assert check["quote"] in evidence[check["evidence_id"]], (annotation["id"], check)
            excerpt_count += 1
        for span in annotation.get("absent_source_spans", []):
            assert span not in source["evidence_brief"]
        for check in annotation.get("output_checks", []):
            item = outputs[check["model"]][annotation["id"]]["conditions"][check["protocol"]]
            answer = item["grounded"]["answer"] if check["protocol"] == "quote" else item["display_answer"]
            assert check["span"] in answer, (annotation["id"], check)
            output_count += 1
    files = [HERE / "preflight.json", HERE / "source_review_v30.json", Path(__file__),
             HERE / "official_download_manifest.json", HERE / "metadata_download_receipt.json",
             HERE / "weights_download_receipt.json",
             *[HERE / key / file for key in pre["models"]
               for file in ("generations.json", "postflight_v1.json")]]
    report = {
        "version": "native_qwen35_capacity_closed_v30",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "promotion_decision": review["promotion_decision"],
        "source_excerpt_checks_passed": excerpt_count,
        "actual_display_span_checks_passed": output_count,
        "calls": 32, "unique_questions": 8,
        "mechanical_summaries": {key: value["summary"] for key, value in posts.items()},
        "basis_sha256": {str(path): shared.sha(path) for path in files},
        "production_changed": False,
        "limitations": review["limitations"],
    }
    shared.save(HERE / "checkpoint_v30.json", report)
    print(json.dumps({"calls": 32, "source_checks": excerpt_count, "display_checks": output_count,
                      "promotion_decision": report["promotion_decision"]}), flush=True)


if __name__ == "__main__":
    main()
