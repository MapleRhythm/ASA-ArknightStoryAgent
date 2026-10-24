"""Verify a paired coverage audit and record separate source-review findings."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_reconstruction_coverage39_v3 import prepare
import probe_answer_slots_v1 as owner
import probe_answer_slots_v3 as rubric


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    path = HERE / "reconstruction_coverage39_v3.json"
    report = load(path)
    destination = HERE / "reconstruction_coverage39_postflight_v3.json"
    assert not destination.exists()
    source = HERE / "canonical_fixed79_validation_v1.json"
    assert sha(source) == report["source_sha256"]
    assert sha(HERE / "audit_reconstruction_coverage39_v3.py") == report["driver_sha256"]
    assert sha(HERE / "probe_answer_slots_v3.py") == report["rubric_sha256"]
    for name, digest in report["rubric_dependency_hashes"].items():
        assert sha(HERE / name) == digest
    assert sha(HERE.parent / "probe_claim_entailment_v1.py") == report["transport_sha256"]
    assert sha(HERE / "reconstruction_coverage39_preflight_v3.json") == report["preflight_sha256"]
    rows, cases, pairs = prepare(load(source))
    assert cases == report["cases"] and pairs == report["pairs"]
    assert len(rows) == 39 and len(cases) == 72
    results = {r["id"]: r for r in report["results"]}
    assert len(results) == 72 and set(results) == {c["id"] for c in cases}
    for case in cases:
        result = results[case["id"]]
        expected = {"model": "glm-5.3", "temperature": 0, "max_tokens": 16384, "messages": rubric.messages(case)}
        assert result["request_without_credentials"] == expected
        assert result["request_sha256"] == owner.transport.digest(expected)
        assert result["response_model"] == "glm-5.3" and result["finish_reason"] == "stop"
        assert set(json.loads(expected["messages"][1]["content"])) == {"question", "shown_answer"}
        if result["status"] == "ok":
            assert rubric.validate(json.loads(result["raw_response"]), case) == result["judgement"]
        else:
            try:
                rubric.validate(json.loads(result["raw_response"]), case)
            except (ValueError, TypeError) as exc:
                assert result["error"] == f"{type(exc).__name__}: {exc}"
            else:
                raise AssertionError("Saved failure did not reproduce")
    assert Counter(r["status"] for r in results.values()) == {"ok": 67, "error": 5}
    expected_pairs = []
    for pair in pairs:
        item = dict(pair)
        for side in ("parsed", "shown"):
            result = results[pair[side + "_group_id"]]
            item[side + "_status"] = result["status"]
            item[side + "_coverage"] = (result.get("judgement") or {}).get("coverage", "audit_error")
        if not pair["changed"]:
            assert item["parsed_group_id"] == item["shown_group_id"]
        expected_pairs.append(item)
    assert expected_pairs == report["paired_results"]
    both_valid = [p for p in expected_pairs if p["parsed_status"] == p["shown_status"] == "ok"]
    assert len(both_valid) == 34
    transitions = Counter(p["parsed_coverage"] + "->" + p["shown_coverage"] for p in both_valid)
    assert transitions == {"complete->complete": 30, "none->none": 2, "complete->none": 1, "partial->partial": 1}
    source_rows = {r["id"]: r for r in rows}
    planet_id = next(rid for rid in source_rows if rid.startswith("b5c3bd1132dc9df8__"))
    planet = source_rows[planet_id]
    anchors = [
        "那些“石块”被称作“行星碎屑”",
        "这么巨大的“行星碎屑”静止时能够堵住前路",
        "站在相邻位置能够攻击“行星碎屑”，让其充能",
        "一定次数后，能够将“行星碎屑”朝干员部署方向推动",
    ]
    assert all(text in planet["evidence_brief"] for text in anchors)
    assert "充能" in planet["parsed"]["answer"] and "充能" not in planet["grounded"]["answer"]
    meisia = next(r for r in rows if r["id"].startswith("2a0c1b2ca6d3bc0d__"))
    yu = next(r for r in rows if r["id"].startswith("relabel-1bf082ec94d1c0d2d169abad::"))
    assert "有些事我还未求证，但恐怕" in meisia["evidence_brief"]
    assert "走私暴露当晚" in meisia["evidence_brief"] and "您想让他们帮您把违禁品运出去" in meisia["evidence_brief"]
    assert "委托当时尚不在朝局中央的我，暗中寻访" in yu["evidence_brief"]
    assert "这件事太冒险，这世上不能有第三个人知道" in yu["evidence_brief"]
    assert "我们还需要有一个清清白白的人" in yu["evidence_brief"]
    source_review = [
        {
            "parent_id": planet_id, "finding": "Source-supported actuation information is lost from displayed answer.",
            "source_anchors": anchors,
            "notes": [
                "Complete model-visible E1-E8 were read, not only the quoted E2/E3 facts.",
                "The raw/parsed charging and deployment-direction information has visible support and disappears in the fact-based display.",
                "The coverage judge's complete->none labels are preserved as automatic opinions; source review certifies the loss of these details, not a human-gold complete answer or a general rewrite bypass.",
            ],
        },
        {
            "parent_id": meisia["id"], "finding": "Do not call every change in certainty harmful.",
            "notes": [
                "All E1-E6 read. E1 distinguishes the earlier unverified suspicion about schemes from later smuggling/contraband conversations; E6 includes Gordon's admission about exposed smuggling and imported Originium devices.",
                "The parsed answer treats the scheme as discovered and smuggling as suspected; displayed facts put suspicion on the scheme and discovery on smuggling. The full context supports that distinction.",
                "This is evidence against a blind policy of restoring every parsed answer or preserving uncertainty word counts across the whole answer.",
            ],
        },
        {
            "parent_id": yu["id"], "finding": "Separate historical covert search from later public truth-seeking.",
            "notes": [
                "All E1-E6 read. E1 supports entrusting Xie Zhen with covert search; E2 supports a separate role for Lin Qingyan through above-board means.",
                "Do not label the added Lin role wholly invented merely because the parsed sentence left it implicit.",
                "Both candidates compress the separate secrecy reason for not telling Lin directly. Automatic complete coverage does not settle this interpretive completeness issue.",
            ],
        },
    ]
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True, "training_ready": False,
        "protocol_verified": True, "all72_requests_verified": True, "old_source_unchanged": True,
        "all_failures_reproduced_without_repair": True, "all39_answer_pairs_read": True,
        "report_sha256": sha(path), "source_sha256": sha(source),
        "summary": report["summary"], "both_valid_pairs": 34, "both_valid_transitions": dict(transitions),
        "source_reviewer": "independent_assistant_not_human_gold", "source_review": source_review,
        "limitations": [
            "Response model fields match requested glm-5.3; this is not independent attestation of the provider's backend weights.",
            "Five failed inputs are still failed; one malformed response contains a JavaScript-like expression, which was never executed.",
            "Coverage equality does not establish semantic equality or source attribution; no factual accuracy from these counts.",
            "The 39 direct-state subset excludes 40 other parent states without removing them from the parent79 denominator.",
            "No model, production validator, old answer, training label, or original judgement has been overwritten.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"postflight": str(destination), "unique_statuses": report["summary"]["unique_statuses"], "both_valid_transitions": dict(transitions)}))


if __name__ == "__main__":
    main()
