"""Read-only human-inspection view of automatic regressions and known examples."""
import json
import sys

from build_fact_coverage_pairs_v1 import HERE, load


def main():
    audited = load(HERE / "blind_audits_v1.json")
    post = load(HERE / "blind_audits_postflight_v1.json")
    rows = {r["id"]: r for r in load(HERE / "validation_v1.json")["results"]}
    states = {r["id"]: r for r in audited["states"]}
    judgements = {r["id"]: r for r in audited["results"]}
    if "--errors-only" in sys.argv:
        cases = {r["id"]: r for r in audited["cases"]}
        for row in audited["results"]:
            if row["status"] != "ok":
                print(json.dumps({
                    "id": row["id"], "question": cases[row["id"]]["question"],
                    "error": row["error"], "response_model": row.get("response_model"),
                    "finish_reason": row.get("finish_reason"), "usage": row.get("usage"),
                    "raw_response_characters": len(row.get("raw_response") or ""),
                }, ensure_ascii=False))
        return
    for pair in post["paired_results"]:
        regression = (
            pair["candidate_semantics"] == "unsupported" and
            pair["original_semantics"] in {"supported_complete", "supported_partial", "no_answer_action"}
        )
        known = pair["parent_id"].startswith(("26ec3b7b11e9fcd4__", "b5c3bd1132dc9df8__", "relabel-9af639"))
        if not regression and not known:
            continue
        item = dict(pair)
        for side, suffix in (("original", "original_prompt_v1"), ("candidate", "fact_coverage_prompt_v1")):
            rid = pair["parent_id"] + "::" + suffix
            item[side + "_shown"] = rows[rid].get("grounded", {}).get("answer", "")
            item[side + "_judgement"] = {}
            for kind in ("coverage", "semantics"):
                gid = states[rid].get(kind + "_group_id")
                if gid:
                    result = judgements[gid]
                    item[side + "_judgement"][kind] = result.get("judgement", {"error": result.get("error")})
        print(json.dumps(item, ensure_ascii=False))
    print(json.dumps({"audit_failures": post["failures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
