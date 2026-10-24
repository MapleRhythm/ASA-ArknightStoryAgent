"""Post-hoc witness tracing on frozen inputs; never used in generation prompts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
V40 = HERE.parent / "routing_frozen_retrieval_20260910.v40"
WITNESSES = [
    ("fixed58-007", "nastie_mother", "谢谢你照顾我的母亲，费尔南小姐。"),
    ("fixed58-010", "master_ling_brother", "那位宗师是令小姐的兄长"),
    ("fixed58-017", "train_accident", "你的父母确实死于列车意外事故"),
    ("fixed58-023", "effort_reason", "我就是看在你花了心思的分上，决定继续教你罢了。"),
    ("fixed58-046", "moyi_instruction_summary", "槐天裴受一个叫莫佚的山海众指使"),
    ("fixed58-046", "moyi_motive", "可我最喜欢看那些自诩武林豪杰的人争得你死我活"),
    ("fixed58-049", "friston_name", "特雷弗·弗里斯顿"),
    ("fixed58-049", "personality_simulation", "我是计划的人格模拟"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contains(needle, haystack):
    return "".join(needle.split()) in "".join(haystack.split())


def object_matches(objects, needle):
    matches = []
    for index, obj in enumerate(objects):
        direct = contains(needle, obj["document"].get("clean_text", ""))
        chain = contains(needle, obj.get("evidence_chain_text", ""))
        if direct or chain:
            matches.append({
                "rank_one_based": index + 1, "document_id": obj["document"]["id"],
                "direct_document_text": direct, "chain_text": chain,
            })
    return matches


def visible_matches(text, needle):
    parts = re.split(r"(?m)^\[(E\d+)\]\s*", text)
    return [eid for eid, block in zip(parts[1::2], parts[2::2], strict=True) if contains(needle, block)]


def main():
    data, pre = load(V40 / "retrieval.json"), load(HERE / "preflight.json")
    assert pre["sha256"][str(V40 / "retrieval.json")] == sha(V40 / "retrieval.json")
    old = {(r["id"], r["condition"]): r for r in data["conditions"]}
    actual = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    rows = []
    for case_id, label, needle in WITNESSES:
        direct_ids = {
            m["document_id"] for kind in ("control", "candidate")
            for m in object_matches(old[(case_id, kind)]["evidence"], needle)
            if m["direct_document_text"]
        }
        for kind in ("control", "candidate"):
            source, reader = old[(case_id, kind)], actual[(case_id, kind)]
            hits = []
            for index, call in enumerate(source["actual_search_calls"]):
                for rank, doc_id in enumerate(call["returned_document_ids"]):
                    if doc_id in direct_ids:
                        hits.append({
                            "call_index": index, "method": call["method"], "query": call["query"],
                            "hit_rank_one_based": rank + 1, "document_id": doc_id,
                        })
            rows.append({
                "id": case_id, "condition": kind, "witness": label, "needle": needle,
                "direct_source_doc_ids_observed_in_either_condition": sorted(direct_ids),
                "initial_query_hits_for_those_docs": hits,
                "final_retrieved_object_matches": object_matches(source["evidence"], needle),
                "prepared_object_matches": object_matches(source["prompt_evidence"], needle),
                "v40_preview_eids": visible_matches(source["evidence_brief"], needle),
                "actual_reader_eids": visible_matches(reader["visible_evidence"], needle),
            })
    output = {
        "version": "posthoc_frozen_witness_trace_v41",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_review_type": "posthoc_assistant_diagnostic_not_gold",
        "production_changed": False, "training_ready": False, "model_calls": 0, "retrieval_calls": 0,
        "basis_sha256": {str(p): sha(p) for p in (V40 / "retrieval.json", HERE / "preflight.json", Path(__file__))},
        "rows": rows,
        "limitations": [
            "Witnesses are selected from inspected source text, not a preregistered recall benchmark.",
            "Exact phrase tracing with whitespace normalization does not test paraphrases or full semantic coverage.",
            "A chain's lead document ID is not automatically the document containing a matched inner quote.",
            "Initial hits are linked only to direct matching document IDs observed in either final evidence set; no unseen corpus is searched.",
            "Initial search hit does not prove that query caused final ranking or packing; no counterfactual retrieval was run.",
            "Story-summary instruction witness is not character dialogue; other witnesses do not establish every identity link.",
        ],
    }
    with (HERE / "evidence_trace_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    for r in rows:
        print({
            "id": r["id"], "condition": r["condition"], "witness": r["witness"],
            "query_hits": len(r["initial_query_hits_for_those_docs"]),
            "full": len(r["final_retrieved_object_matches"]), "prepared": len(r["prepared_object_matches"]),
            "preview_eids": r["v40_preview_eids"], "reader_eids": r["actual_reader_eids"],
        })


if __name__ == "__main__":
    main()
