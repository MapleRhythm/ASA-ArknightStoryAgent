#!/usr/bin/env python3
"""Certify missing/citation-mismatched explicit edges in a restricted grammar.

Unrecognized language stays unknown. This is not a general natural-language judge.
"""
import collections
import re

from build_inputs import ROOT, load, write


def parse_edge(text):
    text = re.sub(r"\s+", "", text).rstrip("。.")
    for pattern, relation in (
        (r"箱号([0-9A-F]{6})指向中转站([0-9A-F]{6})", "routes_to"),
        (r"中转站([0-9A-F]{6})的通行口令为([0-9A-F]{6})", "has_code"),
    ):
        match = re.fullmatch(pattern, text)
        if match:
            return match[1], relation, match[2]
    return None


def audit(source, payload):
    diag = source["diagnostic"]
    if diag["suite"] != "synthetic_controls" or payload.get("next_action") != "answer_directly":
        return None
    fields = diag["input_fields"]
    by_id = {}
    for eid, doc in zip(fields["evidence_ids"], fields["documents"], strict=True):
        edge = parse_edge(doc["text"])
        if edge is None:
            raise ValueError("source outside declared grammar")
        by_id[eid] = edge
    visible_edges = set(by_id.values())
    annotated_edges = {(e["subject"], e["relation"], e["object"])
                       for e in diag["reference"]["visible_graph"]}
    if visible_edges != annotated_edges:
        raise ValueError("visible source vs graph annotation mismatch")
    facts = payload.get("supported_facts")
    if not isinstance(facts, list):
        return {"status": "unknown_output_structure"}
    details = []
    for fact in facts:
        if not isinstance(fact, dict):
            details.append({"status": "unknown_output_structure"})
            continue
        edge = parse_edge(str(fact.get("fact", "")))
        if edge is None:
            details.append({"status": "unparsed", "fact": fact.get("fact")})
            continue
        refs = fact.get("evidence_ids", [])
        refs = refs if isinstance(refs, list) else []
        cited_edges = {by_id[eid] for eid in refs if isinstance(eid, str) and eid in by_id}
        if edge not in visible_edges:
            status = "explicit_edge_absent_from_visible_record"
        elif edge not in cited_edges:
            status = "visible_edge_but_wrong_citation"
        else:
            status = "explicit_edge_supported_by_own_citation"
        details.append({"status": status, "fact": fact.get("fact"), "edge": edge, "refs": refs})
    return {
        "facts": details,
        "any_explicit_missing_edge": any(d["status"] == "explicit_edge_absent_from_visible_record" for d in details),
        "any_wrong_citation": any(d["status"] == "visible_edge_but_wrong_citation" for d in details),
        "unparsed_facts": sum(d["status"] in {"unparsed", "unknown_output_structure"} for d in details),
        "limitation": "Restricted exact affirmative grammar; no verdict on unparsed facts, minimality, or whole-answer correctness.",
    }


def main():
    import json
    sources = {row["id"]: row for row in load(ROOT / "input/eval_matrix.json")}
    records, summary = [], {}
    for run in ("full_base_v1", "full_clean_v1"):
        stats = collections.Counter()
        for pred in load(ROOT / f"runs/{run}/predictions.json"):
            try:
                payload = json.loads(pred["raw_output"])
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            result = audit(sources[pred["id"]], payload)
            if result is None:
                continue
            status = sources[pred["id"]]["diagnostic"]["reference"]["status"]
            stats["answer_rows"] += 1
            stats["any_explicit_missing_edge"] += result.get("any_explicit_missing_edge", False)
            stats["any_wrong_citation"] += result.get("any_wrong_citation", False)
            stats["rows_with_unparsed_facts"] += bool(result.get("unparsed_facts", 0))
            if result.get("any_explicit_missing_edge"):
                stats[f"missing_edge_when_{status}"] += 1
            records.append({"reader": pred["reader"], "id": pred["id"], "reference_status": status,
                            "raw_output_sha256": pred["raw_output_sha256"], **result})
        summary[run] = dict(stats)
    write(ROOT / "audits/symbolic_edge_audit.json", {"summary": summary, "records": records})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
