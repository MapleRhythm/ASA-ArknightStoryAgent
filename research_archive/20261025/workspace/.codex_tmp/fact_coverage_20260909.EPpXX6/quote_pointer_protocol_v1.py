"""Experimental lossless source annotation and fail-closed pointer binding.

No retrieval, model calls, semantic judgement or live runtime edits here.
All offsets refer to the original unannotated visible evidence, not a corpus.
"""
from copy import deepcopy
import json
import re

HEADER = re.compile(r"^\[E([1-9][0-9]*)\][ \t]?", re.MULTILINE)
BREAKS = re.compile(r"[。！？!?；;\n]| +")
POINTER_RULES = (
    "单条引文通过 quote_id 选择该 evidence_id 下标记的一个原文片段（如 Q1）；"
    "程序会逐字读取该片段作为quote，禁止自行输出quote文本；"
    "不得选择别的E编号下的片段，不得编造不存在的quote_id；"
    "每个片段最多80字，仍需选择足以支持该条fact的片段；"
)
OLD_COPY_RULE = "单条 quote 必须从该编号对应的证据原文复制，推荐20-60字，硬上限80字；"
OLD_SCHEMA_REF = '"evidence_refs":[{"evidence_id":"","quote":""}]'
NEW_SCHEMA_REF = '"evidence_refs":[{"evidence_id":"E1","quote_id":"Q1"}]'


def slices(text):
    """Partition, never truncate; prefer punctuation/space after character20."""
    start = 0
    while start < len(text):
        end = min(start + 80, len(text))
        if end < len(text):
            boundaries = [m.end() for m in BREAKS.finditer(text, start, end)
                          if m.end() - start >= 20]
            if boundaries:
                end = boundaries[-1]
        yield start, end
        start = end


def annotate(evidence):
    if not isinstance(evidence, str) or not evidence:
        raise ValueError("visible evidence required")
    headers = list(HEADER.finditer(evidence))
    if not headers or headers[0].start() != 0:
        raise ValueError("expected complete E-labelled visible evidence")
    if len({m.group(1) for m in headers}) != len(headers):
        raise ValueError("duplicate evidence ID")
    catalog, pieces = [], []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(evidence)
        pieces.append(header.group())
        body = evidence[header.end():end]
        for start, stop in slices(body):
            quote = body[start:stop]
            if quote.strip():
                qid = f"Q{len(catalog) + 1}"
                catalog.append({
                    "quote_id": qid, "evidence_id": "E" + header.group(1),
                    "start": header.end() + start, "end": header.end() + stop,
                    "quote": quote,
                })
                pieces.append("[" + qid + "]")
            pieces.append(quote)
    annotated = "".join(pieces)
    validate_catalog(evidence, catalog)
    # Rebuild from source offsets, not by stripping labels that may also occur
    # as literal source text.
    cursor, expected = 0, []
    for ref in catalog:
        expected.append(evidence[cursor:ref["start"]])
        expected.extend(("[" + ref["quote_id"] + "]", ref["quote"]))
        cursor = ref["end"]
    expected.append(evidence[cursor:])
    if annotated != "".join(expected):
        raise AssertionError("annotation changed source text")
    return annotated, catalog


def validate_catalog(evidence, catalog):
    seen, end = set(), 0
    headers = list(HEADER.finditer(evidence))
    blocks = {"E" + h.group(1): (h.end(), headers[i + 1].start()
                                if i + 1 < len(headers) else len(evidence))
              for i, h in enumerate(headers)}
    for ref in catalog:
        if set(ref) != {"quote_id", "evidence_id", "start", "end", "quote"}:
            raise ValueError("invalid catalog fields")
        if ref["quote_id"] in seen or not re.fullmatch(r"Q[1-9][0-9]*", ref["quote_id"]):
            raise ValueError("duplicate or malformed quote_id")
        if type(ref["start"]) is not int or type(ref["end"]) is not int:
            raise ValueError("noninteger source offsets")
        start, stop = ref["start"], ref["end"]
        block = blocks.get(ref["evidence_id"])
        if not block or not (end <= start < stop and block[0] <= start < stop <= block[1]):
            raise ValueError("out of block or overlapping quote")
        if not isinstance(ref["quote"], str) or not ref["quote"].strip():
            raise ValueError("empty quote")
        if len(ref["quote"]) > 80 or evidence[start:stop] != ref["quote"]:
            raise ValueError("quote is not exact bounded source slice")
        seen.add(ref["quote_id"])
        end = stop


def make_user(original):
    evidence = original["evidence_brief"]
    annotated, catalog = annotate(evidence)
    user = original["user"]
    anchor = "evidence_brief:\n" + evidence + "\nminirag_hints: "
    if user.count(anchor) != 1:
        raise ValueError("ambiguous evidence boundary")
    if user.count(OLD_SCHEMA_REF) != 1 or user.count(OLD_COPY_RULE) != 1:
        raise ValueError("unexpected original quote contract")
    user = user.replace(anchor, "evidence_brief:\n" + annotated + "\nminirag_hints: ", 1)
    user = user.replace(OLD_SCHEMA_REF, NEW_SCHEMA_REF, 1)
    user = user.replace(OLD_COPY_RULE, POINTER_RULES, 1)
    user += (
        "\nquote_pointer_protocol: [E编号]仍为原证据编号，[Q编号]是程序添加的引用选择标签，"
        "不是剧情内容；所有原文完整保留。supported_facts和inferred_facts的evidence_refs"
        "都只填evidence_id和quote_id。编号命中不代表事实含义正确，fact仍只能陈述原文支持的内容。\n"
    )
    return user, annotated, catalog


def strict_loads(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def no_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)


def bind(value, evidence, catalog):
    """Resolve selected IDs only. Never fix IDs, select other text or drop facts."""
    validate_catalog(evidence, catalog)
    if not isinstance(value, dict):
        raise ValueError("output must be an object")
    if value.get("next_action") not in {"answer_directly", "retrieve_more", "abstain"}:
        raise ValueError("invalid next_action")
    lookup = {r["quote_id"]: r for r in catalog}
    result = deepcopy(value)
    for field in ("supported_facts", "inferred_facts"):
        if field not in result:
            continue
        if not isinstance(result[field], list):
            raise ValueError("facts must be lists")
        for fact in result[field]:
            if not isinstance(fact, dict) or not isinstance(fact.get("fact"), str):
                raise ValueError("fact must be a text-bearing object")
            refs = fact.get("evidence_refs")
            if not isinstance(refs, list) or not refs:
                raise ValueError("nonempty pointer refs required")
            resolved = []
            for ref in refs:
                if not isinstance(ref, dict) or set(ref) != {"evidence_id", "quote_id"}:
                    raise ValueError("pointer ref must contain exactly evidence_id and quote_id")
                eid, qid = ref["evidence_id"], ref["quote_id"]
                if not isinstance(eid, str) or not isinstance(qid, str) or qid not in lookup:
                    raise ValueError("unknown or non-string pointer")
                match = lookup[qid]
                if eid != match["evidence_id"]:
                    raise ValueError("cross-evidence pointer")
                resolved.append({"evidence_id": eid, "quote": match["quote"]})
            fact["evidence_refs"] = resolved
    if result["next_action"] == "answer_directly" and not result.get("supported_facts"):
        raise ValueError("direct answer requires supported_facts")
    return result
