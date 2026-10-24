"""Experimental single globally unique source pointer; never repair old E/Q pairs."""
from copy import deepcopy

from quote_pointer_protocol_v1 import (
    OLD_COPY_RULE, OLD_SCHEMA_REF, annotate, strict_loads, validate_catalog,
)

SINGLE_SCHEMA_REF = '"evidence_refs":[{"quote_id":"Q1"}]'
OLD_EID_RULE = "evidence_id 必须填该证据的编号（如 E1），不得编造不存在的编号；"
SINGLE_RULES = (
    "每个quote_id（如Q1）在本次提示中全局唯一；"
    "evidence_refs的每项只填写quote_id，不填写evidence_id或quote文本；"
    "程序按冻结目录唯一还原该Q编号所属的E编号和逐字原文，不猜测或修复编号；"
    "不得编造不存在的quote_id，仍需选择足以支持该条fact的片段；"
    "每个片段最多80字；"
)


def make_user(original):
    evidence = original["evidence_brief"]
    annotated, catalog = annotate(evidence)
    user = original["user"]
    anchor = "evidence_brief:\n" + evidence + "\nminirag_hints: "
    if user.count(anchor) != 1:
        raise ValueError("ambiguous evidence boundary")
    if (user.count(OLD_SCHEMA_REF) != 1 or user.count(OLD_COPY_RULE) != 1
            or user.count(OLD_EID_RULE) != 1):
        raise ValueError("unexpected original quote contract")
    user = user.replace(anchor, "evidence_brief:\n" + annotated + "\nminirag_hints: ", 1)
    user = user.replace(OLD_SCHEMA_REF, SINGLE_SCHEMA_REF, 1)
    user = user.replace(OLD_EID_RULE, "", 1)
    user = user.replace(OLD_COPY_RULE, SINGLE_RULES, 1)
    user += (
        "\nsingle_quote_pointer_protocol: [E编号]是原证据分块，[Q编号]是程序添加的全局唯一引用标签，"
        "不是剧情内容；所有原文完整保留。supported_facts和inferred_facts的evidence_refs"
        "每项都只能有quote_id一个字段。Q编号命中不代表事实含义正确，"
        "fact仍只能陈述所选原文支持的内容；已有每fact160字、总400字等规则保持不变。\n"
    )
    return user, annotated, catalog


def bind(value, evidence, catalog):
    """A new contract, not a repair of dual-pointer outputs."""
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
                if not isinstance(ref, dict) or set(ref) != {"quote_id"}:
                    raise ValueError("single pointer ref must contain exactly quote_id")
                qid = ref["quote_id"]
                if not isinstance(qid, str) or qid not in lookup:
                    raise ValueError("unknown or non-string pointer")
                match = lookup[qid]
                resolved.append({"evidence_id": match["evidence_id"], "quote": match["quote"]})
            fact["evidence_refs"] = resolved
    if result["next_action"] == "answer_directly" and not result.get("supported_facts"):
        raise ValueError("direct answer requires supported_facts")
    return result
