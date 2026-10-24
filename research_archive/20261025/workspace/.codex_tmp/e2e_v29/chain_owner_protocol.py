"""Lossless experimental metadata rewrite; never selects or edits story text."""
import re

from selector_protocol import blocks


MEMBER = re.compile(r"\[CHAIN_MEMBER_([1-9][0-9]*)\]")
OWNER = re.compile(r"〔引用归属:(E[1-9][0-9]*)〕")


def with_explicit_owner(evidence):
    """Replace competing member numbers with their unchanged outer E owner."""
    rewritten, receipts = [], []
    for eid, body in blocks(evidence):
        if "〔引用归属:" in body:
            raise ValueError("reserved experimental marker already present")
        matches = list(MEMBER.finditer(body))
        if "[CHAIN_MEMBER_" in MEMBER.sub("", body):
            raise ValueError("malformed member marker")
        indices = [int(match.group(1)) for match in matches]
        if indices and indices != list(range(1, len(indices) + 1)):
            raise ValueError("non-consecutive member metadata")
        replacement = f"〔引用归属:{eid}〕"
        updated = MEMBER.sub(lambda _: replacement, body)
        # Compare every span between markers byte-for-byte, including whitespace.
        before_parts = re.split(r"\[CHAIN_MEMBER_[1-9][0-9]*\]", body)
        after_parts = re.split(r"〔引用归属:E[1-9][0-9]*〕", updated)
        if before_parts != after_parts:
            raise AssertionError("non-metadata content changed")
        if OWNER.findall(updated) != [eid] * len(matches):
            raise AssertionError("wrong enclosing evidence owner")
        rewritten.append(f"[{eid}] {updated}")
        receipts.append({"evidence_id": eid, "member_count": len(matches),
                         "non_metadata_spans_identical": True})
    result = "\n".join(rewritten)
    if [eid for eid, _ in blocks(result)] != [eid for eid, _ in blocks(evidence)]:
        raise AssertionError("evidence labels or order changed")
    return result, receipts
