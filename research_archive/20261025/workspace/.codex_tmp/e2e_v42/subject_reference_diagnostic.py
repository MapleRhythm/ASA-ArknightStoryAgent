"""Read-only name-presence diagnostic, NOT semantic validation or a new gate."""
from copy import deepcopy
import re


def evidence_blocks(text):
    parts = re.split(r"(?m)^\[(E\d+)\][ \t]*", text)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def name_present(name, text):
    if name.isascii():
        return re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text, re.I) is not None
    return name in text


def diagnose(*, question, hypothesis, facts, visible_evidence, catalog):
    blocks = evidence_blocks(visible_evidence)
    entities = hypothesis.get("entities") or []
    subjects = {}
    for entity in entities:
        # Do not turn heuristic fragments, generic nouns, single characters,
        # or model-only inventions into named-subject requirements.
        if not isinstance(entity, str) or len(entity) < 2 or entity not in catalog:
            continue
        if not name_present(entity, question):
            continue
        aliases = catalog[entity]
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("invalid alias catalog")
        names = list(dict.fromkeys([entity, *aliases]))
        if any(not n or len(n) > 80 or any(x in n for x in ("\r", "\n", "<|", "|>")) for n in names):
            raise ValueError("unsafe alias catalog")
        subjects[entity] = [n for n in names if len(n) >= 2]
    diagnostics = []
    for index, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("fact") or "")
        refs = fact.get("evidence_refs") or []
        if not isinstance(refs, list):
            continue
        eids = list(dict.fromkeys(
            str(ref.get("evidence_id") or "") for ref in refs if isinstance(ref, dict)
        ))
        unknown = [eid for eid in eids if eid not in blocks]
        known = [eid for eid in eids if eid in blocks]
        for entity, aliases in subjects.items():
            named_in_claim = [alias for alias in aliases if name_present(alias, claim)]
            if not named_in_claim:
                continue
            matched = {
                eid: [alias for alias in aliases if name_present(alias, blocks[eid])]
                for eid in known
            }
            matched = {eid: names for eid, names in matched.items() if names}
            other = {
                eid: [alias for alias in aliases if name_present(alias, body)]
                for eid, body in blocks.items() if eid not in known
            }
            other = {eid: names for eid, names in other.items() if names}
            diagnostics.append({
                "fact_index": index, "fact": claim, "question_subject": entity,
                "catalog_aliases": deepcopy(aliases), "claim_aliases": named_in_claim,
                "cited_eids": eids, "unavailable_reference_eids": unknown,
                "cited_subject_mentions": matched, "uncited_subject_mentions": other,
                # Missing/invalid references remain an independent old-schema
                # issue, not evidence that a readable source lacks a subject.
                "named_subject_absent_in_available_cited_blocks": bool(known) and not unknown and not matched,
            })
    return diagnostics
