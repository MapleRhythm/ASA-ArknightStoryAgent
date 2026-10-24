"""80-codepoint spans and <=2 citations per fresh fact; old gates unchanged.

T IDs are versioned separately from v53 attempt01 S IDs. All original text
remains visible. Mechanical cuts do not infer speaker or coreference ownership.
"""
import re
import span_protocol as original
from span_protocol import (
    Packet, Span, ProtocolError, require, digest, validate_packet, packet_dict,
    from_dict, select_spans, references, strict_object, parse_prefilled, keys,
    short_string, validate_verdict, semantic_pass, chat,
)

VERIFY_RULES = original.VERIFY_RULES.replace('["S1"]', '["T1"]')
DRAFT_RULES = original.DRAFT_RULES.replace("最多8个", "每条事实最多2个，每个片段不超过80字").replace('["S1"]', '["T1"]')


def make_packet(visible):
    baseline = original.make_packet(visible)
    spans = []
    for eid, text in baseline.blocks:
        start = 0
        boundaries = [m.end() for m in re.finditer(r"[。！？；，,;]\s*|\s+", text)]
        while start < len(text):
            end = min(start + 80, len(text))
            if end < len(text):
                candidates = [cut for cut in boundaries if start + 32 <= cut <= end]
                if candidates:
                    end = candidates[-1]
            spans.append(Span(f"T{len(spans) + 1}", eid, start, end, text[start:end], digest(text)))
            start = end
    packet = Packet(baseline.packet_id, baseline.visible_sha256, baseline.blocks, tuple(spans), visible)
    validate_packet(packet)
    require(all(len(s.text) <= 80 for s in packet.spans), "quote_limit")
    return packet


def render_packet(packet):
    validate_packet(packet)
    parts = []
    for eid, _ in packet.blocks:
        parts.append(f"[{eid}]")
        parts.extend(f"[{s.span_id}] {s.text}" for s in packet.spans if s.evidence_id == eid)
    return "\n".join(parts)


def validate_draft(value, packet):
    original.validate_draft(value, packet)
    for fact in value["facts"]:
        require(len(fact["span_ids"]) <= 2, "canonical_two_quotes_per_fact")
        require(all(len(s.text.strip()) <= 80 for s in select_spans(packet, fact["span_ids"])), "canonical_quote_80")
    return value


def verify_prompt(packet, question, claim, claim_id):
    return chat(VERIFY_RULES, {"packet_id": packet.packet_id, "claim_id": claim_id,
                              "question": question, "claim": claim, "source": render_packet(packet)})


def draft_prompt(packet, question):
    return chat(DRAFT_RULES, {"packet_id": packet.packet_id, "question": question, "source": render_packet(packet)})
