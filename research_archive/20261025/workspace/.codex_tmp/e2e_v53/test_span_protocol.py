from copy import deepcopy
from dataclasses import replace
import json
import unittest

from span_protocol import (
    ProtocolError, draft_prompt, from_dict, make_packet, packet_dict, parse_prefilled,
    references, render_packet, semantic_pass, validate_draft, validate_packet,
    validate_verdict, verify_prompt,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.packet = make_packet("[E1] [CHAIN_MEMBER_1] 甲：我在读书。 乙：我在种树。\n[E2] 丙：我不认识甲。")
        self.verdict = {
            "packet_id": self.packet.packet_id, "claim_id": "C1", "verdict": "supported",
            "relevance": "answers", "subject": "甲", "relation": "在做", "object": "读书",
            "span_ids": ["S1"], "reason": "原文直述。",
        }
        self.draft = {"packet_id": self.packet.packet_id, "action": "answer",
                      "facts": [{"claim": "甲在读书。", "span_ids": ["S1"]}], "unresolved": []}

    def test_partition_roundtrip(self):
        self.assertEqual(from_dict(packet_dict(self.packet)), self.packet)
        for eid, text in self.packet.blocks:
            self.assertEqual("".join(s.text for s in self.packet.spans if s.evidence_id == eid), text)

    def test_unicode_codepoints(self):
        packet = make_packet("[E1] 甲：🌲e\u0301。\r\n乙：好。")
        self.assertEqual(packet.spans[0].text, dict(packet.blocks)["E1"])

    def test_large_sentence_never_truncated(self):
        text = "甲" * 1200
        self.assertEqual(make_packet("[E1] " + text).spans[0].text, text)

    def test_no_inferred_speaker_metadata(self):
        self.assertNotIn("speaker", packet_dict(self.packet)["spans"][0])
        self.assertIn("甲：我在读书。 乙：我在种树。", render_packet(self.packet))

    def test_duplicate_evidence_rejected(self):
        with self.assertRaises(ProtocolError):
            make_packet("[E1] 甲。\n[E1] 乙。")

    def test_unknown_span_rejected(self):
        with self.assertRaises(ProtocolError):
            references(self.packet, ["S999"])

    def test_repeated_span_rejected(self):
        with self.assertRaises(ProtocolError):
            references(self.packet, ["S1", "S1"])

    def test_empty_reference_rejected(self):
        with self.assertRaises(ProtocolError):
            references(self.packet, [])

    def test_foreign_packet_rejected(self):
        self.verdict["packet_id"] = make_packet("[E1] 不同内容。").packet_id
        with self.assertRaises(ProtocolError):
            validate_verdict(self.verdict, self.packet, "C1")

    def test_changed_source_rejected(self):
        packet = replace(self.packet, blocks=(("E1", "错误。"), self.packet.blocks[1]))
        with self.assertRaises(ProtocolError):
            validate_packet(packet)

    def test_changed_parent_rejected(self):
        packet = replace(self.packet, spans=(replace(self.packet.spans[0], evidence_id="E2"), *self.packet.spans[1:]))
        with self.assertRaises(ProtocolError):
            validate_packet(packet)

    def test_changed_quote_rejected(self):
        packet = replace(self.packet, spans=(replace(self.packet.spans[0], text="伪造"), *self.packet.spans[1:]))
        with self.assertRaises(ProtocolError):
            validate_packet(packet)

    def test_gap_rejected(self):
        packet = replace(self.packet, spans=(replace(self.packet.spans[0], start=1), *self.packet.spans[1:]))
        with self.assertRaises(ProtocolError):
            validate_packet(packet)

    def test_copy_quotes_exactly(self):
        refs = references(self.packet, ["S2"])
        self.assertEqual(refs, [{"evidence_id": "E2", "quote": "丙：我不认识甲。"}])

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_prefilled('{"a":1,"a":2}')

    def test_brace_prefill_only(self):
        self.assertEqual(parse_prefilled('"a":1}'), ({"a": 1}, "known_opening_brace_prefill"))
        self.assertEqual(parse_prefilled('{"a":1}')[1], "strict")

    def test_fences_trailing_text_rejected(self):
        for raw in ('```json\n{"a":1}\n```', '{"a":1} trailing', '{"a":NaN}', '{"a":1'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_prefilled(raw)

    def test_extra_keys_rejected(self):
        self.verdict["corrected_claim"] = "不允许改写"
        with self.assertRaises(ProtocolError):
            validate_verdict(self.verdict, self.packet, "C1")

    def test_claim_identity_rejected(self):
        with self.assertRaises(ProtocolError):
            validate_verdict(self.verdict, self.packet, "C2")

    def test_missing_support_rejected(self):
        self.verdict["span_ids"] = []
        with self.assertRaises(ProtocolError):
            validate_verdict(self.verdict, self.packet, "C1")

    def test_insufficient_without_span_allowed(self):
        self.verdict.update(verdict="insufficient", span_ids=[])
        validate_verdict(self.verdict, self.packet, "C1")
        self.assertFalse(semantic_pass(self.verdict))

    def test_supported_unrelated_does_not_pass(self):
        self.verdict["relevance"] = "unrelated"
        self.assertFalse(semantic_pass(validate_verdict(self.verdict, self.packet, "C1")))

    def test_uncited_context_does_not_pass(self):
        self.assertFalse(semantic_pass(self.verdict, ["S2"]))
        self.assertTrue(semantic_pass(self.verdict, ["S1"]))

    def test_draft_no_quote_injection(self):
        self.draft["facts"][0]["quote"] = "伪造"
        with self.assertRaises(ProtocolError):
            validate_draft(self.draft, self.packet)

    def test_draft_empty_answer_rejected(self):
        self.draft["facts"] = []
        with self.assertRaises(ProtocolError):
            validate_draft(self.draft, self.packet)

    def test_draft_insufficient_with_fact_rejected(self):
        self.draft["action"] = "insufficient"
        with self.assertRaises(ProtocolError):
            validate_draft(self.draft, self.packet)

    def test_valid_draft_unchanged(self):
        before = deepcopy(self.draft)
        self.assertEqual(validate_draft(self.draft, self.packet), before)
        self.assertEqual(self.draft, before)

    def test_prompt_contains_all_spans_and_no_labels(self):
        prompt = verify_prompt(self.packet, "谁在读书？", "甲在读书。", "C1")
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n{"))
        self.assertIn("丙：我不认识甲。", prompt)
        self.assertNotIn("assistant_expected", prompt)
        self.assertIn(self.packet.packet_id, draft_prompt(self.packet, "谁在读书？"))


if __name__ == "__main__":
    unittest.main()
