from copy import deepcopy
from pathlib import Path
import sys
import unittest

# Local transfer layout and canonical remote layout differ intentionally.
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent if (HERE.parent / "span_protocol.py").exists() else HERE.parent / "span_semantic_shadow_20260910.v53"
sys.path.insert(0, str(PARENT))
import span_protocol as old
import span_protocol_v2 as protocol


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.visible = "[E1] 甲：我在读书。" + "乙：我在种树，种树需要时间。" * 30 + "\n[E2] 丙：我不认识甲。"
        self.packet = protocol.make_packet(self.visible)

    def test_lossless_every_block(self):
        for eid, text in self.packet.blocks:
            self.assertEqual("".join(s.text for s in self.packet.spans if s.evidence_id == eid), text)

    def test_all_quotes_at_most_80(self):
        self.assertTrue(all(len(s.text) <= 80 for s in self.packet.spans))

    def test_long_sentence_lossless_split(self):
        packet = protocol.make_packet("[E1] " + "甲" * 1200)
        self.assertEqual(len(packet.spans), 15)
        self.assertEqual("".join(s.text for s in packet.spans), "甲" * 1200)

    def test_previous_span_namespace_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.references(self.packet, ["S1"])

    def test_new_namespace_rejected_by_old_packet(self):
        with self.assertRaises(protocol.ProtocolError):
            old.references(old.make_packet(self.visible), ["T1"])

    def test_valid_two_citations(self):
        draft = {"packet_id": self.packet.packet_id, "action": "answer",
                 "facts": [{"claim": "测试陈述。", "span_ids": ["T1", "T2"]}], "unresolved": []}
        before = deepcopy(draft)
        protocol.validate_draft(draft, self.packet)
        self.assertEqual(draft, before)

    def test_three_citations_rejected(self):
        draft = {"packet_id": self.packet.packet_id, "action": "answer",
                 "facts": [{"claim": "测试陈述。", "span_ids": ["T1", "T2", "T3"]}], "unresolved": []}
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_draft(draft, self.packet)

    def test_full_context_in_prompt(self):
        prompt = protocol.verify_prompt(self.packet, "问题", "陈述", "C1")
        self.assertIn("丙：我不认识甲。", prompt)
        for span in self.packet.spans:
            self.assertIn(span.span_id, prompt)
        self.assertNotIn('"span_ids":["S1"]', prompt)

    def test_packet_roundtrip(self):
        self.assertEqual(protocol.from_dict(protocol.packet_dict(self.packet)), self.packet)

    def test_hash_tampering_rejected(self):
        data = protocol.packet_dict(self.packet)
        data["original_visible"] += "伪造"
        with self.assertRaises(protocol.ProtocolError):
            protocol.from_dict(data)


if __name__ == "__main__":
    unittest.main()
