import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_symbolic_edges as a


class EdgeAuditTests(unittest.TestCase):
    def test_explicit_edge(self):
        self.assertEqual(a.parse_edge("箱号ABC123指向中转站DEF456。"), ("ABC123", "routes_to", "DEF456"))

    def test_negation_not_misread_as_positive(self):
        self.assertIsNone(a.parse_edge("箱号ABC123没有指向中转站DEF456。"))
        self.assertIsNone(a.parse_edge("不能确认箱号ABC123指向中转站DEF456。"))

    def test_question_not_positive(self):
        self.assertIsNone(a.parse_edge("箱号ABC123指向中转站DEF456？"))

    def test_other_language_stays_unparsed(self):
        self.assertIsNone(a.parse_edge("该箱子的口令可能是ABC123。"))


if __name__ == "__main__":
    unittest.main()
