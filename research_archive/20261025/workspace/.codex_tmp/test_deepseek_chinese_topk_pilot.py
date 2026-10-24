import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("pilot", ROOT / "deepseek_chinese_topk_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class RequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = pilot.load_rows(ROOT / "asa_small_20260913_reasoning3.jsonl")
        cls.baseline = pilot.load_rows(ROOT / "deepseek_chinese_pilot_20260913.jsonl")

    def test_original_six_request_hashes_match(self):
        pilot.validate_baseline(self.rows, self.baseline)

    def test_twenty_only_changes_evidence(self):
        for row in self.rows:
            old, new = pilot.build_body(row, 10), pilot.build_body(row, 20)
            for key in old:
                if key != "messages":
                    self.assertEqual(old[key], new[key])
            self.assertEqual(old["messages"][0], new["messages"][0])
            text = new["messages"][1]["content"]
            for item in row["evidence"][:20]:
                self.assertIn(f"[{item['id']}]\n{item['clean_text']}", text)
            for item in row["evidence"][20:]:
                self.assertNotIn(f"[{item['id']}]\n", text)
            self.assertNotIn("reference_answers", text)

    def test_full_name_is_newly_included(self):
        row = self.rows[2]
        self.assertNotIn("安娜·莫罗佐娃", pilot.build_body(row, 10)["messages"][1]["content"])
        self.assertIn("安娜·莫罗佐娃", pilot.build_body(row, 20)["messages"][1]["content"])

    def test_invalid_limits_rejected(self):
        for top_k in (0, 33):
            with self.assertRaises(ValueError):
                pilot.build_body(self.rows[0], top_k)


if __name__ == "__main__":
    unittest.main()
