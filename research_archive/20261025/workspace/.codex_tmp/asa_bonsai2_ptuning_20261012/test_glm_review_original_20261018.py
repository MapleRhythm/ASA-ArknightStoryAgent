import json
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as p
import glm_review_original_20261018 as m


class TestReview(unittest.TestCase):
    def test_full_source_selection(self):
        rows = m.load(Path(__file__).with_name(
            "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"))
        self.assertEqual([r["review_position"] for r in rows], list(range(301, 487)))
        self.assertTrue(all(len(r["documents"]) == 20 for r in rows))

    def run_case(self, review):
        row = m.load(Path(__file__).with_name(
            "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"))[0]
        phases = []

        def request(settings, messages):
            phases.append(messages)
            return {"id": "mock", "model": "glm-5.3", "usage": {"total_tokens": 100},
                    "choices": [{"finish_reason": "stop",
                                 "message": {"content": json.dumps(review)}}]}

        with tempfile.TemporaryDirectory() as tmp:
            runner = m.Runner(Path(tmp), [row], p.Settings(
                endpoint=p.CODING_PLAN_ENDPOINT, request_interval=0), request)
            original = row["target"]
            state = runner.process(row)
            self.assertEqual(runner.process(row), state)
            self.assertEqual(len(phases), 1)
            self.assertEqual(row["target"], original)
            self.assertEqual(runner.ledger.receipts()[0]["phase"], "review_original")
            modified = dict(row, target=original + " ")
            self.assertNotEqual(runner.fingerprint(row), runner.fingerprint(modified))
            runner.export_reviews()
            runner.ledger.close()
            return state

    def test_pass_and_cache(self):
        self.assertEqual(self.run_case(dict(label_supported=True,
                         decision_appropriate=True, issues=[])), "accepted")

    def test_issues_reject_even_when_booleans_true(self):
        self.assertEqual(self.run_case(dict(label_supported=True,
                         decision_appropriate=True, issues=["漏答"])), "quarantined")


if __name__ == "__main__":
    unittest.main()
