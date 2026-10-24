import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as p
import expand_input_failures_20261022 as e
from test_expand_reader_errors_v3 import fixture


class Tests(unittest.TestCase):
    def test_bounded_coding_plan_all_reviewed(self):
        for stage in ("questions", "annotation"):
            settings = e.settings(stage).validate()
            self.assertEqual(settings.endpoint, p.CODING_PLAN_ENDPOINT)
            self.assertEqual(settings.review_fraction, 1)
        self.assertEqual(sum(e.settings(s).max_api_calls for s in ("questions", "annotation")), 600)
        counts = Counter(k for i in range(60) for k in e.schedule(i))
        self.assertEqual(counts, dict(speaker_direction=60, minimal_complete=30,
                                     event_state_scope=30, partial_support=30, evidence_boundary=30))

    def test_teacher_only_instructions(self):
        row = fixture()
        before = json.dumps(row, sort_keys=True)
        messages = e.annotation_messages(row)
        self.assertIn(e.LABEL_EXTRA, messages[0]["content"])
        self.assertNotIn(e.LABEL_EXTRA, row["prompt"])
        self.assertEqual(before, json.dumps(row, sort_keys=True))
        self.assertIn("优先answer_directly", e.LABEL_EXTRA)

    def test_review_issues_quarantined_and_cache_reused(self):
        row = fixture()
        calls = []
        def request(settings, messages):
            calls.append(messages)
            value = (dict(next_action="answer_directly", supported_facts=[
                dict(fact="甲把书交给乙。", evidence_ids=["E1"])]) if len(calls) == 1 else
                dict(label_supported=True, decision_appropriate=True, issues=["漏答另一子问"]))
            return dict(id=str(len(calls)), model="glm-5.3", usage=dict(total_tokens=100),
                        choices=[dict(finish_reason="stop", message=dict(content=p.compact(value)))])
        with tempfile.TemporaryDirectory() as tmp:
            settings = e.settings("annotation")
            settings.request_interval = 0
            runner = e.Runner(Path(tmp), [row], settings, request)
            try:
                self.assertEqual(runner.process(row), "quarantined")
                self.assertEqual(runner.process(row), "quarantined")
                self.assertEqual(len(calls), 2)
                self.assertEqual(row["target"], "OLD_TARGET_DO_NOT_COPY")
            finally:
                runner.ledger.close()

    def test_round_and_full20_preserved(self):
        rounds = Counter(e.question_round(seed, index) for seed in range(60) for index in range(3))
        self.assertEqual(rounds, {"1/2": 90, "2/2": 90})
        self.assertNotEqual(e.question_round(0, 0), e.question_round(0, 1))
        row = fixture()
        restored = e.restore_messages(row)
        p.check_row(restored)
        self.assertEqual(restored["round"], row["round"])
        self.assertEqual(len(restored["documents"]), 20)
        row["documents"].pop()
        with self.assertRaises(ValueError):
            e.restore_messages(row)


if __name__ == "__main__":
    unittest.main()
