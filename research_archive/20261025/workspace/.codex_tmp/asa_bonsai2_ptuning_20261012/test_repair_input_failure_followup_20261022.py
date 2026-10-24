import copy
import unittest

import annotation_protocol as p
import repair_input_failure_data_20261022 as base
import repair_input_failure_followup_20261022 as followup


class Tests(unittest.TestCase):
    def test_three_corrections_valid_and_source_unchanged(self):
        source = base.ROOT / "input_failure_review_snapshot_20261022"
        rows = list(p.json_rows(source / "annotation/requests.jsonl"))
        digest = p.file_sha(source / "annotation/requests.jsonl")
        self.assertEqual(set(followup.TARGETS), {108, 144, 179})
        for position, target in followup.TARGETS.items():
            p.check_target(copy.deepcopy(target), rows[position - 1])
        self.assertEqual(digest, p.file_sha(source / "annotation/requests.jsonl"))
        self.assertEqual(followup.TARGETS[108]["next_action"], "abstain")
        self.assertEqual(followup.TARGETS[144]["next_action"], "abstain")
        self.assertEqual(followup.TARGETS[179]["next_action"], "answer_directly")

    def test_followup_is_separately_bounded(self):
        original = base.settings
        try:
            followup.bind_settings()
            settings = base.settings()
            self.assertEqual(settings.max_api_calls, 6)
            self.assertEqual(settings.max_total_tokens, 300000)
            self.assertEqual(settings.max_output_tokens, 16384)
            self.assertEqual(settings.endpoint, p.CODING_PLAN_ENDPOINT)
        finally:
            base.settings = original
        self.assertEqual(base.settings().max_api_calls, 32)


if __name__ == "__main__":
    unittest.main()
