import copy
import json
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as p
import repair_input_failure_data_20261022 as repair


SOURCE = repair.ROOT / "input_failure_review_snapshot_20261022"


class Tests(unittest.TestCase):
    def test_patches_bind_to_immutable_full20_source(self):
        spec = repair.read(repair.SPEC)
        self.assertEqual(p.file_sha(SOURCE / "annotation/requests.jsonl"), spec["requests_sha256"])
        rows = list(p.json_rows(SOURCE / "annotation/requests.jsonl"))
        self.assertEqual(len(spec["repairs"]), 10)
        for change in spec["repairs"]:
            row = rows[change["position"] - 1]
            p.check_row(row)
            if "target" in change:
                p.check_target(change["target"], row)
                self.assertLessEqual(len(change["target"].get("supported_facts", [])), 3)

    def test_four_pending_labels_are_reused_not_regenerated(self):
        rows = list(p.json_rows(SOURCE / "annotation/requests.jsonl"))
        jobs = repair.jobs_readonly(SOURCE)
        before = p.file_sha(SOURCE / "annotation/ledger.sqlite3")
        for position in repair.read(repair.SPEC)["recover_pending_positions"]:
            row = rows[position - 1]
            self.assertEqual(jobs[row["id"]][0], "pending")
            label, receipt = repair.target_for(SOURCE, row, position, jobs)
            p.check_target(label, row)
            self.assertEqual(receipt["phase"], "annotate")
            self.assertEqual(receipt["finish_reason"], "stop")
        self.assertEqual(before, p.file_sha(SOURCE / "annotation/ledger.sqlite3"))

    def test_bounded_review_only_and_issues_are_not_rewritten(self):
        row = copy.deepcopy(list(p.json_rows(SOURCE / "annotation/requests.jsonl"))[1])
        change = repair.read(repair.SPEC)["repairs"][0]
        row.update(source_id=row["id"], source_position=2, candidate_target=change["target"])
        original_target = p.compact(row["candidate_target"])
        calls = []

        def fake(settings, messages):
            calls.append(messages)
            return dict(id="test-review", model="glm-5.3",
                        usage=dict(total_tokens=20),
                        choices=[dict(finish_reason="stop", message=dict(content=p.compact(dict(
                            label_supported=True, decision_appropriate=True, issues=["test issue"]))))])

        with tempfile.TemporaryDirectory() as directory:
            runner = repair.Runner(Path(directory), [row], repair.settings(), request_fn=fake)
            try:
                self.assertEqual(runner.process(row), "quarantined")
                self.assertEqual(runner.process(row), "quarantined")
                self.assertEqual(len(calls), 1)
                self.assertEqual(p.compact(row["candidate_target"]), original_target)
                self.assertEqual(runner.ledger.receipts()[0]["phase"], "review_repair")
            finally:
                runner.ledger.close()
        settings = repair.settings()
        self.assertEqual(settings.endpoint, p.CODING_PLAN_ENDPOINT)
        self.assertEqual(settings.max_api_calls, 32)
        self.assertEqual(settings.max_output_tokens, 16384)


if __name__ == "__main__":
    unittest.main()
