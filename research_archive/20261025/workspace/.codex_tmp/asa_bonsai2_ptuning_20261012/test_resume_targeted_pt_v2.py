import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import annotation_protocol as p
import resume_targeted_pt_v2 as r


class RecoveryTests(unittest.TestCase):
    def test_finished_stage_not_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def operation(out, limit):
                calls.append(limit)
                return {"remaining_tasks": 0}
            r.drain(Path(directory), "questions", operation)
            self.assertEqual(calls, [0])

    def test_pending_retried_until_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            results = iter([{"remaining_tasks": 1}, {"remaining_tasks": 0}])
            self.assertEqual(r.drain(Path(directory), "annotation",
                                    lambda out, limit: next(results))["remaining_tasks"], 0)

    def test_only_failed_jobs_quarantined_payload_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "requests.jsonl").write_text('{"id":"ok"}\n{"id":"bad"}\n')
            ledger = p.Ledger(path / "ledger.sqlite3")
            error = {"error": {"kind": "invalid_json_or_protocol", "retryable": True,
                               "terminal": False}}
            ledger.save_job("ok", "fp1", "accepted", {"target": "untouched"})
            ledger.save_job("bad", "fp2", "pending", error)
            ledger.close()
            self.assertEqual(r.quarantine_exhausted(path, 1), ["bad"])
            ledger = p.Ledger(path / "ledger.sqlite3")
            self.assertEqual(ledger.job("ok"), ("fp1", "accepted", {"target": "untouched"}))
            self.assertEqual(ledger.job("bad")[1], "quarantined")
            self.assertEqual(ledger.job("bad")[2]["error"], error["error"])
            ledger.close()

    def test_budget_stop_never_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "requests.jsonl").write_text('{"id":"budget"}\n')
            ledger = p.Ledger(path / "ledger.sqlite3")
            ledger.save_job("budget", "fp", "pending", {"reason": "budget_limit"})
            ledger.close()
            with self.assertRaisesRegex(ValueError, "budget"):
                r.quarantine_exhausted(path, 1)

    def test_retry_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def operation(out, limit):
                calls.append(limit)
                return {"remaining_tasks": 1 if len(calls) <= 3 else 0}
            with patch.object(r, "quarantine_exhausted", return_value=["bad"]) as isolate:
                r.drain(Path(directory), "questions", operation)
                self.assertEqual(len(calls), 4)
                isolate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
