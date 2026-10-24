import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import continue_input_failure_batch_20261022 as c


class Tests(unittest.TestCase):
    def test_never_bypass_terminal_or_unknown_failures(self):
        self.assertTrue(c.allowed_pending(dict(error=dict(kind="incomplete_output", terminal=False))))
        for kind in ("authentication_billing_or_configuration", "insufficient_balance",
                     "network_or_timeout", "missing_api_key", "unknown"):
            self.assertFalse(c.allowed_pending(dict(error=dict(kind=kind, terminal=False))))
        self.assertFalse(c.allowed_pending(dict(error=dict(kind="incomplete_output", terminal=True))))
        self.assertFalse(c.allowed_pending({}))

    def test_read_only_inspection_does_not_mutate_inflight(self):
        with tempfile.TemporaryDirectory() as name:
            out = Path(name)
            folder = out / "questions"
            folder.mkdir()
            (folder / "requests.jsonl").write_text('{"id":"a"}\n{"id":"b"}\n')
            with sqlite3.connect(folder / "ledger.sqlite3") as db:
                db.execute("CREATE TABLE jobs(task_id TEXT,state TEXT,result TEXT)")
                db.execute("CREATE TABLE calls(state TEXT)")
                db.execute("INSERT INTO calls VALUES ('inflight')")
                db.execute("INSERT INTO jobs VALUES (?,?,?)", ("a", "accepted", "{}"))
                db.execute("INSERT INTO jobs VALUES (?,?,?)", (
                    "b", "pending", json.dumps(dict(error=dict(kind="incomplete_output", terminal=False)))))
            accepted, excluded = c.inspect_generation(out)
            self.assertEqual(accepted, 1)
            self.assertEqual(excluded[0]["id"], "b")
            with sqlite3.connect(folder / "ledger.sqlite3") as db:
                self.assertEqual(db.execute("SELECT state FROM calls").fetchone()[0], "inflight")
                db.execute("DELETE FROM jobs WHERE task_id='b'")
            with self.assertRaisesRegex(RuntimeError, "unexpected/terminal"):
                c.inspect_generation(out)


if __name__ == "__main__":
    unittest.main()
