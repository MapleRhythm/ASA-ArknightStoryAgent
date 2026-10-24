import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch
import glm_compare as g


class ProtocolTests(unittest.TestCase):
    def test_selection_independent_of_order(self):
        rows = [{"id": f"{a}-{i}", "attribute": str(a)} for a in range(7) for i in range(5)]
        self.assertEqual(g.select_ids(rows), g.select_ids(list(reversed(rows))))
        self.assertEqual(len(g.select_ids(rows)), 14)

    def test_no_reference_in_request(self):
        row = {"system": "rules", "user": "question", "answers": ["SECRET_GOLD"]}
        body = g.body_for(row)
        self.assertNotIn("SECRET_GOLD", json.dumps(body))
        self.assertEqual(body["messages"], [{"role": "system", "content": "rules"},
                                          {"role": "user", "content": "question"}])

    def test_valid_reasoning_mode(self):
        body = g.body_for({"system": "", "user": "question"})
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertNotIn("/coding/", g.ENDPOINT)

    def test_redaction_nested(self):
        self.assertEqual(g.redact({"x": ["canary-key"]}, "canary-key"), {"x": ["[REDACTED]"]})

    def test_literal_credential(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "env"
            p.write_text('export BIGMODEL_API_KEY="fake-test-credential"\n')
            p.chmod(0o600)
            self.assertEqual(g.read_credential_file(p), "fake-test-credential")

    def test_credential_expansion_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "env"
            p.write_text("export BIGMODEL_API_KEY='$(false)'\n")
            p.chmod(0o600)
            with self.assertRaises(AssertionError):
                g.read_credential_file(p)

    def test_insecure_credential_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "env"
            p.write_text("export BIGMODEL_API_KEY=fake\n")
            p.chmod(0o644)
            with self.assertRaises(AssertionError):
                g.read_credential_file(p)

    def test_no_redirect(self):
        with self.assertRaises(urllib.error.HTTPError):
            g.NoRedirect().redirect_request(
                urllib.request.Request(g.ENDPOINT), None, 302, "", {}, "https://example.com")

    def test_http_failure_no_retry_no_secret(self):
        import io
        opener = unittest.mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError(g.ENDPOINT, 401, "fake-secret", {},
                                  io.BytesIO(b'{"error":{"code":"401","message":"fake-secret"}}'))
        item = {"id": "x", "condition": "oracle", "request_sha256": "a", "body": {}}
        record = g.call(item, "fake-secret", opener)
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(record["http_status"], 401)
        self.assertNotIn("fake-secret", json.dumps(record))

    def test_success_redacts_and_preserves_answer(self):
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, n):
                return json.dumps({"model": g.MODEL, "id": "fake-secret", "choices": [
                    {"finish_reason": "stop", "message": {"content": "red fox"}}]}).encode()
        opener = unittest.mock.Mock()
        opener.open.return_value = Response()
        item = {"id": "x", "condition": "oracle", "request_sha256": "a", "body": {}}
        record = g.call(item, "fake-secret", opener)
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["raw_output"], "red fox")
        self.assertNotIn("fake-secret", json.dumps(record))


if __name__ == "__main__":
    unittest.main()
