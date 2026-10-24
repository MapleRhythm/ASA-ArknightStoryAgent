from copy import deepcopy
import io
import json
import unittest
import urllib.error
import urllib.request

import glm_annotation as app


class Tests(unittest.TestCase):
    def setUp(self):
        self.packet = app.protocol.make_packet("[E1] 甲：我在读书。 乙：我在种树。")
        self.item = {"id": "C1", "packet": app.protocol.packet_dict(self.packet)}
        self.label = {
            "packet_id": self.packet.packet_id, "claim_id": "C1", "verdict": "supported",
            "relevance": "answers", "subject": "甲", "relation": "正在", "object": "读书",
            "span_ids": ["T1"], "reason": "甲自述。",
        }

    def test_redaction_nested(self):
        self.assertEqual(app.redact({"k": ["x-secret-x"]}, "secret"), {"k": ["x-[REDACTED]-x"]})

    def test_http_error_never_saves_body_or_message(self):
        body = json.dumps({"error": {"code": "1000", "message": "fake-secret"}}).encode()
        error = urllib.error.HTTPError(app.ENDPOINT, 401, "fake-secret", {}, io.BytesIO(body))
        result = app.safe_http_error(error, "fake-secret")
        self.assertEqual(result["http_status"], 401)
        self.assertEqual(result["provider_error_code"], "1000")
        self.assertNotIn("fake-secret", json.dumps(result))
        self.assertTrue(result["response_body_not_saved_for_credential_safety"])

    def test_http_error_rejects_nonallowlisted_or_secret_code(self):
        for code in ("fake-secret", "error with spaces", "a" * 33):
            with self.subTest(code=code):
                body = json.dumps({"error": {"code": code}}).encode()
                error = urllib.error.HTTPError(app.ENDPOINT, 401, "", {}, io.BytesIO(body))
                self.assertIsNone(app.safe_http_error(error, "fake-secret")["provider_error_code"])

    def test_http_error_nonjson_body_withheld(self):
        error = urllib.error.HTTPError(
            app.ENDPOINT, 503, "", {}, io.BytesIO(b"<html>fake-secret</html>"))
        result = app.safe_http_error(error, "fake-secret")
        self.assertIsNone(result["provider_error_code"])
        self.assertNotIn("fake-secret", json.dumps(result))

    def test_redirect_refused(self):
        request = urllib.request.Request(app.ENDPOINT)
        with self.assertRaises(urllib.error.HTTPError):
            app.NoRedirect().redirect_request(
                request, io.BytesIO(), 307, "redirect", {}, "https://example.invalid/")

    def test_machine_label_never_gold(self):
        result = app.annotation(json.dumps(self.label), self.item)
        self.assertFalse(result["human_gold"])
        self.assertFalse(result["training_ready"])

    def test_exact_source_copied(self):
        result = app.annotation(json.dumps(self.label), self.item)
        self.assertEqual(result["source_refs"][0]["quote"], "甲：我在读书。 乙：我在种树。")

    def test_wrong_packet(self):
        self.label["packet_id"] = "other"
        with self.assertRaises(ValueError):
            app.annotation(json.dumps(self.label), self.item)

    def test_wrong_claim(self):
        self.label["claim_id"] = "C2"
        with self.assertRaises(ValueError):
            app.annotation(json.dumps(self.label), self.item)

    def test_invalid_span(self):
        self.label["span_ids"] = ["T999"]
        with self.assertRaises(ValueError):
            app.annotation(json.dumps(self.label), self.item)

    def test_missing_supported_reference(self):
        self.label["span_ids"] = []
        with self.assertRaises(ValueError):
            app.annotation(json.dumps(self.label), self.item)

    def test_insufficient_empty_reference(self):
        self.label.update(verdict="insufficient", span_ids=[])
        self.assertEqual(app.annotation(json.dumps(self.label), self.item)["source_refs"], [])

    def test_no_json_repairs(self):
        for text in ('"x":1}', '{"x":1,}', '```json\n{}\n```', '{"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                app.annotation(text, self.item)

    def test_expected_labels_excluded(self):
        row = {"packet": self.item["packet"], "question": "谁在读书？", "expected_label": "secret_label"}
        outgoing = app.messages(row, "甲在读书。", "C1")
        self.assertNotIn("secret_label", json.dumps(outgoing))

    def test_input_unchanged(self):
        before = deepcopy(self.item)
        app.annotation(json.dumps(self.label), self.item)
        self.assertEqual(before, self.item)


if __name__ == "__main__":
    unittest.main()
