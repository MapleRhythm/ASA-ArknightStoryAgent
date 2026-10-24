"""No-network safety and data-lineage fixtures for the v2 expansion."""
import json
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as p
import targeted_pt_v2 as t
import test_audit_fresh_subset as fixtures


class TargetedTests(unittest.TestCase):
    def test_question_schema(self):
        value = {"questions": [{"question": "甲是谁说话的对象？", "kind": k}
                               for k in ("speaker_identity", "timeline_scope", "missing_evidence")]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            t.check_questions(value, ["speaker_identity", "timeline_scope", "missing_evidence"])
        value["questions"][1]["question"] = "甲在事件前后有什么变化？"
        value["questions"][2]["question"] = "甲是否后来又返回了故乡？"
        t.check_questions(value, ["speaker_identity", "timeline_scope", "missing_evidence"])
        value["questions"][0]["answer"] = "不应进入题生成产物"
        with self.assertRaisesRegex(ValueError, "fields"):
            t.check_questions(value, ["speaker_identity", "timeline_scope", "missing_evidence"])

    def test_coding_plan_only_bounded_settings(self):
        for stage in ("questions", "annotation"):
            settings = t.config(stage).validate()
            self.assertEqual(settings.endpoint, p.CODING_PLAN_ENDPOINT)
            self.assertEqual(settings.review_fraction, 1.0)
            self.assertEqual(settings.attempts, 2)
            self.assertLessEqual(settings.workers, 2)

    def test_second_pass_review_does_not_change_native_prompt(self):
        row = {"question": "谁救出了博士？", "round": "2/2",
               "documents": [{"evidence_id": "E1", "text": "阿米娅救出了博士。"}]}
        target = {"next_action": "answer_directly", "supported_facts": []}
        original = p.prompt_for(row["question"], row["round"], row["documents"])
        messages = t.targeted_review(row, target)
        self.assertIn(t.REVIEW_EXTRA, messages[-1]["content"])
        self.assertEqual(original, p.prompt_for(row["question"], row["round"], row["documents"]))
        augmented = t.annotation_messages(original[0])
        self.assertIn(t.ANNOTATION_EXTRA, augmented[0]["content"])
        self.assertNotIn(t.ANNOTATION_EXTRA, original[0][0]["content"])
        self.assertEqual(original[0][1], augmented[1])

    def test_prepare_copies_no_old_target_and_protects_val(self):
        fixture = fixtures.AuditTests(methodName="test_no_overwrite")
        fixture.setUp()
        try:
            fixture.change_rows("train", lambda row: row.update(prompt_tokens=128))
            fixture.build()
            out = fixture.root / "expansion"
            plan = t.prepare(fixture.out, out, 1)
            t.verify_plan(out)
            seed = t.read_rows(out / "questions/requests.jsonl")[0]
            self.assertNotIn("target", seed)
            self.assertNotIn("semantic_review", seed)
            self.assertIn("val", plan["protected_families"])
            self.assertFalse(set(seed["source_families"]) & set(plan["protected_families"]))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                t.prepare(fixture.out, out, 1)
            (out / "questions/requests.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "packs changed"):
                t.verify_plan(out)
        finally:
            fixture.tearDown()

    def test_credential_literal_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.env"
            path.write_text('export BIGMODEL_API_KEY="test-only-literal"\n')
            path.chmod(0o600)
            self.assertEqual(t.read_credential(path), "test-only-literal")
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "permissions"):
                t.read_credential(path)

    def test_missing_review_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "api_receipts.jsonl").write_text("")
            row = dict(id="x", annotation_response_id="response-x", question="谁救出了博士？", round="2/2",
                       documents=[dict(evidence_id="E1", text="阿米娅救出了博士。")],
                       teacher_prompt_sha256="hash", semantic_review_sampled=True,
                       semantic_review={"label_supported": True, "decision_appropriate": True, "issues": []},
                       target=json.dumps({"next_action": "abstain", "reason": "缺证"}))
            with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                t.verify_reviews(path, [row])

    def test_warning_is_quarantined_without_rewriting_label(self):
        fixture = fixtures.AuditTests(methodName="test_no_overwrite")
        fixture.setUp()
        prior_review = p.review_messages
        runner = None
        try:
            row = fixture.rows["train"][0].copy()
            row["teacher_messages"], row["prompt"] = p.prompt_for(
                row["question"], row["round"], row["documents"])
            native = row["teacher_messages"][0]["content"]
            target = {"next_action": "answer_directly", "supported_facts": [
                {"fact": "阿米娅救出博士。", "evidence_ids": ["E1"]}]}
            captured = []

            def fake(settings, messages):
                captured.append(messages)
                value = ({"label_supported": True, "decision_appropriate": True,
                          "issues": ["有轻度引用缺口，不能用于训练。"]}
                         if len(captured) == 2 else target)
                return {"id": f"response-{len(captured)}", "model": "glm-5.3",
                        "usage": {"total_tokens": 100},
                        "choices": [{"finish_reason": "stop",
                                     "message": {"content": json.dumps(value, ensure_ascii=False)}}]}

            p.review_messages = t.targeted_review
            out = fixture.root / "fake-annotation"
            out.mkdir()
            settings = t.config("annotation")
            settings.request_interval = 0
            runner = t.TargetRunner(out, [row], settings, request_fn=fake)
            self.assertEqual(runner.process(row), "quarantined")
            result = runner.ledger.job(row["id"])[2]
            self.assertEqual(result["label"]["parsed"], target)
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[0][0]["content"], native + t.ANNOTATION_EXTRA)
            self.assertEqual(row["teacher_messages"][0]["content"], native)
        finally:
            p.review_messages = prior_review
            if runner:
                runner.ledger.close()
            fixture.tearDown()

    def test_review_receipt_key_binds_teacher_only_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            target = {"next_action": "abstain", "reason": "当前证据不足。"}
            row = dict(id="x", annotation_response_id="a1", question="是谁救出了博士？",
                       round="2/2", documents=[{"evidence_id": "E1", "text": "博士醒来了。"}],
                       teacher_prompt_sha256="native-hash", semantic_review_sampled=True,
                       semantic_review={"label_supported": True, "decision_appropriate": True, "issues": []},
                       target=p.compact(target))
            settings = t.config("annotation").cache_settings()
            native, _ = p.prompt_for(row["question"], row["round"], row["documents"])
            key = lambda phase, messages: p.sha(p.compact(
                {"phase": phase, "messages": messages, "settings": settings}))
            label = dict(task_id="x", response_id="a1", phase="annotate", accepted=True,
                         cache_key=key("annotate", t.annotation_messages(native)))
            review = dict(task_id="x", phase="review", accepted=True,
                          cache_key=key("review", t.targeted_review(row, target)),
                          endpoint=p.CODING_PLAN_ENDPOINT, requested_model="glm-5.3",
                          teacher_prompt_sha256="native-hash", parsed=row["semantic_review"],
                          content=p.compact(row["semantic_review"]))

            def write():
                (source / "api_receipts.jsonl").write_text(
                    p.compact(label) + "\n" + p.compact(review) + "\n")

            write()
            t.verify_reviews(source, [row])
            label["cache_key"] = key("annotate", native)
            write()
            with self.assertRaisesRegex(ValueError, "teacher-only"):
                t.verify_reviews(source, [row])


if __name__ == "__main__":
    unittest.main()
