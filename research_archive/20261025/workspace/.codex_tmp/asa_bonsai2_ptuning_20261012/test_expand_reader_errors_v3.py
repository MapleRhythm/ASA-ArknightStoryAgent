"""Offline tests; no network, keys, model loading, or GPU allocation."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

import annotation_protocol as p
import expand_reader_errors_v3 as e


def fixture(split="train", index=0):
    docs = [dict(evidence_id=f"E{i}", document_id=f"{split}/{index}/{i}",
                 text=f"片段{i}：甲把书交给乙，乙没有打开书。")
            for i in range(1, 21)]
    for d in docs:
        d["sha256"] = p.sha(d["text"])
    question = f"{split}{index}：谁把书交给了谁？"
    messages, prompt = p.prompt_for(question, "2/2", docs)
    return dict(
        id=f"{split}-{index}", split=split, documents=docs, evidence_count=20,
        truncated=False, question=question, round="2/2", source_families=[split],
        prompt=prompt, teacher_messages=messages, prompt_tokens=256,
        teacher_prompt_sha256=p.sha(p.compact(messages)),
        old_labels_used=False, target="OLD_TARGET_DO_NOT_COPY", semantic_review="OLD_REVIEW",
    )


class ExpansionTests(unittest.TestCase):
    def test_kind_counts_and_budgets(self):
        counts = Counter(k for i in range(200) for k in e.kind_schedule(i))
        self.assertEqual(counts, dict(speaker_direction=100, event_state_scope=100,
                                     minimal_complete=100, partial_support=100, evidence_boundary=200))
        for smoke in (False, True):
            for stage in ("questions", "annotation"):
                s = e.settings(stage, smoke).validate()
                self.assertEqual(s.endpoint, p.CODING_PLAN_ENDPOINT)
                self.assertEqual(s.review_fraction, 1)
                self.assertEqual(s.attempts, 2)
                self.assertEqual(s.workers, 2)
        self.assertEqual(sum(e.settings(s).max_api_calls for s in ("questions", "annotation")), 2280)
        self.assertEqual(sum(e.settings(s).max_total_tokens for s in ("questions", "annotation")), 20_000_000)

    def test_seeds_no_targets_no_eval_content_deterministic(self):
        train, val = [fixture(index=i) for i in range(4)], [fixture("val")]
        before = copy.deepcopy((train, val))
        a, protected, n = e.make_seeds(train, val, 4)
        b, _, _ = e.make_seeds(train, val, 4)
        self.assertEqual(a, b)
        self.assertEqual((train, val), before)
        self.assertEqual(n, 4)
        self.assertIn("val", protected)
        for row in a:
            self.assertNotIn("target", row)
            self.assertNotIn("semantic_review", row)
            self.assertNotIn("OLD_TARGET", p.compact(e.generation_messages(row)))
            self.assertNotIn("val0", p.compact(e.generation_messages(row)))
            p.check_row(row)
        with self.assertRaisesRegex(ValueError, "not enough"):
            e.make_seeds(train, val, 5)

    def test_family_document_leak_rejected(self):
        train, val = [fixture()], [fixture("val")]
        train[0]["source_families"] = ["val"]
        with self.assertRaisesRegex(ValueError, "protected"):
            e.make_seeds(train, val, 1)
        train = [fixture()]
        train[0]["documents"][0]["document_id"] = val[0]["documents"][0]["document_id"]
        with self.assertRaisesRegex(ValueError, "document"):
            e.make_seeds(train, val, 1)

    def test_order_schema_and_duplicate_question_rejected(self):
        requested = e.kind_schedule(0)
        value = {"questions": [{"question": f"角色甲的第{i}个情况是怎样的？", "kind": k}
                               for i, k in enumerate(requested)]}
        e.check_questions(value, requested)
        value["questions"].reverse()
        with self.assertRaisesRegex(ValueError, "order"):
            e.check_questions(value, requested)
        value["questions"].reverse()
        value["questions"][0]["answer"] = "do not accept answer"
        with self.assertRaisesRegex(ValueError, "fields"):
            e.check_questions(value, requested)

    def test_teacher_instructions_not_in_student_prompt(self):
        row = fixture()
        native = copy.deepcopy(row["teacher_messages"])
        augmented = e.annotation_messages(native)
        self.assertEqual(row["teacher_messages"], native)
        self.assertEqual(augmented[1], native[1])
        self.assertNotIn(e.ANNOTATION_EXTRA, row["prompt"])
        self.assertIn(e.ANNOTATION_EXTRA, augmented[0]["content"])
        self.assertIn(e.REVIEW_EXTRA, e.review_messages(row, {})[-1]["content"])

    def fake_run(self, warning=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        out, row = Path(temp.name), fixture()
        target = dict(next_action="answer_directly",
                      supported_facts=[dict(fact="甲把书交给乙。", evidence_ids=["E1"])])
        captured = []

        def fake(settings, messages):
            captured.append(messages)
            value = target if len(captured) == 1 else dict(
                label_supported=True, decision_appropriate=True,
                issues=["额外背景有问题"] if warning else [])
            return dict(id=f"r{len(captured)}", model="glm-5.3", usage=dict(total_tokens=100),
                        choices=[dict(finish_reason="stop", message=dict(content=p.compact(value)))])
        config = e.settings("annotation", True)
        config.request_interval = 0
        runner = e.Runner(out, [row], config, request_fn=fake)
        self.addCleanup(runner.ledger.close)
        state = runner.process(row)
        return runner, row, target, captured, state

    def test_review_warnings_quarantine_no_rewrite(self):
        runner, row, target, captured, state = self.fake_run(warning=True)
        self.assertEqual(state, "quarantined")
        self.assertEqual(runner.ledger.job(row["id"])[2]["label"]["parsed"], target)
        self.assertEqual(len(captured), 2)
        self.assertNotIn(e.ANNOTATION_EXTRA, row["teacher_messages"][0]["content"])

    def test_accept_resume_no_new_call_and_receipt_binding(self):
        runner, row, target, captured, state = self.fake_run()
        self.assertEqual(state, "accepted")
        self.assertEqual(runner.process(row), "accepted")
        self.assertEqual(len(captured), 2)
        job = runner.ledger.job(row["id"])[2]
        exported = dict(row, target=p.compact(target), annotation_response_id="r1",
                        semantic_review_sampled=True, semantic_review=job["review"]["parsed"])
        receipts = list(runner.ledger.receipts())
        path = runner.out / "api_receipts.jsonl"

        def write():
            path.write_text("".join(p.compact(r) + "\n" for r in receipts))
        write()
        e.verify_reviews(runner.out, [exported], smoke=True)
        receipts[0]["cache_key"] = p.sha(p.compact(dict(
            phase="annotate", messages=row["teacher_messages"],
            settings=e.settings("annotation", True).cache_settings())))
        write()
        with self.assertRaisesRegex(ValueError, "lineage"):
            e.verify_reviews(runner.out, [exported], smoke=True)

    def test_formal_and_smoke_fingerprints_bind_prompts_not_budgets(self):
        # Different datasets/directories keep smoke data isolated, while actual
        # request cache keys continue to bind model/endpoint/messages.
        self.assertEqual(e.settings("annotation", True).cache_settings(),
                         e.settings("annotation", False).cache_settings())
        self.assertLess(e.settings("annotation", True).max_api_calls,
                        e.settings("annotation", False).max_api_calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
