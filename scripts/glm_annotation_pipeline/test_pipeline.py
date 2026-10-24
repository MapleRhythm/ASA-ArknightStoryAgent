"""Offline tests: no credentials, paid requests, GPUs or production writes."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error

import pipeline as p
import start_pipeline as starter


def row(number=1, group="activities/family_a", question=None, round_value="2/2"):
    docs = [{"document_id": f"{group}/story#chunk-{i:04}", "evidence_id": f"E{i+1}",
             "text": f"原文段落{i}，完整保留。", "sha256": p.sha(f"原文段落{i}，完整保留。")}
            for i in range(20)]
    question = question or f"测试人物{number}做了什么？"
    messages, prompt = p.prompt_for(question, round_value, docs)
    return {"id": str(number), "question": question, "round": round_value, "documents": docs,
            "evidence_count": 20, "truncated": False, "source_families": [group],
            "teacher_messages": messages, "teacher_prompt_sha256": p.sha(p.compact(messages)),
            "prompt": prompt, "prompt_tokens": 500, "split": "train"}


def answer():
    return {"next_action": "answer_directly",
            "supported_facts": [{"fact": "事实概括。", "evidence_ids": ["E1"]}]}


def completion(value=None, finish="stop"):
    return {"id": "response_test", "model": "glm-5.3",
            "usage": {"total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
            "choices": [{"finish_reason": finish,
                         "message": {"content": p.compact(value or answer()),
                                     "reasoning_content": "HIDDEN_REASONING_NOT_TO_SAVE"}}]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.out = Path(self.temporary.name)
        self.rows = [row(1), dict(row(2, "activities/family_b"), split="val")]
        self.manifest()
        self.settings = p.Settings(workers=1, attempts=2, review_fraction=0,
                                   request_interval=0, backoff_seconds=0,
                                   max_api_calls=20, max_total_tokens=1_000_000)
        self.config = self.out / "config.json"
        p.atomic_json(self.config, vars(self.settings))

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        (self.out / "requests.jsonl").write_text("".join(p.compact(r) + "\n" for r in self.rows))
        p.atomic_json(self.out / "manifest.json", {
            "version": p.VERSION, "requests_sha256": p.file_sha(self.out / "requests.jsonl"),
            "max_output_tokens_reserved": 8192, "old_labels_used": False,
            "planned_train": 1, "planned_val": 1})

    def run_with(self, fn, limit=0, settings=None):
        runner = p.Runner(self.out, self.rows, settings or self.settings, fn)
        with redirect_stdout(io.StringIO()):
            runner.run(limit)
        return runner

    def test_full_prompt_and_hash_integrity(self):
        p.check_row(self.rows[0])
        self.rows[0]["documents"][0]["text"] += "变化"
        with self.assertRaises(ValueError):
            p.check_row(self.rows[0])

    def test_schema_three_actions_and_final_round(self):
        p.check_target(answer(), self.rows[0])
        p.check_target({"next_action": "abstain", "reason": "证据不足"}, self.rows[0])
        retrieve = {"next_action": "retrieve_more", "follow_up_hypothesis": {
            "question": "需要什么证据？", "query_type": "causality", "entities": ["人物"],
            "keywords": ["动机"], "expected_answer_type": "原因"}}
        with self.assertRaises(ValueError):
            p.check_target(retrieve, self.rows[0])
        p.check_target(retrieve, row(round_value="1/2"))
        bad = answer()
        bad["supported_facts"][0]["evidence_ids"] = ["E21"]
        with self.assertRaises(ValueError):
            p.check_target(bad, self.rows[0])

    def test_unknown_round_rejected(self):
        with self.assertRaises(ValueError):
            p.source_question({"question": "题", "round": "4/3", "evidence": []})

    def test_historical_three_round_phase_mapping_explicit(self):
        for value, expected in (("1/3", "1/2"), ("2/3", "1/2"), ("3/3", "2/2")):
            self.assertEqual(p.source_question({"question": "题", "round": value, "evidence": []})[2], value)
            self.assertEqual(p.annotation_round(value, "phase-to-two"), expected)
            self.assertEqual(p.annotation_round(value, "preserve"), value)

    def test_single_round_source_preserved_and_is_final(self):
        raw = {"question": "题", "round": "1/1", "evidence": [{"doc_id": "activities/a/story#chunk-0000"}]}
        self.assertEqual(p.source_question(raw)[2], "1/1")
        retrieve = {"next_action": "retrieve_more", "follow_up_hypothesis": {
            "question": "需要什么证据？", "query_type": "fact", "entities": [],
            "keywords": ["事实"], "expected_answer_type": "事实"}}
        with self.assertRaises(ValueError):
            p.check_target(retrieve, row(round_value="1/1"))

    def test_earlier_holdout_family_never_enters_train(self):
        rows = [row(i, f"activities/f{i}") for i in range(10)]
        splits, _ = p.choose_splits(rows, 7, 3, 42, heldout_families=["activities/f9"])
        self.assertIn("activities/f9", {f for r in splits["val"] for f in r["source_families"]})
        self.assertNotIn("activities/f9", {f for r in splits["train"] for f in r["source_families"]})

    def test_source_answers_never_used(self):
        raw = {"question": "题", "round": "1/2",
               "evidence": [{"doc_id": "activities/a/story#chunk-0000"}],
               "label": {"fact": "HIDDEN_GOLD"}}
        self.assertEqual(p.source_question(raw), ("题", ["activities/a/story#chunk-0000"], "1/2"))

    def test_exclusion_matches_path_component(self):
        self.assertTrue(p.forbidden("activities/act10d5/story#chunk-01", ["act10d5"]))
        self.assertFalse(p.forbidden("activities/act10d50/story#chunk-01", ["act10d5"]))

    def test_connected_components_use_all_families_and_questions(self):
        a, b, c = row(1), row(2, "activities/b"), row(3, "activities/c")
        a["source_families"].append("activities/b")
        c["question"] = b["question"]
        self.assertEqual(len(p.components([a, b, c])), 1)

    def test_split_zero_leakage(self):
        rows = [row(i, f"activities/f{i}") for i in range(10)]
        splits, _ = p.choose_splits(rows, 7, 3, 42)
        p.assert_split(splits)
        self.assertEqual((len(splits["train"]), len(splits["val"])), (7, 3))

    def test_impossible_split_fails(self):
        with self.assertRaises(ValueError):
            p.choose_splits([row(1), row(2)], 1, 1, 42)

    def test_success_export_and_no_hidden_reasoning(self):
        runner = self.run_with(lambda *_: completion())
        try:
            result = runner.export()
            self.assertTrue(result["complete"])
            self.assertEqual(result["labelled_examples"], 2)
            all_receipts = (self.out / "api_receipts.jsonl").read_text()
            self.assertNotIn("HIDDEN_REASONING", all_receipts)
            self.assertIn("fresh_GLM_API_full20", (self.out / "train.jsonl").read_text())
        finally:
            runner.ledger.close()

    def test_export_preserves_literal_chat_marker_in_evidence(self):
        r = self.rows[0]
        r["documents"][0]["text"] += "原文含字面量<|im_end|>，不是截断标记。"
        r["documents"][0]["sha256"] = p.sha(r["documents"][0]["text"])
        r["teacher_messages"], r["prompt"] = p.prompt_for(r["question"], r["round"], r["documents"])
        r["teacher_prompt_sha256"] = p.sha(p.compact(r["teacher_messages"]))
        runner = self.run_with(lambda *_: completion())
        try:
            runner.export()
            exported = json.loads((self.out / "train.json").read_text())[0]
            self.assertEqual(exported["conversations"][0]["value"], r["teacher_messages"][1]["content"])
            self.assertIn(r["documents"][-1]["text"], exported["conversations"][0]["value"])
        finally:
            runner.ledger.close()

    def test_export_only_keeps_previous_block_reason(self):
        def blocked(*_):
            raise p.APIError("insufficient_balance", 429, "1113", terminal=True)
        runner = self.run_with(blocked)
        runner.export()
        runner.ledger.close()
        runner = p.Runner(self.out, self.rows, self.settings, lambda *_: self.fail("export must not call API"))
        try:
            self.assertEqual(runner.export()["run_status"], "insufficient_balance")
        finally:
            runner.ledger.close()

    def test_resume_uses_no_additional_requests(self):
        runner = self.run_with(lambda *_: completion())
        runner.export()
        runner.ledger.close()
        runner = self.run_with(lambda *_: self.fail("paid API must not be called on resume"))
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 2)
        finally:
            runner.ledger.close()

    def test_limit_resumes_remaining_only(self):
        runner = self.run_with(lambda *_: completion(), limit=1)
        runner.ledger.close()
        runner = self.run_with(lambda *_: completion())
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 2)
        finally:
            runner.ledger.close()

    def test_transient_error_retried_and_counted(self):
        calls = []
        def request(*_):
            calls.append(1)
            if len(calls) == 1:
                raise p.APIError("rate_limit", 429, "1302", retryable=True)
            return completion()
        runner = self.run_with(request)
        try:
            self.assertEqual(len(calls), 3)
            self.assertEqual(runner.ledger.stats()["api_calls"], 3)
        finally:
            runner.ledger.close()

    def test_wrapped_answer_retried_without_unwrapping_or_repair(self):
        calls = []
        def request(_settings, messages):
            calls.append(messages)
            return completion({"answer": answer()}) if len(calls) == 1 else completion()
        runner = self.run_with(request)
        try:
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0], calls[1])
            receipts = runner.ledger.receipts()
            self.assertEqual(receipts[0]["validation_error"], "unknown_action")
            self.assertFalse(receipts[0]["accepted"])
            self.assertNotIn("parsed", receipts[0])
            self.assertTrue(runner.export()["complete"])
        finally:
            runner.ledger.close()

    def test_retry_limit_change_on_resume_does_not_relabel_accepted_jobs(self):
        runner = self.run_with(lambda *_: completion(), limit=1)
        fingerprint = runner.fingerprint(self.rows[0])
        runner.export()
        runner.ledger.close()
        changed = p.Settings(**dict(vars(self.settings), attempts=3))
        calls = []
        def request(*_):
            calls.append(1)
            return completion()
        runner = self.run_with(request, settings=changed)
        try:
            self.assertEqual(len(calls), 1)
            self.assertEqual(fingerprint, runner.fingerprint(self.rows[0]))
            self.assertEqual(runner.ledger.stats()["api_calls"], 2)
        finally:
            runner.ledger.close()

    def test_balance_stops_without_retry(self):
        calls = []
        def request(*_):
            calls.append(1)
            raise p.APIError("insufficient_balance", 429, "1113", terminal=True)
        runner = self.run_with(request)
        try:
            self.assertEqual(len(calls), 1)
            self.assertEqual(runner.reason, "insufficient_balance")
            self.assertFalse(runner.export()["complete"])
        finally:
            runner.ledger.close()

    def test_resume_after_balance_restored(self):
        def blocked(*_):
            raise p.APIError("insufficient_balance", 429, "1113", terminal=True)
        runner = self.run_with(blocked)
        runner.ledger.close()
        runner = self.run_with(lambda *_: completion())
        try:
            self.assertTrue(runner.export()["complete"])
            self.assertEqual(runner.ledger.stats()["api_calls"], 3)
        finally:
            runner.ledger.close()

    def test_hard_call_budget_across_resume(self):
        self.settings.max_api_calls = 1
        runner = self.run_with(lambda *_: completion())
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 1)
            self.assertEqual(runner.reason, "budget_limit")
        finally:
            runner.ledger.close()
        runner = self.run_with(lambda *_: self.fail("budget exceeded"))
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 1)
        finally:
            runner.ledger.close()

    def test_token_reservation_blocks_before_call(self):
        self.settings.max_total_tokens = 10
        runner = self.run_with(lambda *_: self.fail("insufficient token budget"))
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 0)
        finally:
            runner.ledger.close()

    def test_length_limit_output_quarantined_without_repair(self):
        runner = self.run_with(lambda *_: completion(finish="length"))
        try:
            self.assertEqual(runner.ledger.stats()["job_states"]["quarantined"], 2)
            self.assertEqual(runner.export()["labelled_examples"], 0)
        finally:
            runner.ledger.close()

    def test_invalid_json_bounded_retries(self):
        value = completion()
        value["choices"][0]["message"]["content"] = "not JSON"
        runner = self.run_with(lambda *_: value)
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 4)
            self.assertEqual(runner.export()["labelled_examples"], 0)
        finally:
            runner.ledger.close()

    def test_full_evidence_model_review(self):
        settings = p.Settings(**dict(vars(self.settings), review_fraction=1))
        inputs = []
        def request(_settings, messages):
            inputs.append(messages)
            if len(inputs) % 2 == 0:
                return completion({"label_supported": False, "decision_appropriate": True,
                                   "issues": ["错误引用"]})
            return completion()
        runner = self.run_with(request, settings=settings)
        try:
            self.assertEqual(runner.export()["labelled_examples"], 0)
            for doc in self.rows[0]["documents"]:
                self.assertIn(doc["text"], inputs[1][1]["content"])
        finally:
            runner.ledger.close()

    def test_protocol_settings_change_cannot_reuse_dataset(self):
        runner = self.run_with(lambda *_: completion())
        runner.ledger.close()
        changed = p.Settings(**dict(vars(self.settings), model="different-model"))
        runner = p.Runner(self.out, self.rows, changed, lambda *_: completion())
        try:
            with self.assertRaises(ValueError):
                runner.run()
        finally:
            runner.ledger.close()

    def test_changed_immutable_requests_refused(self):
        (self.out / "requests.jsonl").write_text("{}\n")
        with self.assertRaises(ValueError):
            p.load_requests(self.out)

    def test_only_one_process_per_dataset(self):
        with p.exclusive_run(self.out):
            with self.assertRaises(RuntimeError):
                with p.exclusive_run(self.out):
                    pass

    def test_billing_http_error_sanitized_and_terminal(self):
        payload = io.BytesIO(b'{"error":{"code":"1113","message":"secret-credential"}}')
        error = urllib.error.HTTPError(p.ENDPOINT, 429, "bad", {}, payload)
        opener = types.SimpleNamespace(open=lambda *_args, **_kw: (_ for _ in ()).throw(error))
        with patch.dict("os.environ", {"BIGMODEL_API_KEY": "secret-credential"}):
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaises(p.APIError) as caught:
                    p.transport(self.settings, [])
        self.assertTrue(caught.exception.terminal)
        self.assertNotIn("secret-credential", p.compact(caught.exception.record()))

    def test_official_endpoints_allowed_and_untrusted_refused(self):
        for url in (p.ENDPOINT, p.CODING_PLAN_ENDPOINT):
            self.assertEqual(p.Settings(endpoint=url).validate().endpoint, url)
        for url in ("https://other.invalid/chat", p.CODING_PLAN_ENDPOINT + "?key=secret",
                    p.CODING_PLAN_ENDPOINT + "#fragment",
                    p.CODING_PLAN_ENDPOINT.replace("open.bigmodel.cn", "open.bigmodel.cn:444"),
                    p.CODING_PLAN_ENDPOINT.replace("https://", "http://"),
                    p.CODING_PLAN_ENDPOINT.replace("https://", "https://user:secret@")):
            with self.assertRaises(ValueError):
                p.Settings(endpoint=url).validate()

    def test_endpoint_change_cannot_reuse_dataset_or_cache(self):
        runner = self.run_with(lambda *_: completion())
        original_key = runner.fingerprint(self.rows[0])
        runner.ledger.close()
        changed = p.Settings(**dict(vars(self.settings), endpoint=p.CODING_PLAN_ENDPOINT))
        runner = p.Runner(self.out, self.rows, changed, lambda *_: self.fail("must not call"))
        try:
            self.assertNotEqual(original_key, runner.fingerprint(self.rows[0]))
            with self.assertRaises(ValueError):
                runner.run()
        finally:
            runner.ledger.close()

    def test_progress_counts_pending_and_unstarted_separately(self):
        runner = self.run_with(lambda *_: completion(), limit=1)
        try:
            status = json.loads((self.out / "status.json").read_text())
            self.assertEqual(status["total_tasks"], 2)
            self.assertEqual(status["accepted"], 1)
            self.assertEqual(status["quarantined"], 0)
            self.assertEqual(status["remaining_tasks"], 1)
            self.assertEqual(status["unstarted_tasks"], 1)
            self.assertEqual(status["teacher_endpoint"], p.ENDPOINT)
        finally:
            runner.ledger.close()

    def test_fork_inputs_preserves_questions_not_labels_or_ledger(self):
        runner = self.run_with(lambda *_: completion())
        runner.export()
        runner.ledger.close()
        destination = self.out / "forked"
        result = p.fork_inputs(self.out, destination)
        self.assertEqual(p.load_requests(destination), self.rows)
        self.assertEqual(p.file_sha(self.out / "requests.jsonl"),
                         p.file_sha(destination / "requests.jsonl"))
        self.assertEqual(result["labelled_examples"], 0)
        self.assertFalse(result["complete"])
        self.assertNotIn("teacher_endpoint", result)
        self.assertNotIn("new_api_calls", result)
        self.assertFalse((destination / "ledger.sqlite3").exists())
        self.assertFalse((destination / "train.json").exists())
        self.assertEqual(json.loads((self.out / "manifest.json").read_text())["labelled_examples"], 2)

    def test_fork_inputs_refuses_existing_destination(self):
        destination = self.out / "existing"
        destination.mkdir()
        sentinel = destination / "keep.txt"
        sentinel.write_text("user data")
        with self.assertRaises(FileExistsError):
            p.fork_inputs(self.out, destination)
        self.assertEqual(sentinel.read_text(), "user data")

    def test_fork_inputs_refuses_changed_source_before_creating_directory(self):
        (self.out / "requests.jsonl").write_text("{}\n")
        destination = self.out / "forked"
        with self.assertRaises(ValueError):
            p.fork_inputs(self.out, destination)
        self.assertFalse(destination.exists())

    def test_coding_plan_export_and_receipt_identify_endpoint(self):
        settings = p.Settings(**dict(vars(self.settings), endpoint=p.CODING_PLAN_ENDPOINT))
        runner = self.run_with(lambda *_: completion(), settings=settings)
        try:
            manifest = runner.export()
            self.assertEqual(manifest["labels_provider"], "bigmodel_coding_plan_api")
            self.assertEqual(manifest["teacher_endpoint"], p.CODING_PLAN_ENDPOINT)
            self.assertTrue(all(r["endpoint"] == p.CODING_PLAN_ENDPOINT
                                for r in runner.ledger.receipts()))
        finally:
            runner.ledger.close()

    def test_multithread_budget_not_overshot(self):
        settings = p.Settings(**dict(vars(self.settings), workers=2, max_api_calls=1))
        runner = self.run_with(lambda *_: completion(), settings=settings)
        try:
            self.assertEqual(runner.ledger.stats()["api_calls"], 1)
        finally:
            runner.ledger.close()

    def test_interrupted_call_budget_kept(self):
        ledger = p.Ledger(self.out / "interrupted.sqlite3")
        ledger.reserve("task", "annotate", "fingerprint", 1000, self.settings)
        ledger.close()
        ledger = p.Ledger(self.out / "interrupted.sqlite3")
        try:
            self.assertEqual(ledger.stats()["accounted_tokens"], 1000)
            self.assertEqual(ledger.db.execute("SELECT state FROM calls").fetchone()[0],
                             "interrupted_unknown_billing")
        finally:
            ledger.close()

    def test_prepare_full20_and_old_label_absence(self):
        docs = []
        tasks = []
        for group_index in range(4):
            group = f"activities/g{group_index}"
            for i in range(24):
                docs.append({"id": f"{group}/chapter#chunk-{i:04}",
                             "clean_text": f"剧情段落{group_index}-{i}的全文。"})
            tasks.append({"task_id": str(group_index), "question": f"人物{group_index}的动机？",
                          "round": "1/2" if group_index % 2 else "2/2",
                          "evidence": [{"doc_id": d["id"]} for d in docs[-24:-20]],
                          "label": {"answer": "HIDDEN_GOLD"}})
        doc_path, src_path = self.out / "docs.jsonl", self.out / "source.json"
        doc_path.write_text("".join(p.compact(d) + "\n" for d in docs))
        src_path.write_text(p.compact(tasks))
        fake_tokenizer = types.SimpleNamespace(from_file=lambda _: types.SimpleNamespace(
            encode=lambda text, **_: types.SimpleNamespace(ids=[0] * len(text))))
        args = argparse.Namespace(out=str(self.out / "prepared"), tokenizer="unused",
            documents=str(doc_path), source=[str(src_path)], exclude_family=["act10d5", "act10mini"],
            single_family=True, seed=42, max_output_tokens=8192, max_context_tokens=32768,
            train=3, val=1, allow_fewer=False)
        with patch.dict("sys.modules", {"tokenizers": types.SimpleNamespace(Tokenizer=fake_tokenizer)}):
            with redirect_stdout(io.StringIO()):
                p.prepare(args)
        requests = p.load_requests(args.out)
        self.assertEqual(len(requests), 4)
        for r in requests:
            self.assertEqual(len(r["documents"]), 20)
            self.assertNotIn("HIDDEN_GOLD", p.compact(r["teacher_messages"]))

    def test_launcher_balance_failure_never_detaches_batch(self):
        def preflight(*_args, **_kwargs):
            p.atomic_json(self.out / "status.json", {"stage": "blocked", "stop_reason": "insufficient_balance"})
            return types.SimpleNamespace(returncode=2)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run", side_effect=preflight), patch("subprocess.Popen") as child:
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                starter.launch(args)
        child.assert_not_called()
        self.assertEqual(json.loads((self.out / "launch.json").read_text())["stage"], "blocked")

    def test_launcher_success_detaches_batch(self):
        def preflight(*_args, **_kwargs):
            manifest = json.loads((self.out / "manifest.json").read_text())
            manifest.update(labelled_examples=1, complete=False)
            p.atomic_json(self.out / "manifest.json", manifest)
            return types.SimpleNamespace(returncode=0)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run", side_effect=preflight), patch("subprocess.Popen", return_value=types.SimpleNamespace(pid=123)) as child:
            with redirect_stdout(io.StringIO()):
                starter.launch(args)
        self.assertEqual(child.call_count, 1)
        self.assertTrue(child.call_args.kwargs["start_new_session"])
        self.assertEqual(json.loads((self.out / "launch.json").read_text())["worker_pid"], 123)

    def test_launcher_preflight_only_does_not_detach(self):
        def preflight(*_args, **_kwargs):
            manifest = json.loads((self.out / "manifest.json").read_text())
            manifest.update(labelled_examples=1, complete=False)
            p.atomic_json(self.out / "manifest.json", manifest)
            return types.SimpleNamespace(returncode=0)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=True)
        with patch("subprocess.run", side_effect=preflight), patch("subprocess.Popen") as child:
            with redirect_stdout(io.StringIO()):
                starter.launch(args)
        child.assert_not_called()

    def test_launcher_old_labels_do_not_mask_failed_new_preflight(self):
        manifest = json.loads((self.out / "manifest.json").read_text())
        manifest.update(labelled_examples=1, complete=False)
        p.atomic_json(self.out / "manifest.json", manifest)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run", return_value=types.SimpleNamespace(returncode=0)):
            with patch("subprocess.Popen") as child:
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    starter.launch(args)
        child.assert_not_called()
        receipt = json.loads((self.out / "launch.json").read_text())
        self.assertEqual(receipt["stage"], "blocked")
        self.assertEqual(receipt["new_labels_in_preflight"], 0)

    def test_launcher_completed_batch_makes_no_calls(self):
        runner = self.run_with(lambda *_: completion())
        runner.export()
        runner.ledger.close()
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run") as preflight, patch("subprocess.Popen") as child:
            with redirect_stdout(io.StringIO()):
                starter.launch(args)
        preflight.assert_not_called()
        child.assert_not_called()
        self.assertEqual(json.loads((self.out / "launch.json").read_text())["stage"], "already_completed")

    def test_launcher_completed_batch_still_rejects_changed_backend(self):
        runner = self.run_with(lambda *_: completion())
        runner.export()
        runner.ledger.close()
        changed = dict(vars(self.settings), endpoint=p.CODING_PLAN_ENDPOINT)
        p.atomic_json(self.config, changed)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run") as preflight, patch("subprocess.Popen") as child:
            with self.assertRaises(ValueError):
                starter.launch(args)
        preflight.assert_not_called()
        child.assert_not_called()

    def test_launcher_pins_preflight_and_worker_to_same_config(self):
        commands = []
        def preflight(command, **_kwargs):
            commands.append(command)
            p.atomic_json(self.config, dict(vars(self.settings), endpoint=p.CODING_PLAN_ENDPOINT))
            manifest = json.loads((self.out / "manifest.json").read_text())
            manifest.update(labelled_examples=1, complete=False)
            p.atomic_json(self.out / "manifest.json", manifest)
            return types.SimpleNamespace(returncode=0)
        args = argparse.Namespace(out=str(self.out), config=str(self.config), preflight_only=False)
        with patch("subprocess.run", side_effect=preflight):
            with patch("subprocess.Popen", return_value=types.SimpleNamespace(pid=123)) as child:
                with redirect_stdout(io.StringIO()):
                    starter.launch(args)
        preflight_config = commands[0][commands[0].index("--config") + 1]
        worker_command = child.call_args.args[0]
        self.assertEqual(worker_command[worker_command.index("--config") + 1], preflight_config)
        self.assertNotEqual(preflight_config, str(self.config))
        self.assertEqual(json.loads(Path(preflight_config).read_text()), vars(self.settings))


if __name__ == "__main__":
    unittest.main()
