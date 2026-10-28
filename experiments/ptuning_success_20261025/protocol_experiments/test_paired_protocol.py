"""Synthetic protocol tests only. No model gain can be inferred from these fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import paired_protocol as p


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registered = p.load(Path(__file__).with_name("preregistered_scope_ab.json"))
        self.config = copy.deepcopy(self.registered)
        self.config["synthetic"] = True
        # Synthetic fixtures carry no private slot file; real bindings are tested separately.
        self.config["frozen_inputs"]["per_task_slots"]["sha256"] = None
        self.docs = [dict(evidence_id=f"E{i}", text=f"synthetic evidence unit {i}",
                          sha256=p.digest(f"synthetic evidence unit {i}")) for i in range(1, 21)]
        prompt = ("<|im_start|>user\n" + "\n".join(d["text"] for d in self.docs) +
                  "<|im_end|>\n<|im_start|>assistant\n")
        self.task = dict(id="synthetic-case", question="synthetic question", question_group="synthetic-family",
                         round="1/2", documents=self.docs, bonsai_prompt=prompt,
                         bonsai_prompt_sha256=p.digest(prompt))
        p.write(self.root / "config.json", self.config)
        p.write_rows(self.root / "tasks.jsonl", [self.task])
        p.prepare(self.root / "tasks.jsonl", self.root / "config.json", self.root / "run")
        self.requests = p.rows(self.root / "run/requests.private.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def values(self):
        # Both answers synthetically identical; deliberately malformed JSON must not force content failure.
        return [dict(request_id=r["request_id"], prompt_sha256=r["prompt_sha256"],
                     config_sha256=r["config_sha256"], task_identity_sha256=r["task_identity_sha256"],
                     generation=r["generation"], model_bundle=r["model_bundle"], attempt=1,
                     evidence_count=20, input_truncated=False, status="ok", raw_text="synthetic readable text",
                     output_token_ids=[900, 248046], finish_reason="stop") for r in self.requests]

    def review(self, values):
        p.write_rows(self.root / "results.jsonl", values)
        p.blind(self.root / "run", self.root / "results.jsonl", self.root / "review")
        forms = p.rows(self.root / "review/judgments.template.jsonl")
        for form in forms:
            form.update(reviewer_id="synthetic-reviewer", adjudicated=True, all20_checked=True,
                        necessary_slots=["synthetic necessary slot"], action_appropriate=True,
                        sufficient=True, source_eids=["E1"], note="synthetic positive fixture")
        p.write_rows(self.root / "judgments.jsonl", forms)
        return forms

    def test_identical_synthetic_answers_and_blinding(self):
        self.review(self.values())
        result = p.analyze(self.root / "run", self.root / "review", self.root / "judgments.jsonl")
        self.assertEqual(result["paired_difference_B_minus_A"], 0)
        self.assertTrue(result["synthetic"])
        self.assertFalse(result["model_gain_claim_permitted"])
        for packet in p.rows(self.root / "review/review_packets.private.jsonl"):
            self.assertNotIn("arm", packet)
            self.assertNotIn("prompt", packet)
            self.assertEqual(len(packet["documents"]), 20)
        self.assertEqual(result["diagnostics"]["A"]["invalid_json_separate_from_content"], 1)
        self.assertEqual(result["diagnostics"]["A"]["action_unparsed"], 1)

    def test_action_distribution_is_counted(self):
        values = self.values()
        for value in values:
            value["raw_text"] = json.dumps({"next_action": "abstain", "reason": "synthetic"})
        self.review(values)
        result = p.analyze(self.root / "run", self.root / "review", self.root / "judgments.jsonl")
        self.assertEqual(result["diagnostics"]["B"]["action_abstain"], 1)
        self.assertNotIn("invalid_json_separate_from_content", result["diagnostics"]["B"])
        self.assertEqual(result["diagnostics"]["A"]["whole_answer_correct"], 1)

    def test_missing_is_retained_failure(self):
        values = [v for r, v in zip(self.requests, self.values()) if r["arm"] == "A"]
        self.review(values)
        result = p.analyze(self.root / "run", self.root / "review", self.root / "judgments.jsonl")
        self.assertEqual(result["diagnostics"]["B"]["missing_output"], 1)
        self.assertEqual(result["diagnostics"]["B"]["denominator"], 1)
        self.assertEqual(result["transitions"]["lost"], 1)

    def test_duplicate_attempt_refused(self):
        values = self.values()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.check_results(self.requests, values + [values[0]])

    def test_full20_and_prompt_integrity(self):
        for request in self.requests:
            self.assertTrue(all(d["text"] in request["prompt"] for d in self.docs))
        bad = copy.deepcopy(self.task)
        bad["documents"].pop()
        with self.assertRaisesRegex(ValueError, "20"):
            p.task_identity(bad)

    def test_decoder_and_after_eos_mismatch_refused(self):
        values = self.values()
        values[0]["output_token_ids"] = [248046, 900, 248046]
        with self.assertRaisesRegex(ValueError, "after EOS"):
            p.check_results(self.requests, values)
        values = self.values()
        values[0]["generation"] = dict(values[0]["generation"], max_new_tokens=5)
        with self.assertRaisesRegex(ValueError, "decoding"):
            p.check_results(self.requests, values)

    def test_registered_arm_and_rubric_hashes_match_files(self):
        rule = p.check_frozen_inputs(self.config, None)
        self.assertEqual(rule, p.digest((Path(__file__).with_name("SLOT_RUBRIC.md")).read_text()))
        arm = p.load(p.EXPERIMENT_ROOT / "answer_mechanisms/prompt_arm.json")
        self.assertEqual(self.registered["arms"][1]["user_instruction_suffix"], arm["instruction"])
        self.assertFalse(self.registered["synthetic"])
        self.assertIsNotNone(self.registered["frozen_inputs"]["per_task_slots"]["sha256"])

    def test_changed_arm_text_or_rubric_hash_refused(self):
        bad = copy.deepcopy(self.config)
        bad["arms"][1]["user_instruction_suffix"] += "x"
        with self.assertRaisesRegex(ValueError, "prompt_arm"):
            p.check_frozen_inputs(bad, None)
        bad = copy.deepcopy(self.config)
        bad["frozen_inputs"]["slot_rubric"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "slot_rubric"):
            p.check_frozen_inputs(bad, None)
        with self.assertRaisesRegex(ValueError, "slot file required"):
            p.check_frozen_inputs(self.registered, None)
        real = dict(self.config, synthetic=False)
        del real["frozen_inputs"]
        with self.assertRaisesRegex(ValueError, "frozen_inputs"):
            p.check_frozen_inputs(real, None)

    def test_slot_rows_must_carry_rubric_hash(self):
        rule = p.check_frozen_inputs(self.config, None)
        slot = dict(task_id="synthetic-case", necessary_slots=["s"], slot_rubric_sha256="0" * 64)
        p.write_rows(self.root / "slots_bad.jsonl", [slot])
        with self.assertRaisesRegex(ValueError, "different SLOT_RUBRIC"):
            p.prepare(self.root / "tasks.jsonl", self.root / "config.json", self.root / "r1",
                      self.root / "slots_bad.jsonl")
        p.write_rows(self.root / "slots.jsonl", [dict(slot, slot_rubric_sha256=rule)])
        p.prepare(self.root / "tasks.jsonl", self.root / "config.json", self.root / "r2",
                  self.root / "slots.jsonl")
        self.assertEqual(p.load(self.root / "r2/manifest.json")["slot_rule_sha256"], rule)

    def test_input_layer_panel_is_separate_bundle(self):
        p.prepare(self.root / "tasks.jsonl", self.root / "config.json", self.root / "r3",
                  panel="input_layer")
        request = p.rows(self.root / "r3/requests.private.jsonl")[0]
        self.assertEqual(request["model_bundle"]["adapter"], "input_layer_curated_final_step1117")
        self.assertNotEqual(request["config_sha256"], self.requests[0]["config_sha256"])

    def test_new_error_types_and_unnecessary_supported_counted(self):
        forms = self.review(self.values())
        key = p.rows(self.root / "review/blind_key.private.jsonl")
        arm_b = next(k["review_id"] for k in key if k["arm"] == "B")
        next(f for f in forms if f["review_id"] == arm_b).update(
            unnecessary_supported=["fact 4"],
            material_errors=[dict(scope="extra", type="scene", claim_locator="fact 5",
                                  explanation="synthetic")])
        p.write_rows(self.root / "judgments3.jsonl", forms)
        result = p.analyze(self.root / "run", self.root / "review", self.root / "judgments3.jsonl")
        self.assertEqual(result["diagnostics"]["B"]["extra_scene"], 1)
        self.assertEqual(result["diagnostics"]["B"]["with_unnecessary_supported"], 1)
        self.assertEqual(result["transitions"]["lost"], 1)

    def test_omission_is_separate_from_extra_error(self):
        forms = self.review(self.values())
        key = p.rows(self.root / "review/blind_key.private.jsonl")
        arm_b = next(k["review_id"] for k in key if k["arm"] == "B")
        next(f for f in forms if f["review_id"] == arm_b).update(
            sufficient=False, omitted_slots=["synthetic necessary slot"])
        p.write_rows(self.root / "judgments2.jsonl", forms)
        result = p.analyze(self.root / "run", self.root / "review", self.root / "judgments2.jsonl")
        self.assertEqual(result["diagnostics"]["B"]["with_omission"], 1)
        self.assertEqual(result["diagnostics"]["B"]["with_extra_error"], 0)
        self.assertEqual(result["transitions"]["lost"], 1)


if __name__ == "__main__":
    unittest.main()
