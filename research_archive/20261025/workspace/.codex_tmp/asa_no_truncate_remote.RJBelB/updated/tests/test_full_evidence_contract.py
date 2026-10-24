"""Offline tests of the real module, with only retrieval imports excluded."""
import ast
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/goldenglow/inference/cpu_pipeline.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
tree.body = [
    node for node in tree.body
    if not (
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("goldenglow")
    )
]
module = types.ModuleType("full_evidence_contract_module")
module.__file__ = str(SOURCE)
sys.modules[module.__name__] = module
exec(compile(tree, str(SOURCE), "exec"), module.__dict__)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def hypothesis():
    return module.HypothesisDocument(
        question="薇卡是谁？为什么保留她的东西？", intent="character_relation",
        query_type="fact", entities=["薇卡", "真理"], keywords=["薇卡"],
        expected_answer_type="short_text", dialogue_context="",
    )


def evidence():
    body = "\n".join(f"这是前置背景第{i}段。" for i in range(100))
    return [
        {
            "document": {"id": f"doc-{i}", "clean_text": body + f"\n完整结尾{i}：最好的朋友。"},
            "evidence_chain_text": "不应替换完整正文的链。",
        }
        for i in range(32)
    ]


def pipeline(generator=None):
    return module.CPUInferencePipeline(
        retriever=Mock(), generator=generator or Mock(),
        query_config=types.SimpleNamespace(max_retrieval_queries=0),
        prompt_evidence_top_k=20, prompt_evidence_require_full_documents=True,
        # Even a stale caller supplying old limits must not truncate full mode.
        prompt_evidence_max_chars_per_doc=1000,
        prompt_conclusion_evidence_max_total_chars=12000,
        enable_crag_refinement=True, conclusion_prompt_mode="minimal",
    )


class FullEvidenceTests(unittest.TestCase):
    def test_full_flag_overrides_all_character_limits(self):
        obj = pipeline()
        self.assertIsNone(obj.prompt_evidence_max_chars_per_doc)
        self.assertIsNone(obj.prompt_conclusion_evidence_max_total_chars)

    def test_first_twenty_bypass_dedupe_refinement_and_chain_replacement(self):
        obj, items = pipeline(), evidence()
        obj.refine_evidence_strips = Mock(side_effect=AssertionError("Must not refine"))
        selected = obj.prepare_prompt_evidence(hypothesis().question, hypothesis(), items)
        self.assertEqual(len(selected), 20)
        self.assertEqual([x["document"]["id"] for x in selected], [f"doc-{i}" for i in range(20)])
        for old, new in zip(items, selected):
            self.assertEqual(old["document"]["clean_text"], new["document"]["clean_text"])
            self.assertNotIn("evidence_chain_text", new)
        self.assertIn("evidence_chain_text", items[0])

    def test_all_renderers_keep_long_tails_and_total_over_twelve_thousand(self):
        obj, hyp = pipeline(), hypothesis()
        selected = obj.prepare_prompt_evidence(hyp.question, hyp, evidence())
        for mode in ("minimal", "full"):
            prompt = module.build_conclusion_prompt(
                hyp.question, hyp, selected, [], 1, 2, 20,
                prompt_evidence=selected, evidence_max_chars_per_doc=None,
                evidence_max_total_chars=None, prompt_mode=mode,
            )
            self.assertGreater(len(prompt), 12000)
            for item in selected:
                self.assertIn(norm(item["document"]["clean_text"]), norm(prompt))
        direct = module.build_answer_prompt(
            hyp.question, hyp, selected, prompt_evidence_top_k=20,
            prompt_evidence=selected, evidence_max_chars_per_doc=None,
            evidence_max_total_chars=None,
        )
        for item in selected:
            self.assertIn(norm(item["document"]["clean_text"]), norm(direct))

    def test_followup_preserves_full_evidence_only_when_requested(self):
        obj, hyp = pipeline(), hypothesis()
        selected = obj.prepare_prompt_evidence(hyp.question, hyp, evidence())
        previous = types.SimpleNamespace(next_action="retrieve_more")
        kwargs = dict(
            question=hyp.question, current_hypothesis=hyp, evidence=selected,
            unresolved_points=[], retrieval_trace=[], previous_conclusion=previous,
            current_round=2, max_retrieval_rounds=2, prompt_evidence_top_k=20,
            prompt_evidence=selected,
        )
        full = module.build_follow_up_hypothesis_prompt(**kwargs, require_full_prompt_evidence=True)
        legacy = module.build_follow_up_hypothesis_prompt(**kwargs)
        for item in selected:
            self.assertIn(norm(item["document"]["clean_text"]), norm(full))
        self.assertNotIn("完整结尾19", legacy)

    def test_context_overflow_is_not_converted_to_abstention(self):
        generator = Mock()
        generator.max_tokens = 512
        generator.generate.side_effect = module.PromptContextOverflowError("No evidence was truncated")
        obj, hyp = pipeline(generator), hypothesis()
        with self.assertRaises(module.PromptContextOverflowError):
            obj.generate_conclusion(hyp.question, hyp, evidence(), [], 2)
        self.assertEqual(generator.generate.call_count, 1)
        self.assertIn("完整结尾19", obj.prompt_snapshots[-1]["prompt"])

    def test_vllm_sends_exact_checked_tokens_and_rejects_overflow(self):
        runner = module.VllmRunner(base_model_path=Path("."), max_model_len=10, max_tokens=2)
        engine = Mock()
        engine.get_tokenizer.return_value.encode.return_value = list(range(8))
        engine.generate.return_value = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="ok")])]
        runner._llm = engine
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            runner.generate("complete prompt")
            sent = engine.generate.call_args.args[0]
            self.assertEqual(sent, [{"prompt_token_ids": list(range(8))}])
            with self.assertRaises(module.PromptContextOverflowError):
                runner.generate("complete prompt", max_tokens=3)
            self.assertEqual(engine.generate.call_count, 1)

    def test_snapshot_reset_between_questions(self):
        obj = pipeline()
        obj.record_full_prompt("test", "previous", [])
        obj.build_hypothesis = Mock(side_effect=RuntimeError("stop before retrieval"))
        with self.assertRaises(RuntimeError):
            obj.run("new question")
        self.assertEqual(obj.prompt_snapshots, [])

    def test_active_gpu_config_enables_full_twenty(self):
        cfg = json.loads((ROOT / "configs/runtime_inference_gpu.json").read_text())["inference"]
        self.assertEqual(cfg["prompt_evidence_top_k"], 20)
        self.assertTrue(cfg["prompt_evidence_require_full_documents"])
        self.assertFalse(cfg["enable_crag_refinement"])
        self.assertEqual(cfg["prompt_evidence_max_chars_per_doc"], 0)
        self.assertEqual(cfg["prompt_conclusion_evidence_max_total_chars"], 0)


if __name__ == "__main__":
    unittest.main()
