import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("api_e2e", HERE / "run_deepseek_e2e.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
BASELINE = HERE.parent / "asa_e2e20_20260914.9KSGhY"


class TransportTests(unittest.TestCase):
    def test_all_baseline_prompts_preserve_system_user(self):
        count = 0
        for line in (BASELINE / "attempt1/calls.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["event"] != "generation_end":
                continue
            messages, prefill = mod.messages_from_prompt(row["prompt"])
            parsed = mod.CHATML.findall(row["prompt"])
            self.assertEqual(messages, [
                {"role": role, "content": text} for role, text in parsed if role != "assistant"
            ])
            self.assertTrue(prefill.lstrip().startswith("{"))
            self.assertTrue(all(x["content"] in row["prompt"] for x in messages))
            count += 1
        self.assertEqual(count, 56)

    def test_reject_unexpected_prompt(self):
        with self.assertRaises(ValueError):
            mod.messages_from_prompt("not ChatML")

    def test_frozen_inputs(self):
        self.assertEqual(mod.sha(BASELINE / "questions.txt"), mod.EXPECTED["questions"])
        self.assertEqual(mod.sha(BASELINE / "PROTOCOL.md"), mod.EXPECTED["rubric"])


if __name__ == "__main__":
    unittest.main()
