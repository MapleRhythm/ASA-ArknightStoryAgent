import unittest
from protocol_helpers import messages_from_prompt, answer_after_thinking


class ProtocolTests(unittest.TestCase):
    def test_conversion_preserves_body(self):
        body = '题目\n证据：<think>只是被引用的文本</think>\n'
        prompt = ("<|im_start|>system\n只输出JSON。<|im_end|>\n"
                  "<|im_start|>user\n" + body + "<|im_end|>\n"
                  "<|im_start|>assistant\n{")
        self.assertEqual(messages_from_prompt(prompt)[1]["content"], body)

    def test_bad_chatml(self):
        with self.assertRaises(ValueError):
            messages_from_prompt("<|im_start|>assistant\n{}")

    def test_normal_thinking(self):
        final, reasoning = answer_after_thinking('分析\n</think>\n{"answer":"甲"}', True)
        self.assertEqual(final, '{"answer":"甲"}')
        self.assertEqual(reasoning, "分析\n")

    def test_reasoning_json_not_final(self):
        with self.assertRaises(ValueError):
            answer_after_thinking('草稿 {"answer":"甲"}', True)

    def test_empty_final(self):
        with self.assertRaises(ValueError):
            answer_after_thinking("分析</think>  ", True)

    def test_ambiguous_terminator(self):
        with self.assertRaises(ValueError):
            answer_after_thinking("分析</think>答案</think>", True)

    def test_nothink_markers_rejected(self):
        with self.assertRaises(ValueError):
            answer_after_thinking("<think>分析</think>答案", False)

    def test_plain_final(self):
        self.assertEqual(answer_after_thinking(' {"a":1} ', False), ('{"a":1}', ""))


if __name__ == "__main__":
    unittest.main()
