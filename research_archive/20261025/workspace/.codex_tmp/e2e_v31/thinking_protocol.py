"""Strict extraction of the answer after the model's declared reasoning phase."""


def answer_after_thinking(raw, thinking):
    if not thinking:
        if "<think>" in raw or "</think>" in raw:
            raise ValueError("unexpected thinking markers in non-thinking output")
        return raw, 0
    if raw.count("</think>") != 1:
        raise ValueError("missing or ambiguous thinking terminator")
    reasoning, answer = raw.split("</think>", 1)
    if "<think>" in answer or not answer.strip():
        raise ValueError("missing or malformed final answer")
    return answer.strip(), len(reasoning)
