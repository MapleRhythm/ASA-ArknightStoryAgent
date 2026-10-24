import copy

from build_source_fidelity_matrix import RULES, transform


def test_intervention_keeps_evidence_question_and_prefix():
    original = {
        "id": "sample", "system": "S", "user": "U", "assistant_prefix": "{",
        "question": "Q", "hypothesis": {"question": "Q"},
        "evidence_brief": "[E1]完整证据", "evidence_sha256": "original",
        "prompt_evidence": [{"document": {"clean_text": "完整证据"}}],
    }
    before = copy.deepcopy(original)
    seen = {}
    def render(system, user, **kwargs):
        seen.update(system=system, user=user, **kwargs)
        return system + user + kwargs["assistant_prefix"]
    result = transform(original, render)
    assert original == before
    for key in ("question", "hypothesis", "evidence_brief", "evidence_sha256", "prompt_evidence"):
        assert result[key] == original[key]
    assert seen["assistant_prefix"] == "{"
    assert seen["user"] == "U\n" + RULES
    assert result["training_ready"] is False
