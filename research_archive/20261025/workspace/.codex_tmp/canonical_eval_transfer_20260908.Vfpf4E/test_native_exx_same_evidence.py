import copy

from build_native_exx_same_evidence import transform


def test_only_native_protocol_changes_while_evidence_and_round_stay_fixed():
    source = {
        "id": "id", "question": "问题", "hypothesis": {"intent": "plot_fact"},
        "current_round": 2, "max_retrieval_rounds": 3,
        "evidence_brief": "[E1] 完整原文", "prompt_evidence": [{"text": "完整原文"}],
        "evidence_id_to_corpus_id": {"E1": "doc"}, "evidence_sha256": "unchanged",
        "packing_trace": [{"status": "included"}],
    }
    original = copy.deepcopy(source)
    captured = {}
    def render_user(**kwargs):
        captured.update(kwargs)
        return "USER " + kwargs["evidence_text"]
    def render_chat(system, user, *, assistant_prefix):
        assert assistant_prefix == ""
        return system + user
    row = transform(source, render_user, render_chat, "SYSTEM")
    assert source == original
    for key in ("question", "hypothesis", "current_round", "max_retrieval_rounds",
                "evidence_brief", "prompt_evidence", "evidence_id_to_corpus_id", "evidence_sha256", "packing_trace"):
        assert row[key] == original[key]
    assert captured["round_value"] == "2/3"
    assert captured["evidence_text"] == original["evidence_brief"]
    assert row["grounding_mode"] == "evidence_id"
