"""Logic tests for apply_verifier.py on synthetic data only (no story text)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import apply_verifier as av  # noqa: E402

DOCS = {
    "E1": "Alice: I will open the gate tonight.\nBob: Then I guard the bridge.\nnarration without label line",
    "E2": "Carol: The storm came from the east.",
}


def answer(*facts):
    return {"next_action": "answer_directly",
            "facts": [{"text": t, "evidence_ids": ["E1"]} for t in facts], "valid_json": True}


def bind(idx, quote, ev, cl, role="speaker", eid="E1", conf=0.95, span=None, core=None):
    v = {"position": 1, "fact_index": idx, "verdict": "binding_error", "binding_role": role,
         "confidence": conf, "anchor": {"evidence_id": eid, "quote": quote,
                                        "evidence_role_value": ev, "claim_role_value": cl}}
    if span:
        v["error_span"] = span
    if core is not None:
        v["addresses_question"] = core
    return v


# ---------------------------------------------------------------- parsing

def test_parse_valid_and_invalid_json_agree():
    raw = json.dumps({"next_action": "answer_directly", "supported_facts": [
        {"fact": "a \"q\" b", "evidence_ids": ["E1", "E2"]}, {"fact": "c", "evidence_ids": []}]})
    ok = av.parse_answer(raw)
    broken = av.parse_answer(raw[:-1])  # drop closing brace -> invalid JSON
    assert ok["valid_json"] and not broken["valid_json"]
    assert ok["facts"] == broken["facts"]
    assert broken["facts"][0]["text"] == 'a "q" b'
    assert broken["next_action"] == "answer_directly"


# ---------------------------------------------------------------- anchors

def test_anchor_quote_must_exist():
    v = bind(1, "I will burn the gate", "Alice", "Bob")
    assert av.validate_anchor(v, DOCS, {}) == (False, "anchor_quote_not_in_evidence")


def test_anchor_punctuation_insensitive_and_label_consistent():
    v = bind(1, "I will open the gate tonight", "Alice", "Bob")
    assert av.validate_anchor(v, DOCS, {})[0]


def test_speaker_anchor_rejected_when_label_contradicts_verifier():
    # verifier says the evidence speaker is Bob, but the quoted line is labeled Alice
    v = bind(1, "I will open the gate tonight", "Bob", "Carol")
    assert av.validate_anchor(v, DOCS, {}) == (False, "label_contradicts_evidence_role")


def test_identical_roles_are_not_a_binding_error_including_aliases():
    v = bind(1, "I will open the gate tonight", "Alice", "A.")
    assert av.validate_anchor(v, DOCS, {"Alice": {"A."}, "A.": {"Alice"}}) == \
        (False, "binding_roles_identical")


def test_unlabeled_line_policies():
    # quote sits on an unlabeled line whose nearest label above is Bob
    v = bind(1, "narration without label", "Bob", "Alice")
    assert av.validate_anchor(v, DOCS, {}, "inherit") == (True, "ok_unlabeled_inherited")
    assert av.validate_anchor(v, DOCS, {}, "strict") == (False, "unlabeled_speaker_unverified")
    w = bind(1, "narration without label", "Carol", "Alice")
    assert av.validate_anchor(w, DOCS, {}, "inherit") == (False, "unlabeled_speaker_unverified")
    assert av.validate_anchor(w, DOCS, {}, "permissive") == (True, "ok_unlabeled_permissive")


def test_non_speaker_role_needs_only_quote_and_distinct_values():
    v = bind(1, "narration without label", "Carol", "Alice", role="addressee")
    assert av.validate_anchor(v, DOCS, {}, "strict") == (True, "ok")


def test_missing_evidence_id():
    v = bind(1, "x", "Alice", "Bob", eid="E9")
    assert av.validate_anchor(v, DOCS, {}) == (False, "anchor_evidence_missing")


# ---------------------------------------------------------------- policy

def test_unsupported_and_binding_error_deleted_supported_kept():
    a = answer("core fact", "wrong speaker fact", "extra")
    vs = {1: {"verdict": "supported"},
          2: bind(2, "Then I guard the bridge", "Bob", "Alice")}
    r = av.apply_policy(a, vs, DOCS)
    assert r["status"] == "pruned"
    assert [f["text"] for f in r["facts"]] == ["core fact", "extra"]
    assert r["deleted"][0]["fact_index"] == 2


def test_below_threshold_and_bad_anchor_are_kept_and_reported():
    a = answer("f1", "f2")
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", conf=0.5),
          2: bind(2, "not in passage", "Bob", "Alice")}
    r = av.apply_policy(a, vs, DOCS, threshold=0.9)
    assert r["status"] == "unchanged"
    assert {x["reason"] for x in r["rejected"]} == {"below_threshold", "anchor_quote_not_in_evidence"}


def test_no_anchor_ablation_deletes_unanchored():
    a = answer("f1", "f2")
    vs = {2: bind(2, "not in passage", "Bob", "Alice")}
    r = av.apply_policy(a, vs, DOCS, require_anchor=False)
    assert [f["text"] for f in r["facts"]] == ["f1"]


def test_all_deleted_routes_to_regenerate():
    a = answer("only fact")
    r = av.apply_policy(a, {1: bind(1, "Then I guard the bridge", "Bob", "Alice")}, DOCS)
    assert r["status"] == "regenerate"


def test_core_lost_routes_to_regenerate_even_if_facts_remain():
    a = answer("core", "background")
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", core=True),
          2: {"verdict": "supported", "addresses_question": False}}
    assert av.apply_policy(a, vs, DOCS)["status"] == "regenerate"


def test_core_survives_when_another_core_fact_kept():
    a = answer("core-a", "core-b")
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", core=True),
          2: {"verdict": "supported", "addresses_question": True}}
    assert av.apply_policy(a, vs, DOCS)["status"] == "pruned"


def test_clause_mode_excises_span_and_keeps_rest():
    text = "Alice guards the bridge, and the gate opens tonight"
    a = answer(text)
    span = [0, text.index(",") + 1]
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", span=span, core=True)}
    r = av.apply_policy(a, vs, DOCS, mode="clause")
    assert r["status"] == "clause_repaired"
    assert r["facts"][0]["text"] == "and the gate opens tonight"


def test_clause_substitution_is_role_swap_only():
    text = "Alice guards the bridge tonight"
    a = answer(text, "other")
    v = bind(1, "Then I guard the bridge", "Bob", "Alice")
    v.update(error_text="Alice guards", repair="substitute")
    r = av.apply_policy(a, {1: v}, DOCS, mode="clause")
    assert r["status"] == "clause_repaired"
    assert r["facts"][0]["text"] == "Bob guards the bridge tonight"
    w = bind(1, "Then I guard the bridge", "Bob", "Dave")
    w.update(error_text="Alice guards", repair="substitute")
    r = av.apply_policy(a, {1: w}, DOCS, mode="clause")
    assert r["status"] == "pruned" and [f["text"] for f in r["facts"]] == ["other"]
    assert r["rejected"] == [{"fact_index": 1, "reason": "substitution_role_not_in_error_text"}]


def test_error_text_locates_span_and_must_be_unique():
    a = answer("Alice and Alice met; Bob left")
    v = bind(1, "Then I guard the bridge", "Bob", "Alice", core=True)
    v["error_text"] = "; Bob left"
    r = av.apply_policy(a, {1: v}, DOCS, mode="clause")
    assert r["facts"][0]["text"] == "Alice and Alice met"
    v["error_text"] = "Alice"
    with pytest.raises(av.ContractError):
        av.apply_policy(a, {1: v}, DOCS, mode="clause")


def test_clause_mode_too_short_remainder_falls_back_to_delete():
    text = "Alice guards the bridge."
    a = answer(text, "other")
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", span=[0, len(text) - 2])}
    r = av.apply_policy(a, vs, DOCS, mode="clause", min_clause_chars=8)
    assert r["status"] == "pruned" and [f["text"] for f in r["facts"]] == ["other"]


def test_exact_duplicates_removed_without_verdict():
    a = answer("same fact.", "same fact", "other")
    r = av.apply_policy(a, {}, DOCS)
    assert r["status"] == "pruned"
    assert [f["text"] for f in r["facts"]] == ["same fact.", "other"]
    assert r["duplicates"] == [{"fact_index": 2, "dup_of": 1}]


def test_verdict_for_missing_fact_raises():
    with pytest.raises(av.ContractError):
        av.apply_policy(answer("only"), {3: {"verdict": "supported"}}, DOCS)


def test_span_beyond_text_raises():
    a = answer("short")
    vs = {1: bind(1, "Then I guard the bridge", "Bob", "Alice", span=[0, 99])}
    with pytest.raises(av.ContractError):
        av.apply_policy(a, vs, DOCS, mode="clause")


# ---------------------------------------------------------------- verdict file

def _write(tmp_path, rows):
    p = tmp_path / "v.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_load_verdicts_rejects_unknown_label_and_duplicates(tmp_path):
    with pytest.raises(av.ContractError):
        av.load_verdicts(_write(tmp_path, [{"position": 1, "fact_index": 1, "verdict": "maybe"}]))
    row = {"position": 1, "fact_index": 1, "verdict": "supported"}
    with pytest.raises(av.ContractError):
        av.load_verdicts(_write(tmp_path, [row, row]))
    with pytest.raises(av.ContractError):
        av.load_verdicts(_write(tmp_path, [{**row, "confidence": 1.5}]))


# ---------------------------------------------------------------- counterfactual

def _ann():
    return {"version": "fake", "n_questions": 10, "baseline_complete": 3, "unannotated_complete": 2,
            "cases": [
                # fixable by deletion
                {"position": 1, "historical": "partial", "required_points": ["p"],
                 "facts": [{"label": "supported", "covers": ["p"]},
                           {"label": "binding_error", "covers": []}]},
                # core entangled: only clause repair fixes it
                {"position": 2, "historical": "partial", "required_points": ["p"],
                 "facts": [{"label": "binding_error", "covers": [], "covers_after_repair": ["p"]},
                           {"label": "supported", "covers": []}]},
                # omission: never fixed
                {"position": 3, "historical": "partial", "required_points": ["missing"],
                 "facts": [{"label": "binding_error", "covers": []}]},
                # historically complete, two coverers of p -> redundancy
                {"position": 4, "historical": "complete", "required_points": ["p"],
                 "facts": [{"label": "supported", "covers": ["p"]},
                           {"label": "supported", "covers": ["p"]}]},
                # boundary success in fact mode only; duplicate wrong fact
                {"position": 5, "historical": "partial", "boundary": ["fact"], "required_points": ["p"],
                 "facts": [{"label": "supported", "covers": ["p"]},
                           {"label": "unsupported", "covers": []},
                           {"label": "unsupported", "covers": [], "dup_of": 2}]},
            ]}


def test_oracle_counts():
    res = av.counterfactual(_ann(), [0.0], [1.0], [1])
    f, c = res["oracle"]["fact"], res["oracle"]["clause"]
    assert f["complete_positions"] == [1, 4] and f["boundary_positions"] == [5]
    assert f["not_complete_positions"] == [2, 3]
    assert c["complete_positions"] == [1, 2, 4, 5] and c["boundary_positions"] == []
    assert f["total_complete_low"] == 3 + 1 and f["total_complete_high"] == 3 + 2
    assert c["total_complete_low"] == 3 + 3
    assert res["status"].endswith("not_model_accuracy")


def test_probability_independent_errors_and_duplicates():
    case = _ann()["cases"][4]
    # one unique wrong fact (duplicate excluded), one correct coverer
    assert av._case_probability(case, 0.1, 0.8, "fact") == pytest.approx(0.8 * 0.9)
    red = _ann()["cases"][3]
    assert av._case_probability(red, 0.1, 1.0, "fact") == pytest.approx(1 - 0.1 ** 2)


def test_grid_unannotated_decay():
    res = av.counterfactual(_ann(), [0.5], [1.0], [2])
    row = res["grid"][0]
    annotated = 0.5 + 0.75 + 0.5  # cases 1, 4, (5 boundary in fact mode -> high only)
    assert row["fact"]["expected_complete_low"] == pytest.approx(0.5 + 0.75 + 2 * 0.25)
    assert row["fact"]["expected_complete_high"] == pytest.approx(annotated + 2 * 0.25)


def test_bad_label_raises():
    ann = _ann()
    ann["cases"][0]["facts"][0]["label"] = "fine"
    with pytest.raises(av.ContractError):
        av.counterfactual(ann, [0.0], [1.0], [1])


# ---------------------------------------------------------------- cli end-to-end

def test_cli_apply_end_to_end(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"position": 1, "documents": [
        {"evidence_id": k, "text": v} for k, v in DOCS.items()]}) + "\n")
    gen = tmp_path / "gen.jsonl"
    raw = json.dumps({"next_action": "answer_directly", "supported_facts": [
        {"fact": "core", "evidence_ids": ["E1"]}, {"fact": "bad", "evidence_ids": ["E1"]}]})
    gen.write_text(json.dumps({"id": "x", "raw_text": raw}) + "\n")
    ver = _write(tmp_path, [bind(2, "Then I guard the bridge", "Bob", "Alice")])
    out, rep = tmp_path / "out.jsonl", tmp_path / "rep.json"
    assert av.main(["apply", "--generated", str(gen), "--tasks", str(tasks), "--verdicts", ver,
                    "--out", str(out), "--report", str(rep)]) == 0
    row = json.loads(out.read_text())
    assert row["status"] == "pruned"
    assert [f["fact"] for f in json.loads(row["raw_text"])["supported_facts"]] == ["core"]
    report = json.loads(rep.read_text())
    assert report["deleted_by_role"] == {"speaker": 1} and report["facts_after"] == 1
