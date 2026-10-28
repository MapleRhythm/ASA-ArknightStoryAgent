"""Abstract fixtures only: no story text, no model, no data edits."""
import copy
import json
import unittest

import binding_pairs as b

RUBRIC = "r" * 64


def docs(prefix="family-a"):
    texts = ["甲对乙说：你要接班。", "无关段落2。", "丙在别处。"] + [f"无关段落{i}。" for i in range(4, 21)]
    return [dict(evidence_id=f"E{i}", document_id=f"{prefix}#chunk-{i:04d}", text=t, sha256=b.sha(t))
            for i, t in enumerate(texts, 1)]


def pair(pid="p1", **change):
    correct, wrong = "甲要乙接班。", "甲要丙接班。"
    target = json.dumps(dict(next_action="answer_directly",
                             supported_facts=[dict(fact="甲劝阻乙。", evidence_ids=["E1"]),
                                              dict(fact=correct, evidence_ids=["E1"])]), ensure_ascii=False)
    value = dict(pair_id=pid, source_row_id="row1", ready_for_training=False,
                 source_review=dict(label_supported=True, decision_appropriate=True, issues=[]),
                 documents=docs(), anchors=[dict(evidence_id="E1", char_range=[0, 4])],
                 hazard="pronoun_referent", perturbation="addressee_swap", role="addressee",
                 swap_from="乙", swap_to="丙", correct_clause=correct, wrong_clause=wrong,
                 item_kind="uncued_binding",
                 training_item=dict(question="甲为什么劝阻乙？", round="1/2", necessary_slots=["劝阻原因"],
                                    slot_rubric_sha256=RUBRIC, target=target),
                 reviews=[dict(reviewer_id=r, all20_checked=True, correct_supported=True, wrong_false=True)
                          for r in ("rev1", "rev2")])
    value.update(change)
    return value


def errors(pairs, excluded=(set(), set()), base_rows=1117):
    return [e["error"] for e in b.validate(pairs, *excluded, RUBRIC, base_rows)["errors"]]


class BindingPairTest(unittest.TestCase):
    def test_valid_pair_aggregate_has_no_text(self):
        result = b.validate([pair()], set(), set(), RUBRIC, 1117)
        self.assertTrue(result["valid"], result["errors"])
        self.assertNotIn("甲", json.dumps(result, ensure_ascii=False))
        self.assertFalse(result["ready_for_training"])

    def test_never_ready_for_training(self):
        self.assertIn("ready_for_training must be false; merging needs user authorization",
                      errors([pair(ready_for_training=True)]))

    def test_unreviewed_source_or_single_reviewer_refused(self):
        self.assertIn("source row lacks a passing recorded review",
                      errors([pair(source_review=dict(label_supported=True, decision_appropriate=True,
                                                      issues=["needs_review"]))]))
        one = pair()
        one["reviews"] = one["reviews"][:1] * 2
        self.assertIn("two distinct passing reviews required", errors([one]))

    def test_evaluation_overlap_refused(self):
        value = pair()
        self.assertIn("passage shared with an evaluation set",
                      errors([value], ({value["documents"][0]["sha256"]}, set())))
        self.assertIn("source family shared with an evaluation set",
                      errors([value], (set(), {"family-a"})))

    def test_substitution_must_be_single_present_entity(self):
        self.assertIn("substitution must be a single entity replacement",
                      errors([pair(wrong_clause="丁要丙接班。")]))
        self.assertIn("swapped-in entity must be a confusable present in the same passages",
                      errors([pair(swap_to="戊", wrong_clause="甲要戊接班。")]))

    def test_wrong_clause_never_a_target_fact(self):
        bad = pair()
        target = json.loads(bad["training_item"]["target"])
        target["supported_facts"].append(dict(fact=bad["wrong_clause"], evidence_ids=["E1"]))
        bad["training_item"]["target"] = json.dumps(target, ensure_ascii=False)
        self.assertIn("a wrong clause is a fact of a training target", errors([bad]))

    def test_attribution_flip_substring_is_not_a_false_alarm(self):
        value = pair(perturbation="attribution_flip", hazard="allegation_vs_fact",
                     correct_clause="甲称乙偷了钱。", wrong_clause="乙偷了钱。")
        target = json.loads(value["training_item"]["target"])
        target["supported_facts"][1]["fact"] = "甲称乙偷了钱。"
        value["training_item"]["target"] = json.dumps(target, ensure_ascii=False)
        self.assertEqual(errors([value]), [])

    def test_slots_must_use_frozen_rubric_and_twin_same_passages(self):
        stale = pair()
        stale["training_item"]["slot_rubric_sha256"] = "0" * 64
        self.assertIn("training item slots not written under the frozen SLOT_RUBRIC", errors([stale]))
        other = docs("family-b")
        other[5].update(text="另一段。", sha256=b.sha("另一段。"))
        twin = pair("p2", item_kind="contrast_twin", twin_of="p1", documents=other)
        self.assertIn("contrast twin must reference a pair on the identical passages",
                      errors([pair(), twin]))

    def test_share_and_family_caps(self):
        many = [pair(f"p{i}") for i in range(4)]
        found = errors(many, base_rows=20)
        self.assertIn("trainable items exceed 10% of base rows", found)
        self.assertTrue(any("source family used by 4 pairs" in e for e in found))
        probes = [pair(f"q{i}", item_kind="probe_only", training_item=None, documents=docs(f"f{i}"))
                  for i in range(3)]
        self.assertEqual(errors(probes, base_rows=20), [])


if __name__ == "__main__":
    unittest.main()
