"""Stdlib-only tests of the ORIGINAL decode control flow and of the parity/probe logic.

AST extraction runs the sealed greedy_generate without importing torch/triton.
Fakes record calls; no claim about 27B logits or model quality follows.
Run: PYTHONDONTWRITEBYTECODE=1 python3 test_decode_contract.py --source <sealed source dir>
"""
import argparse
import ast
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = None
HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parity = load("decode_parity")
probe = load("binding_probe")


class FakeTensor:
    def __init__(self, values, device="cpu"):
        self.values, self.device = values, device
        self.shape = (1, len(values[0]))
        self.ndim = 2


class FakeLogits:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, key):
        return self

    def argmax(self):
        return self.token


class Controller:
    mode = None

    @contextmanager
    def activate(self, prefill=True):
        old, self.mode = self.mode, "prefill" if prefill else "decode"
        try:
            yield
        finally:
            self.mode = old


class Adapter:
    length = 3

    def __init__(self):
        self.prepares = 0

    def prepare(self, embedded):
        self.prepares += 1
        return ("prepared", embedded)


class Encoder:
    length = 3

    def __init__(self):
        self.injects = 0

    def inject(self, embedded):
        self.injects += 1
        return ("injected", embedded)


class Model:
    def __init__(self, tokens, controller):
        self.tokens, self.controller, self.calls = iter(tokens), controller, []
        self.model = SimpleNamespace(embed_tokens=lambda ids: ids)
        self.cache = object()

    def eval(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append(dict(mode=self.controller.mode, **kwargs))
        return SimpleNamespace(logits=FakeLogits(next(self.tokens)), past_key_values=self.cache)


def original_greedy(path):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "greedy_generate")
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(tensor=FakeTensor)}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["greedy_generate"]


class DecodeContractTests(unittest.TestCase):
    def setUp(self):
        self.generate = original_greedy(SOURCE / "ptuning_v2_hybrid.py")
        self.adapter, self.control = Adapter(), Controller()
        self.ids = FakeTensor([[4, 5, 6, 7]])

    def run_model(self, tokens, maximum=4, eos=(99,)):
        model = Model(tokens, self.control)
        got = self.generate(model, self.adapter, self.control, self.ids, maximum, set(eos))
        return model, got

    def test_prefill_once_decode_never_prepares_or_reinserts_prefix(self):
        model, got = self.run_model([10, 11, 12, 13])
        self.assertEqual(got, [10, 11, 12, 13])
        self.assertEqual(self.adapter.prepares, 1)
        self.assertEqual([c["mode"] for c in model.calls], ["prefill", "decode", "decode", "decode"])
        self.assertIn("inputs_embeds", model.calls[0])
        for call, token in zip(model.calls[1:], got):
            self.assertNotIn("inputs_embeds", call)
            self.assertNotIn("position_ids", call)
            self.assertNotIn("attention_mask", call)
            self.assertEqual(call["input_ids"].values, [[token]])
            self.assertIs(call["past_key_values"], model.cache)
        self.assertIsNone(self.control.mode)

    def test_eos_is_retained_and_stops_before_decode(self):
        for eos in (98, 99):
            self.adapter, self.control = Adapter(), Controller()
            model, got = self.run_model([eos], eos=(98, 99))
            self.assertEqual(got, [eos])
            self.assertEqual(len(model.calls), 1)

    def test_budget_emits_exact_number_without_extra_forward(self):
        model, got = self.run_model([10], maximum=1)
        self.assertEqual(got, [10])
        self.assertEqual(len(model.calls), 1)

    def test_budget_refuses_input_truncation(self):
        model = Model([], self.control)
        with self.assertRaisesRegex(ValueError, "Context budget"):
            self.generate(model, self.adapter, self.control, self.ids, 4, {99}, context_limit=10)
        self.assertEqual(model.calls, [])

    def test_empty_input_rejected(self):
        model = Model([], self.control)
        with self.assertRaises(ValueError):
            self.generate(model, self.adapter, self.control, FakeTensor([[]]), 1, {99})

    def test_input_layer_adapter_runs_unchanged_greedy_with_null_controller(self):
        encoder, control = Encoder(), parity.NullController()
        adapter = parity.InputPrompt(encoder)
        model = Model([10, 11, 99], control)
        got = self.generate(model, adapter, control, self.ids, 8, {99})
        self.assertEqual(got, [10, 11, 99])
        self.assertEqual(encoder.injects, 1)
        self.assertEqual(adapter.length, 3)
        self.assertEqual([c["mode"] for c in model.calls], [None, None, None])

    def test_sealed_sources_match_recorded_identity(self):
        for name in ("backend.py", "ptuning_v2_hybrid.py", "official/tokenizer.json"):
            path = SOURCE / name
            if path.is_file():
                self.assertEqual(parity.sha(path), parity.COMMON[f"source/{name}"], name)


def trace(total, n, deep, override=None):
    rows = []
    for k in range(n):
        past = 0 if k == 0 else total + k - 1
        cur = total if k == 0 else 1
        rows.append(dict(past_seq_length=past, current_tokens=cur, inputs_embeds=k == 0,
                         controller_mode=("prefill" if k == 0 else "decode") if deep else None,
                         attention_mask_shape=[1, past + cur], attention_mask_all_ones=True,
                         caller_supplied_positions=True,
                         effective_rope_positions=dict(minimum=past, maximum=past + cur - 1)))
    for (k, key), value in (override or {}).items():
        rows[k][key] = value
    return rows


def record(top, values=(5.0, 4.0, 1.0), digest="h"):
    return dict(top_ids=list(top), top_values=list(values), sha256=digest, finite=True)


class TraceAndVerdictTests(unittest.TestCase):
    def test_clean_trace_has_no_problems(self):
        for deep in (True, False):
            self.assertEqual(parity.check_trace(trace(20, 4, deep), 20, 4, deep), [])

    def test_trace_violations_are_named(self):
        cases = {
            ("reinserted prefix", 2, "inputs_embeds"): True,
            ("cache double counted", 2, "past_seq_length"): 40,
            ("rope", 1, "effective_rope_positions"): dict(minimum=0, maximum=0),
            ("mode", 0, "controller_mode"): None,
            ("mask", 3, "attention_mask_shape"): [1, 5],
        }
        for (label, k, key), value in cases.items():
            rows = trace(20, 4, True, {(k, key): value})
            self.assertTrue(parity.check_trace(rows, 20, 4, True), label)
        self.assertTrue(parity.check_trace(trace(20, 3, True), 20, 4, True))

    def test_emitted_and_stop_contracts(self):
        recs = [record((10, 11, 12)), record((99, 11, 12))]
        self.assertEqual(parity.check_emitted(recs, [10, 99]), [])
        self.assertTrue(parity.check_emitted(recs, [11, 99]))
        tied = [record((11, 10, 12), (5.0, 5.0, 1.0))]
        self.assertEqual(parity.check_emitted(tied, [10]), [])
        self.assertEqual(parity.check_stop([1, 2, 99], {99}, 8), [])
        self.assertTrue(parity.check_stop([1, 99, 2], {99}, 8))
        self.assertTrue(parity.check_stop([1, 2], {99}, 8))
        self.assertEqual(parity.check_stop([1, 2], {99}, 2), [])

    def path(self, tokens, recs, problems=()):
        return dict(tokens=tokens, records=recs, problems=list(problems))

    def test_bitwise_equivalence(self):
        recs = [record((1, 2, 3)), record((99, 2, 3))]
        v = parity.classify(self.path([1, 99], recs), self.path([1, 99], recs))
        self.assertEqual(v["verdict"], "equivalent_bitwise")
        self.assertTrue(v["prefill_logits_bitwise_equal"])

    def test_token_equal_numeric_drift(self):
        a = [record((1, 2, 3)), record((99, 2, 3), digest="a")]
        b = [record((1, 2, 3)), record((99, 2, 3), (5.1, 4.0, 1.0), digest="b")]
        v = parity.classify(self.path([1, 99], a), self.path([1, 99], b))
        self.assertEqual(v["verdict"], "equivalent_tokens_numeric_drift")
        self.assertEqual(v["first_non_bitwise_logit_step"], 1)

    def test_near_tie_divergence_is_not_a_bug(self):
        a = [record((1, 2)), record((7, 8, 3), (5.0, 4.9, 1.0), "a")]
        b = [record((1, 2)), record((8, 7, 3), (5.0, 4.95, 1.0), "b")]
        v = parity.classify(self.path([1, 7], a), self.path([1, 8], b))
        self.assertEqual(v["verdict"], "numeric_near_tie_divergence")
        self.assertEqual(v["first_token_difference"], 1)

    def test_bug_verdicts(self):
        same = [record((1, 2)), record((7, 8, 3), (5.0, 4.9, 1.0), "x")]
        v = parity.classify(self.path([1, 7], same), self.path([1, 8], same))
        self.assertEqual(v["verdict"], "IMPLEMENTATION_BUG")  # same logits, other token
        a = [record((1, 2)), record((7, 8, 3), (9.0, 4.0, 1.0), "a")]
        b = [record((1, 2)), record((8, 7, 3), (9.0, 4.0, 1.0), "b")]
        v = parity.classify(self.path([1, 7], a), self.path([1, 8], b))
        self.assertEqual(v["verdict"], "IMPLEMENTATION_BUG")  # confident flip
        v = parity.classify(self.path([1, 2], same), self.path([1], same[:1]))
        self.assertEqual(v["verdict"], "IMPLEMENTATION_BUG")  # stop differs
        v = parity.classify(self.path([1], same[:1], ["x"]), self.path([1], same[:1]))
        self.assertEqual(v["verdict"], "IMPLEMENTATION_BUG")  # invariant violated

    def test_prefill_hash_anomaly_flagged(self):
        a, b = [record((1, 2), digest="a")], [record((1, 2), digest="b")]
        v = parity.classify(self.path([1], a), self.path([1], b))
        self.assertTrue(v["anomalies"])

    def test_phase_selection_from_cache(self):
        self.assertEqual(parity.phase_for(None), "prefill")
        self.assertEqual(parity.phase_for(SimpleNamespace(get_seq_length=lambda: 0)), "prefill")
        self.assertEqual(parity.phase_for(SimpleNamespace(get_seq_length=lambda: 9)), "decode")

    def test_first_difference(self):
        self.assertIsNone(parity.first_difference([1, 2], [1, 2]))
        self.assertEqual(parity.first_difference([1, 2], [1, 3]), 1)
        self.assertEqual(parity.first_difference([1], [1, 2]), 1)


def args_for(root, arm="input", **extra):
    return SimpleNamespace(arm=arm, source=root / "source", artifact=root / "artifact",
                           adapter=root / "run" / "final_adapter", prompts=root / "val.jsonl",
                           continuations=root / "gen.jsonl", **extra)


class PrerequisiteTests(unittest.TestCase):
    def test_missing_files_listed_with_expected_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = parity.check_prerequisites(args_for(Path(tmp)), check_environment=False)
        self.assertFalse(report["prerequisites_present"])
        roles = {m["role"]: m["expected_sha256"] for m in report["missing"]}
        self.assertEqual(roles["adapter/prompt_encoder.safetensors"],
                         parity.ARMS["input"]["files"]["adapter/prompt_encoder.safetensors"])
        self.assertIn("artifact tensor files listed in manifest.json", roles)
        self.assertFalse(report["GPU_execution_started"])

    def test_mismatch_and_pilot_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run").mkdir()
            (root / "val.jsonl").write_text("not the dev46 file\n")
            (root / "run" / "run.json").write_text(json.dumps(dict(training_complete=True,
                                                                   optimizer_steps=12)))
            report = parity.check_prerequisites(args_for(root, "v2"), check_environment=False)
        problems = {m["role"] for m in report["mismatched"]}
        self.assertEqual(problems, {"prompts", "run.json"})


def item(**override):
    docs = [dict(text=f"passage {i}") for i in range(20)]
    base = dict(pair_id="p1", panel="training_side", condition="cued", prefix_fact_count=0,
                prompt=" ".join(d["text"] for d in docs), documents=docs, truncated=False,
                target_prefix='{"supported_facts":[{"fact":"', correct_clause="A to B",
                wrong_clause="B to A", target_suffix='"}]}')
    base.update(override)
    return base


class BindingProbeTests(unittest.TestCase):
    def test_valid_items_pass_and_bad_items_are_named(self):
        self.assertEqual(probe.validate_items([item(), item(condition="uncued",
                                                            prefix_fact_count=2)]), [])
        bad = [item(condition="uncued"), item(pair_id="p2", wrong_clause="A to B"),
               item(pair_id="p3", documents=item()["documents"][:19]),
               item(pair_id="p4", panel="pooled"), item(), item()]
        errors = probe.validate_items(bad)
        self.assertEqual(len(errors), 5)
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_clause_span_and_boundary_cleanliness(self):
        offsets = [(0, 2), (2, 4), (4, 6), (6, 8)]
        self.assertEqual(probe.clause_span(offsets, 2, 6), ([1, 2], True))
        self.assertEqual(probe.clause_span(offsets, 3, 6), ([1, 2], False))

    def test_differing_span_aligns_shared_prefix_and_suffix(self):
        a, b = [1, 2, 3, 4, 9, 9], [1, 2, 5, 9, 9]
        self.assertEqual(probe.differing_span(a, b), ((2, 4), (2, 3)))
        self.assertEqual(probe.differing_span([1, 2], [1, 2, 3]), ((2, 2), (2, 3)))

    def test_score_pair_sums_means_and_margin(self):
        enc_c = dict(ids=[1, 2, 3, 9], clause=[1, 2], clean_boundaries=True)
        enc_w = dict(ids=[1, 5, 3, 9], clause=[1, 2], clean_boundaries=True)
        r = probe.score_pair([-0.1, -0.5, -0.2, -0.1], [-0.1, -2.0, -0.4, -0.1], enc_c, enc_w)
        self.assertAlmostEqual(r["differing_margin"], 1.5)
        self.assertAlmostEqual(r["clause_sum_margin"], 1.7)
        self.assertAlmostEqual(r["clause_mean_margin"], 0.85)
        self.assertTrue(r["prefers_correct"])
        self.assertEqual(r["correct"]["clause"]["tokens"], 2)

    def test_aggregate_never_pools_panels_or_conditions(self):
        rows = [dict(panel="training_side", condition="cued", prefers_correct=True,
                     differing_margin=2.0, clause_sum_margin=2.0, clause_mean_margin=1.0,
                     clean_boundaries=True),
                dict(panel="training_side", condition="uncued", prefers_correct=False,
                     differing_margin=-1.0, clause_sum_margin=-1.0, clause_mean_margin=-0.5,
                     clean_boundaries=False)]
        out = probe.aggregate(rows)
        self.assertEqual(set(out), {"training_side/cued", "training_side/uncued"})
        self.assertEqual(out["training_side/uncued"]["prefers_correct_fraction"], 0.0)
        self.assertEqual(out["training_side/uncued"]["unclean_boundaries"], 1)

    def test_probe_prerequisites_drop_continuations_but_require_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "items.jsonl").write_text(json.dumps(item()) + "\n")
            args = args_for(root, items=root / "items.jsonl", parity_summary=None)
            report = probe.check_prerequisites(args, check_environment=False)
        roles = {m["role"] for m in report["missing"]}
        self.assertNotIn("continuations", roles)
        self.assertNotIn("prompts", roles)
        self.assertIn("adapter/prompt_encoder.safetensors", roles)
        self.assertEqual(report["item_errors"], [])
        self.assertFalse(report["prerequisites_present"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[__file__] + remaining, verbosity=2)
