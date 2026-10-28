"""Same-adapter decoder parity: HF model.generate vs ptuning_v2_hybrid.greedy_generate.

Inference only; never trains or downloads. Default CLI checks prerequisites and
exits nonzero listing every missing/mismatched file with its expected SHA-256.
--execute runs both decoders on the SAME loaded adapter and prompt, records the
top-k logits and a bitwise hash of every step's logits, and reports the first
divergent token and whether it is a numeric near-tie or an implementation bug.
Results are decoder diagnostics, never answer-correctness measurements.
Outputs contain answer token IDs (decodable to answer text): write them only to
private task evidence. See README.md for the decision rules.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

EOS_IDS = [248046, 248044]  # <|im_end|>, <|endoftext|>; historical pad = EOS_IDS[0]
TIE_TOL = 0.25    # 2 BF16 ulps at |logit| in [16, 32)
DRIFT_TOL = 0.5   # historical 27B cache-vs-recompute max_abs was 0.078-0.125
TOP_K = 16

# Files whose identity binds a run to the historical 46-question evaluation.
# sha None: must exist; identity recorded, enforced by the loader where possible.
COMMON = {
    "source/backend.py": "397cbfa82e943bee67ea513cf6516170ccfc7eec6a12c2723a2697369effbd82",
    "source/ptuning_v2_hybrid.py": "4b1f00288ac04704192efc7775796756b036376675591f308ec6c04c0e1a467b",
    "source/official/tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "artifact/manifest.json": "77b2c71cfb46fb3a056385a9a613e0b4c5ba2aa4630b7933db1664e0600121b2",
    "artifact/signs.safetensors": None,
    # glm_reviewed_full20_data_v1_20261014/val.jsonl == both runs' data_sha256.val
    "prompts": "046adcee638d5a7c36c93836eb12e96ec29e883337153bee0d687849069a3d8d",
}
ARMS = {
    # ptuning_input_curated_20261019_attempt01: HF generate was the historical decoder.
    "input": dict(historical_path="hf", files={
        "adapter/prompt_encoder.safetensors":
            "157456cc075a6e3387ab094b906f8da625f448b8fe8519a2cf441e3fb4a0e607",
        "adapter/adapter_config.json": None,
        "run.json": "fa634c4311a89f32323aff95d63e4d72431ee0275b847d887902b3fa0681ceaf",
        "continuations": "1cc1c65e34a43d766e0439808d0a06b022be640dfdb10d41dab8918e2447b804",
    }),
    # ptuning_v2_optimized_curated_20261019_attempt01: explicit greedy was historical.
    # deep_prefix weights are verified by DeepPrefix.load against adapter_config;
    # checkpoint-1117 recorded adapter_sha256 c392ebc7... (same state, other file).
    "v2": dict(historical_path="explicit", files={
        "adapter/deep_prefix.safetensors": None,
        "adapter/adapter_config.json": None,
        "run.json": "0b221b330ac1261204dcbcd955a6724a5a586bf4d2588c46a6d780cb26a56251",
        "continuations": "291ebcd00cb8d525ec944384963c4e5250d334e8ed3a57c73ab3edf81830da66",
    }),
}
ENVIRONMENT = {
    "transformers": "5.6.2",
    "transformers/models/qwen3_5/modeling_qwen3_5.py":
        "3e2b6239e4b2c3e512d4f9836a1dba12e975ae652407e84f2d3c2beebf0c9528",
    "transformers/generation/utils.py":
        "2cf094451e1f30391c3449cfd44828ab8b27ba721ae2cdc0a0415413689b0f9c",
    "transformers/masking_utils.py":
        "7fd2bf34f3abc87953a500c0d53e3e633e5efba10aebcc4dc11422e5769092dd",
    "transformers/cache_utils.py":
        "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def resolve(args, key):
    if key == "prompts":
        return args.prompts
    if key == "continuations":
        return args.continuations
    if key == "run.json":
        return args.adapter.parent / "run.json"
    root, rest = key.split("/", 1)
    return dict(source=args.source, artifact=args.artifact, adapter=args.adapter)[root] / rest


def environment_report():
    """Locate installed Transformers without importing it (no torch/GPU import)."""
    spec = importlib.util.find_spec("transformers")
    problems, found = [], {}
    if spec is None or spec.origin is None:
        return dict(problems=[dict(kind="missing_package", name="transformers",
                                   expected=ENVIRONMENT["transformers"])], found=found)
    root = Path(spec.origin).parent
    found["transformers"] = importlib.metadata.version("transformers")
    if found["transformers"] != ENVIRONMENT["transformers"]:
        problems.append(dict(kind="version", name="transformers", actual=found["transformers"],
                             expected=ENVIRONMENT["transformers"]))
    for key, expected in ENVIRONMENT.items():
        if key == "transformers":
            continue
        path = root / key.split("/", 1)[1]
        actual = sha(path) if path.is_file() else None
        found[key] = actual
        if actual != expected:
            problems.append(dict(kind="sha256", path=str(path), actual=actual, expected=expected))
    for name in ("torch", "tokenizers", "safetensors", "triton"):
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(dict(kind="missing_package", name=name))
    return dict(problems=problems, found=found)


def check_prerequisites(args, check_environment=True, files=None):
    """Everything the GPU run needs, all at once; no fallback and no torch import."""
    missing, mismatched, identity = [], [], {}
    if files is None:
        files = dict(COMMON, **ARMS[args.arm]["files"])
    for key, expected in files.items():
        path = resolve(args, key)
        if not path.is_file():
            missing.append(dict(role=key, path=str(path), expected_sha256=expected))
            continue
        actual = identity[str(path)] = sha(path)
        if expected is not None and actual != expected:
            mismatched.append(dict(role=key, path=str(path), actual=actual, expected=expected))
    manifest = args.artifact / "manifest.json"
    if manifest.is_file():
        records = json.loads(manifest.read_text()).get("tensors", [])
        for name in sorted({r["file"] for r in records}):
            if not (args.artifact / name).is_file():
                missing.append(dict(role="artifact tensor file", path=str(args.artifact / name),
                                    expected_sha256=None))
    else:
        missing.append(dict(role="artifact tensor files listed in manifest.json",
                            path=str(args.artifact / "*"), expected_sha256=None))
    run_path = resolve(args, "run.json")
    if run_path.is_file():
        run = json.loads(run_path.read_text())
        if not (run.get("training_complete") and run.get("optimizer_steps") == 1117
                and run.get("train_examples") == 1117 and run.get("backbone_bytewise_unchanged")):
            mismatched.append(dict(role="run.json", path=str(run_path),
                                   problem="not the completed frozen-backbone 1117-step run"))
    environment = environment_report() if check_environment else dict(problems=[], found={})
    ok = not (missing or mismatched or environment["problems"])
    return dict(prerequisites_present=ok, arm=args.arm, missing=missing, mismatched=mismatched,
                environment=environment, files_sha256=identity, GPU_execution_started=False)


class NullController:
    """Input-layer adapter has no layer hooks; lets greedy_generate drive it unchanged."""
    mode = None

    @contextmanager
    def activate(self, prefill=True):
        yield self

    def close(self):
        pass


class InputPrompt:
    """PromptEncoder seen through the DeepPrefix interface greedy_generate uses."""
    def __init__(self, encoder):
        self.encoder, self.length = encoder, encoder.length

    def prepare(self, embedded):
        return self.encoder.inject(embedded)


def phase_for(cache):
    return "decode" if cache is not None and cache.get_seq_length() > 0 else "prefill"


@contextmanager
def trace_forward(model, control, bridge_hf=False):
    """Observe each top-level forward and the RoPE positions it actually used.

    bridge_hf=True selects the deep-prefix phase from the cache before layer
    hooks run; HF generate has no other way to drive PrefixController.
    """
    trace = []
    previous = control.mode

    def before(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        length = cache.get_seq_length() if cache is not None else 0
        if bridge_hf:
            control.mode = phase_for(cache)
        embedded, ids = kwargs.get("inputs_embeds"), kwargs.get("input_ids")
        tensor = embedded if embedded is not None else ids
        mask = kwargs.get("attention_mask")
        trace.append(dict(
            past_seq_length=length, current_tokens=int(tensor.shape[1]),
            inputs_embeds=embedded is not None, controller_mode=control.mode,
            attention_mask_shape=list(mask.shape) if mask is not None else None,
            attention_mask_all_ones=bool(mask.all()) if mask is not None else None,
            caller_supplied_positions=kwargs.get("position_ids") is not None))

    def rope_before(module, args, kwargs):
        positions = kwargs.get("position_ids")
        if positions is None:
            positions = args[1]
        trace[-1]["effective_rope_positions"] = dict(
            minimum=int(positions.min()), maximum=int(positions.max()))

    handles = [model.register_forward_pre_hook(before, with_kwargs=True),
               model.model.rotary_emb.register_forward_pre_hook(rope_before, with_kwargs=True)]
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()
        control.mode = previous


def summarize_logits(vector, k=TOP_K):
    import torch
    values, ids = vector.float().topk(k)
    raw = vector.detach().contiguous().cpu()
    return dict(top_ids=ids.tolist(), top_values=values.tolist(),
                sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(vector).all()))


@contextmanager
def record_logits(model, k=TOP_K):
    """One record per lm_head call = per decoding step (logits_to_keep=1)."""
    records = []

    def hook(module, args, output):
        records.append(summarize_logits(output[0, -1], k))

    handle = model.lm_head.register_forward_hook(hook)
    try:
        yield records
    finally:
        handle.remove()


def check_trace(trace, total_prompt, n_tokens, deep):
    """Prefix inserted once, cache grows by one, RoPE == cache length, phases right."""
    problems = []
    if len(trace) != n_tokens:
        problems.append(f"{len(trace)} forwards for {n_tokens} tokens (expected one per token)")
    for k, t in enumerate(trace):
        prefill = k == 0
        past = 0 if prefill else total_prompt + k - 1
        current = total_prompt if prefill else 1
        expect = dict(past_seq_length=past, current_tokens=current, inputs_embeds=prefill)
        for key, value in expect.items():
            if t[key] != value:
                problems.append(f"forward {k}: {key}={t[key]} expected {value}")
        rope = t.get("effective_rope_positions")
        if rope != dict(minimum=past, maximum=past + current - 1):
            problems.append(f"forward {k}: RoPE {rope} expected [{past}, {past + current - 1}]")
        if t["attention_mask_shape"] is not None and (
                t["attention_mask_shape"][-1] != past + current or not t["attention_mask_all_ones"]):
            problems.append(f"forward {k}: attention mask {t['attention_mask_shape']} "
                            f"all_ones={t['attention_mask_all_ones']} for {past + current} keys")
        mode = ("prefill" if prefill else "decode") if deep else None
        if t["controller_mode"] != mode:
            problems.append(f"forward {k}: controller mode {t['controller_mode']} expected {mode}")
    return problems


def check_emitted(records, tokens):
    """Greedy contract: every emitted token is a maximal logit of its own step."""
    problems = []
    if len(records) != len(tokens):
        problems.append(f"{len(records)} logit records for {len(tokens)} tokens")
    for k, (r, token) in enumerate(zip(records, tokens)):
        best = r["top_values"][0]
        maximal = {i for i, v in zip(r["top_ids"], r["top_values"]) if v == best}
        if token not in maximal:
            problems.append(f"step {k}: emitted {token}, maximal logits at {sorted(maximal)}")
        if not r["finite"]:
            problems.append(f"step {k}: nonfinite logits")
    return problems


def check_stop(tokens, eos_ids, max_new_tokens):
    hits = [k for k, t in enumerate(tokens) if t in eos_ids]
    if hits and hits[0] != len(tokens) - 1:
        return [f"continued after EOS at step {hits[0]}"]
    if not hits and len(tokens) != max_new_tokens:
        return [f"stopped at {len(tokens)} tokens without EOS or budget {max_new_tokens}"]
    return []


def margin(record):
    return record["top_values"][0] - record["top_values"][1]


def topk_difference(a, b):
    """max |logit_a - logit_b| over token IDs present in both top-k lists."""
    left = dict(zip(a["top_ids"], a["top_values"]))
    right = dict(zip(b["top_ids"], b["top_values"]))
    shared = left.keys() & right.keys()
    return max((abs(left[i] - right[i]) for i in shared), default=float("inf"))


def classify(hf, explicit, tie_tol=TIE_TOL, drift_tol=DRIFT_TOL):
    """hf/explicit: dict(tokens, records, problems). Returns verdict + evidence."""
    problems = [f"hf: {p}" for p in hf["problems"]] + [f"explicit: {p}" for p in explicit["problems"]]
    ta, tb, ra, rb = hf["tokens"], explicit["tokens"], hf["records"], explicit["records"]
    d = first_difference(ta, tb)
    compared = min(len(ra), len(rb)) if d is None else min(d + 1, len(ra), len(rb))
    bitwise = [ra[k]["sha256"] == rb[k]["sha256"] for k in range(compared)]
    drift = [topk_difference(ra[k], rb[k]) for k in range(compared)]
    evidence = dict(
        first_token_difference=d, compared_steps=compared,
        first_non_bitwise_logit_step=next((k for k, same in enumerate(bitwise) if not same), None),
        max_topk_logit_difference=max(drift, default=0.0),
        prefill_logits_bitwise_equal=bitwise[0] if bitwise else None)
    anomalies = []
    if bitwise and not bitwise[0]:
        anomalies.append("prefill logits not bitwise equal although both paths run identical "
                         "kernels on identical inputs; inspect mask/position/prefix handling")
    if problems:
        return dict(verdict="IMPLEMENTATION_BUG", reasons=problems, anomalies=anomalies, **evidence)
    if d is None:
        verdict = "equivalent_bitwise" if all(bitwise) else "equivalent_tokens_numeric_drift"
        return dict(verdict=verdict, reasons=[], anomalies=anomalies, **evidence)
    if d >= min(len(ta), len(tb)):
        return dict(verdict="IMPLEMENTATION_BUG", anomalies=anomalies, **evidence,
                    reasons=[f"same tokens through step {d - 1} but different stop/length"])
    evidence.update(hf_token=ta[d], explicit_token=tb[d], hf_margin=margin(ra[d]),
                    explicit_margin=margin(rb[d]), divergence_logit_difference=drift[d])
    if bitwise[d]:
        return dict(verdict="IMPLEMENTATION_BUG", anomalies=anomalies, **evidence,
                    reasons=[f"step {d}: bitwise-identical logits but different tokens"])
    if (evidence["hf_margin"] <= tie_tol and evidence["explicit_margin"] <= tie_tol
            and max(drift) <= drift_tol):
        return dict(verdict="numeric_near_tie_divergence", reasons=[], anomalies=anomalies,
                    **evidence)
    return dict(verdict="IMPLEMENTATION_BUG", anomalies=anomalies, **evidence, reasons=[
        f"step {d}: tokens differ with margins {evidence['hf_margin']:.4f}/"
        f"{evidence['explicit_margin']:.4f} (tie_tol {tie_tol}) and max top-k logit "
        f"difference {max(drift):.4f} (drift_tol {drift_tol}) before/at divergence"])


def compare_logits(cached, full, eos_ids):
    import torch
    a, b = cached.float(), full.float()
    pa, pb = a.log_softmax(-1), b.log_softmax(-1)
    values_a, tokens_a = a.topk(2)
    values_b, tokens_b = b.topk(2)
    return dict(
        max_abs=float((a - b).abs().max()),
        probability_max_abs=float((pa.exp() - pb.exp()).abs().max()),
        kl_full_to_cached=float((pb.exp() * (pb - pa)).sum().clamp_min(0)),
        cached_top1=int(tokens_a[0]), full_top1=int(tokens_b[0]),
        top1_equal=bool(tokens_a[0] == tokens_b[0]),
        cached_margin=float(values_a[0] - values_a[1]),
        full_margin=float(values_b[0] - values_b[1]),
        cached_top1_is_eos=int(tokens_a[0]) in eos_ids,
        full_top1_is_eos=int(tokens_b[0]) in eos_ids,
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


def replay_same_trajectory(model, adapter, control, input_ids, continuation,
                           eos_ids, replay_tokens=128, checkpoints=(0, 1, 2, 7, 31, 127)):
    """Cache vs no-cache recomputation along ONE forced (historical) trajectory.

    step 0 predicts continuation[0]; step k has consumed continuation[:k].
    Full recomputation creates no cache and cannot mutate the cached branch.
    """
    import torch
    if replay_tokens < 1 or not continuation:
        raise ValueError("Need a nonempty continuation and positive replay budget")
    count = min(len(continuation), replay_tokens)
    selected = {s for s in set(checkpoints) | {count - 1} if 0 <= s < count}
    ids = input_ids.clone()
    measurements = []
    model.eval()
    with torch.no_grad():
        with control.activate(prefill=True):
            cached = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                           use_cache=True, logits_to_keep=1)
        for step in range(count):
            expected_length = adapter.length + ids.shape[1]
            actual_length = cached.past_key_values.get_seq_length()
            if actual_length != expected_length:
                raise RuntimeError(f"Cache length {actual_length} != {expected_length} at step {step}")
            if step in selected:
                with control.activate(prefill=True):
                    full = model(inputs_embeds=adapter.prepare(model.model.embed_tokens(ids)),
                                 use_cache=False, logits_to_keep=1)
                entry = compare_logits(cached.logits[0, -1], full.logits[0, -1], eos_ids)
                entry.update(step=step, cache_length=actual_length,
                             forced_next_token=int(continuation[step]))
                measurements.append(entry)
                del full
            if step + 1 == count:
                break
            token = torch.tensor([[continuation[step]]], device=input_ids.device)
            ids = torch.cat((ids, token), dim=1)
            with control.activate(prefill=False):
                cached = model(input_ids=token, past_key_values=cached.past_key_values,
                               use_cache=True, logits_to_keep=1)
    return dict(trajectory="fixed_historical_continuation", replayed_predictions=count,
                checked_steps=sorted(selected),
                first_checked_top1_disagreement=next(
                    (m["step"] for m in measurements if not m["top1_equal"]), None),
                checks=measurements, unmeasured_steps_are_not_certified=True)


def run_hf(model, adapter, control, input_ids, max_new_tokens, eos_ids, deep):
    """Exactly the historical input-layer call (train_curated_shallow_20261019.evaluate)."""
    import torch
    with torch.no_grad(), trace_forward(model, control, bridge_hf=deep) as trace, \
            record_logits(model) as records:
        embedded = adapter.prepare(model.model.embed_tokens(input_ids))
        tokens = model.generate(
            inputs_embeds=embedded,
            attention_mask=torch.ones(embedded.shape[:2], dtype=torch.long, device=input_ids.device),
            max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            eos_token_id=list(eos_ids), pad_token_id=eos_ids[0])[0].tolist()
    return tokens, records, trace


def run_explicit(model, adapter, control, input_ids, max_new_tokens, eos_ids):
    """Exactly the historical v2 call (run_ptuning_v2_hybrid.evaluate)."""
    from ptuning_v2_hybrid import greedy_generate
    with trace_forward(model, control) as trace, record_logits(model) as records:
        tokens = greedy_generate(model, adapter, control, input_ids, max_new_tokens, set(eos_ids))
    return tokens, records, trace


def diagnose(model, adapter, control, input_ids, continuation, eos_ids, deep,
             historical_path, max_new_tokens=1024, replay_tokens=128, context_limit=32768):
    if input_ids.shape[0] != 1 or input_ids.ndim != 2:
        raise ValueError("Unpadded batch one required")
    if adapter.length + input_ids.shape[1] + max(max_new_tokens, replay_tokens) > context_limit:
        raise ValueError("Context exceeded; refusing to truncate prompt")
    model.eval()
    model.gradient_checkpointing_disable()
    total = adapter.length + input_ids.shape[1]
    paths = {}
    for name in ("hf", "explicit"):
        if name == "hf":
            tokens, records, trace = run_hf(model, adapter, control, input_ids,
                                            max_new_tokens, eos_ids, deep)
        else:
            tokens, records, trace = run_explicit(model, adapter, control, input_ids,
                                                  max_new_tokens, eos_ids)
        problems = (check_trace(trace, total, len(tokens), deep) + check_emitted(records, tokens)
                    + check_stop(tokens, eos_ids, max_new_tokens))
        paths[name] = dict(tokens=tokens, records=records, trace=trace, problems=problems)
    verdict = classify(paths["hf"], paths["explicit"])
    d = verdict["first_token_difference"]
    fixed = replay_same_trajectory(model, adapter, control, input_ids, continuation, eos_ids,
                                   replay_tokens=replay_tokens,
                                   checkpoints=(0, 1, 2, 7, 31, 127) + (() if d is None else (d,)))
    history = paths[historical_path]["tokens"]
    return dict(
        kind="same_adapter_decoder_parity_not_QA_accuracy",
        verdict=verdict, historical_path=historical_path,
        historical_reproduced=history == list(continuation),
        historical_first_difference=first_difference(list(continuation), history),
        other_path_vs_history_first_difference=first_difference(
            list(continuation), paths["explicit" if historical_path == "hf" else "hf"]["tokens"]),
        generation_config=model.generation_config.to_dict(),
        eos_ids=list(eos_ids), pad_token_id=eos_ids[0], max_new_tokens=max_new_tokens,
        prompt_tokens=int(input_ids.shape[1]), prefix_tokens=adapter.length,
        paths={k: dict(tokens=v["tokens"], problems=v["problems"], trace=v["trace"],
                       records=v["records"]) for k, v in paths.items()},
        cache_replay=fixed, no_input_truncation=True, training_performed=False,
        QA_accuracy_measured=False)


def load_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="sealed .codex_tmp/asa_bonsai2_ptuning_20261012 (read only)")
    parser.add_argument("--artifact", type=Path, required=True, help="packed 27B backbone dir")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="final_adapter dir; run.json is read from its parent")
    parser.add_argument("--prompts", type=Path, required=True, help="dev46 val.jsonl (full 20)")
    parser.add_argument("--continuations", type=Path, required=True,
                        help="the arm's historical generated_native46.jsonl")
    parser.add_argument("--id", action="append", default=[], help="task id; repeat, or use --all")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, required=True, help="new PRIVATE evidence directory")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--replay-tokens", type=int, default=128)
    parser.add_argument("--execute", action="store_true", help="explicit GPU execution")
    args = parser.parse_args()
    report = check_prerequisites(args)
    print(json.dumps(report, indent=2))
    if not report["prerequisites_present"]:
        raise SystemExit("Prerequisites missing or mismatched (no fallback); see JSON above")
    if not args.execute:
        return
    if bool(args.id) == args.all:
        raise SystemExit("Give --id (repeatable) or --all")
    args.out.mkdir(parents=True, exist_ok=False)
    # Nothing above this line imports a GPU runtime or model package.
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0, str(args.source.resolve()))
    from backend import PromptEncoder, load_backend
    from ptuning_v2_hybrid import DeepPrefix, PrefixController, architecture
    rows = {r["id"]: r for r in load_rows(args.prompts)}
    history = {r["id"]: r for r in load_rows(args.continuations)}
    ids = list(history) if args.all else args.id
    tokenizer = Tokenizer.from_file(str(args.source / "official/tokenizer.json"))
    eos = [tokenizer.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
    if eos != EOS_IDS:
        raise SystemExit(f"EOS ids {eos} != historical {EOS_IDS}")
    model, manifest = load_backend(args.artifact, dtype=torch.bfloat16)
    deep = args.arm == "v2"
    if deep:
        adapter = DeepPrefix.load(args.adapter, architecture(model.config), manifest["source_sha256"])
        control = PrefixController(model, adapter)
    else:
        encoder = PromptEncoder.load(args.adapter)
        encoder.eval()
        adapter, control = InputPrompt(encoder), NullController()
    summary = dict(arm=args.arm, prerequisites=report, results={},
                   kind="same_adapter_decoder_parity_not_QA_accuracy")
    try:
        for task_id in ids:
            row, native = rows[task_id], history[task_id]
            prompt = row["prompt"]
            if len(row["documents"]) != 20 or row.get("truncated") is not False \
                    or not all(d["text"] in prompt for d in row["documents"]):
                raise ValueError(f"{task_id}: require explicit full-20 untruncated prompt")
            if native["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                raise ValueError(f"{task_id}: continuation prompt identity mismatch")
            ids_ = tokenizer.encode(prompt, add_special_tokens=False).ids
            if len(ids_) != native["prompt_tokens"] or native["prefix_tokens"] != adapter.length:
                raise ValueError(f"{task_id}: prompt/prefix token count differs from history")
            result = diagnose(model, adapter, control, torch.tensor([ids_], device="cuda"),
                              native["token_ids"], eos, deep, ARMS[args.arm]["historical_path"],
                              args.max_new_tokens, args.replay_tokens)
            with (args.out / f"{args.arm}_{task_id}.json").open("x") as stream:
                json.dump(result, stream, indent=1)
            summary["results"][task_id] = dict(
                verdict=result["verdict"]["verdict"],
                first_token_difference=result["verdict"]["first_token_difference"],
                historical_reproduced=result["historical_reproduced"],
                cache_replay_first_top1_disagreement=result["cache_replay"][
                    "first_checked_top1_disagreement"])
            print(json.dumps(dict(id=task_id, **summary["results"][task_id])), flush=True)
    finally:
        control.close()
        with (args.out / f"summary_{args.arm}.json").open("x") as stream:
            json.dump(summary, stream, indent=1)


if __name__ == "__main__":
    main()
