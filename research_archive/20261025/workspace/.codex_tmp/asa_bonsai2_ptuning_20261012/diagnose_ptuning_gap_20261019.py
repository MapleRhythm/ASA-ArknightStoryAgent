"""Read-only CPU diagnostics on completed 750-example adapters and run traces."""
from collections import Counter
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_file
from backend import PromptEncoder
import annotation_protocol as p

ROOT = Path(__file__).resolve().parent


def drift(a, b):
    a, b = a.float(), b.float()
    delta = b - a
    return dict(
        elements=a.numel(), initial_rms=a.square().mean().sqrt().item(),
        final_rms=b.square().mean().sqrt().item(),
        delta_rms=delta.square().mean().sqrt().item(),
        relative_l2=(delta.norm() / a.norm().clamp_min(1e-30)).item(),
        fp32_changed_fraction=(a != b).float().mean().item(),
        bf16_changed_fraction=(a.bfloat16() != b.bfloat16()).float().mean().item())


def trace(path):
    rows = [r for r in p.json_rows(path) if r.get("event") == "optimizer_step"]
    return dict(
        steps=len(rows), first50_mean_loss=statistics.mean(r["loss"] for r in rows[:50]),
        last50_mean_loss=statistics.mean(r["loss"] for r in rows[-50:]),
        mean_step_seconds=statistics.mean(r["elapsed_seconds"] for r in rows),
        ids=[r["id"] for r in rows])


def main():
    torch.set_num_threads(2)
    deep_path = ROOT / "ptuning_v2_deep_20261017_attempt01"
    shallow_path = ROOT / "targeted_pt_v2_20261014_attempt02/trained_v2"
    di = load_file(str(deep_path / "initial_adapter/deep_prefix.safetensors"))
    df = load_file(str(deep_path / "final_adapter/deep_prefix.safetensors"))
    blocks = {k: drift(di[k], df[k]) for k in di if k.startswith("prefix.")}
    groups = {}
    for name, value in blocks.items():
        group = name.rsplit("_", 1)[-1]
        groups.setdefault(group, []).append(value)
    group_summary = {}
    for name, values in groups.items():
        group_summary[name] = {
            "blocks": len(values),
            **{key + "_median": statistics.median(v[key] for v in values)
               for key in ("initial_rms", "delta_rms", "relative_l2",
                           "fp32_changed_fraction", "bf16_changed_fraction")}}
    si = PromptEncoder.load(shallow_path / "initial_adapter", device="cpu")
    sf = PromptEncoder.load(shallow_path / "final_adapter", device="cpu")
    with torch.no_grad():
        shallow_output_drift = drift(si(), sf())
    shallow_run = json.loads((shallow_path / "run.json").read_text())
    deep_run = json.loads((deep_path / "run.json").read_text())
    st = trace(shallow_path / "progress.jsonl")
    dt = trace(deep_path / "progress.jsonl")
    ids_equal = st.pop("ids") == dt.pop("ids")
    old = json.loads((ROOT / "NATIVE46_V2_CONTENT_REVIEW_20261017.json").read_text())["cases"]
    new = json.loads((ROOT / "DEEP_PTUNING_CONTENT_REVIEW_20261017.json").read_text())["cases"]
    assert len(old) == len(new) == 46
    transitions = Counter()
    regressed, improved = [], []
    for a, b in zip(old, new):
        assert a["position"] == b["position"]
        transitions[(a["category"], b["category"])] += 1
        if a["category"] == "complete" and b["category"] != "complete":
            regressed.append(dict(position=a["position"], old=a["note"], new=b["note"]))
        if a["category"] != "complete" and b["category"] == "complete":
            improved.append(dict(position=a["position"], old=a["note"], new=b["note"]))
    result = dict(
        evidence_scope="completed historical 750-example runs, not new1117 training",
        shallow_trainable_parameters=shallow_run["trainable_parameters"],
        deep_trainable_parameters=deep_run["trainable_parameters"],
        parameters_ratio=deep_run["trainable_parameters"] / shallow_run["trainable_parameters"],
        same_learning_rate=shallow_run["learning_rate"] == deep_run["learning_rate"],
        learning_rate=shallow_run["learning_rate"], same_data_order=ids_equal,
        same_train_data_hash=shallow_run["data_sha256"]["train"] ==
                            deep_run["identity"]["data_sha256"]["train"],
        initial_shallow_weights_sha256=p.file_sha(shallow_path / "initial_adapter/prompt_encoder.safetensors"),
        deep_initialized_from_initial_shallow_sha256=
            deep_run["identity"]["initial_adapter_sha256"],
        shallow_trace=st, deep_trace=dt, deep_block_drift=blocks,
        deep_group_drift=group_summary, shallow_output_drift=shallow_output_drift,
        quantization_diagnostic_note="BF16 changed fraction measures each group's local tensor "
        "after cast, not functional equivalence or comparable influence across parameterizations.",
        transitions={f"{a}->{b}": n for (a, b), n in transitions.items()},
        regressed=regressed, improved=improved,
        paired_discordant_count=len(regressed)+len(improved),
        two_sided_exact_mcnemar_p=min(1,2 * sum(
            __import__("math").comb(len(regressed)+len(improved), k)
            for k in range(min(len(regressed),len(improved))+1)) /
            2 ** (len(regressed)+len(improved))),
        statistical_caveat="46 repeatedly-used development rows, only43 normalized question groups; "
        "paired-binomial calculation is descriptive, not an independent confirmatory benchmark.",
        causal_diagnosis_not_proven=True,
    )
    p.atomic_json(ROOT / "PTUNING_GAP_DIAGNOSTICS_20261019.json", result)
    print(p.compact({k: v for k,v in result.items() if k not in ("deep_block_drift","regressed","improved")}))


if __name__ == "__main__":
    main()
