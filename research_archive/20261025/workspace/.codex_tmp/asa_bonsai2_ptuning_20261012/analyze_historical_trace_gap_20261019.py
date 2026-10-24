"""Read-only paired analysis of the two historical 750-example training traces."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steps(path):
    return [row for row in map(json.loads, path.read_text().splitlines())
            if row.get("event") == "optimizer_step"]


def analyze(shallow, deep):
    assert len(shallow) == len(deep) == 750
    assert all(a["step"] == b["step"] == i and a["id"] == b["id"]
               for i, (a, b) in enumerate(zip(shallow, deep), 1))
    assert all((a["prompt_tokens"], a["target_tokens"])
               == (b["prompt_tokens"], b["target_tokens"]) for a, b in zip(shallow, deep))
    result = dict(
        scope="historical_completed_750_example_runs_not_new1117",
        same_sample_order_and_encoded_lengths=True,
        interpretation_caveats=[
            "Gradient norms in different parameter coordinates are not functional-effect ratios.",
            "Clipping affects Adam moment history; it is not simply a proportional learning-rate change.",
            "Losses are pre-update on a changing sequence of examples, not a fixed validation loss.",
            "Different optimizers/parameterizations have not been tested by a controlled intervention.",
        ])
    for name, rows, key in (
        ("shallow750", shallow, "prompt_grad_norm"),
        ("deep750", deep, "prefix_gradient_norm"),
    ):
        gradients = [r[key] for r in rows]
        result[name] = dict(
            n=len(rows), gradient_min=min(gradients),
            gradient_median=statistics.median(gradients), gradient_max=max(gradients),
            clipped_steps=sum(v > 1 for v in gradients),
            clipped_fraction=sum(v > 1 for v in gradients) / len(gradients),
            mean_loss=statistics.mean(r["loss"] for r in rows))
    result["paired_windows"] = []
    for start in range(0, 750, 150):
        differences = [b["loss"] - a["loss"]
                       for a, b in zip(shallow[start:start + 150], deep[start:start + 150])]
        result["paired_windows"].append(dict(
            start_step=start + 1, end_step=start + len(differences), n=len(differences),
            deep_minus_shallow_mean=statistics.mean(differences),
            deep_minus_shallow_median=statistics.median(differences),
            deep_higher_loss_count=sum(d > 0 for d in differences)))
    return result


def main():
    shallow_path = ROOT / "targeted_pt_v2_20261014_attempt02/trained_v2/progress.jsonl"
    deep_path = ROOT / "ptuning_v2_deep_20261017_attempt01/progress.jsonl"
    result = analyze(steps(shallow_path), steps(deep_path))
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): sha(p) for p in (shallow_path, deep_path)}
    result["analysis_script_sha256"] = sha(Path(__file__))
    output = ROOT / "PTUNING_TRAINING_TRACE_GAP_20261019.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
