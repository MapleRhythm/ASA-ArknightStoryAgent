#!/usr/bin/env python3
"""Read-only metadata/source audit. Produces no story text and no new semantic scores."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit(root):
    root = Path(root)
    old_review = read(root / "INPUT_FINAL1117_CONTENT_REVIEW_20261019.json")
    new_review = read(root / "V2_OPTIMIZED1117_CONTENT_REVIEW_20261022.json")
    task_path = root / new_review["tasks"]
    tasks = rows(task_path)
    before = rows(root / old_review["run"] / "generated_native46.jsonl")
    after = rows(root / new_review["snapshot"] / "generated_native46.jsonl")
    assert len(tasks) == len(before) == len(after) == 46
    eos = [248046, 248044]
    import paired_protocol as p
    for task, old, new in zip(tasks, before, after):
        p.task_identity(task)
        assert task["id"] == old["id"] == new["id"]
        assert old["prompt_sha256"] == new["prompt_sha256"] == task["bonsai_prompt_sha256"]
        for output in (old, new):
            assert output["evidence_count"] == 20 and not output["truncated"]
            assert output["prefix_tokens"] == 16
            assert not output["output_limit_reached"]
            assert output["token_ids"][-1] in eos
            assert not any(t in eos for t in output["token_ids"][:-1])
    deps = root / "diagnostic_deps_20261019/transformers"
    conf_path = deps / "generation/configuration_utils.py"
    tree = ast.parse(conf_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GenerationConfig")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {}
    for node in init.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and
                isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "pop" and
                len(node.value.args) == 2 and isinstance(node.targets[0], ast.Attribute)):
            defaults[node.targets[0].attr] = ast.literal_eval(node.value.args[1])
    global_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                         and n.name == "_get_default_generation_params")
    global_defaults = ast.literal_eval(next(n.value for n in global_method.body if isinstance(n, ast.Return)))
    defaults.update(global_defaults)
    cfg_path = root / "official/pack_config.json"
    official_cfg = read(cfg_path)["text_config"]
    reconstructed = dict(defaults)
    reconstructed.update({k: v for k, v in official_cfg.items() if k in defaults and v is not None})
    reconstructed.update(max_new_tokens=1024, do_sample=False, use_cache=True,
                         eos_token_id=eos, pad_token_id=eos[0])
    sources = ["backend.py", "train_curated_shallow_20261019.py", "run_ptuning_v2_hybrid.py",
               "ptuning_v2_hybrid.py", "train_optimized_v2_20261019.py",
               "official/pack_config.json", "official/tokenizer.json",
               "diagnostic_deps_20261019/transformers/generation/configuration_utils.py",
               "diagnostic_deps_20261019/transformers/generation/utils.py",
               "diagnostic_deps_20261019/transformers/modeling_utils.py"]
    return dict(
        observed_historical_files=dict(tasks_sha256=sha(task_path),
          before_output_sha256=sha(root / old_review["run"] / "generated_native46.jsonl"),
          after_output_sha256=sha(root / new_review["snapshot"] / "generated_native46.jsonl"),
          identical_task_prompt_hashes=46, full20_documents_verified=46,
          before_max_output_tokens=max(len(o["token_ids"]) for o in before),
          after_max_output_tokens=max(len(o["token_ids"]) for o in after),
          all_first_eos_stop=True, neither_output_limit_reached=True,
          task_rows=46, question_groups=len({t["question_group"] for t in tasks}),
          historical_content_counts={
              "before": dict(Counter(c["category"] for c in old_review["cases"])),
              "after": dict(Counter(c["category"] for c in new_review["cases"]))},
          review_blind=False, independent_human_gold=False,
          prior_baseline_judgments_reused=True),
        reconstructed_from_local_source_not_runtime_dump=dict(
          transformers_version="5.6.2", generation_config=reconstructed,
          official_cfg_generation_relevant={k: v for k, v in official_cfg.items() if k in defaults},
          hf_construction="Qwen3_5ForCausalLM(cfg); GenerationConfig.from_model_config(cfg); no from_pretrained in backend",
          hf_effective_logits_processors_expected=[],
          why="Unit penalties, zero minimum/ngram and no forced/suppressed/bias tokens; sampling warpers disabled by do_sample=False.",
          hf_expected_stopping=["MaxLengthCriteria adjusted by max_new_tokens=1024", "EosTokenCriteria([248046,248044])"],
          custom_stopping="Append argmax, break at first EOS or 1024 generated tokens; no logits processors.",
          unknowns=["historical actual artifact/manifest.json config not present locally; official config is a reconstruction input, not a bytewise runtime manifest recheck",
                    "full historical runtime generation_config, applied logits processor list and package/kernel runtime receipt were not saved here",
                    "final trained 27B long cached-vs-full/HF-vs-explicit token parity not established by static source audit"],
          interpretation="No positive evidence of a logits-processor/stop defect. Distinct functions alone are not evidence of a bug; numerical parity remains a future preflight."),
        source_sha256={name: sha(root / name) for name in sources},
        semantic_grading_performed=False, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.source)
    with Path(args.out).open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["observed_historical_files"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
