"""Strict accepted-subset gate before the unchanged external-encoder trainer."""
import argparse
import json
import math
from pathlib import Path
import sys

from audit_fresh_subset import file_sha, require, verify
import train_pilot


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--memory-gate", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    review = verify(args.data)
    gate = json.loads(args.memory_gate.read_text())
    require(gate["completed"] and gate["activation_offload"], "longest input gate failed")
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    require(gate["data_sha256"] == data_manifest["data_sha256"], "memory gate data mismatch")
    require(gate["backend_sha256"] == file_sha(Path(__file__).with_name("backend.py")),
            "memory gate backend mismatch")
    require(math.isfinite(gate["prompt_gradient_norm"]) and gate["prompt_gradient_norm"] > 0
            and gate["evidence_count"] == 20 and not gate["truncated"],
            "invalid memory gate gradient or evidence")
    require("--offload" in sys.argv, "offload required for reviewed full20 data")
    receipt = dict(
        review, kind="reviewed_subset_training_preflight",
        data_manifest_sha256=file_sha(args.data / "manifest.json"),
        memory_gate_sha256=file_sha(args.memory_gate),
        trainer_sha256=file_sha(Path(train_pilot.__file__)),
        wrapper_sha256=file_sha(__file__),
        production_modified=False,
    )
    receipt_path = args.out.parent / (args.out.name + ".reviewed_preflight.json")
    if args.resume:
        require(json.loads(receipt_path.read_text()) == receipt, "review receipt changed on resume")
    else:
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    train_pilot.main()
    run_path = args.out / "run.json"
    run = json.loads(run_path.read_text())
    require(run["training_complete"], "training did not complete")
    run["reviewed_subset_preflight"] = receipt
    run_path.write_text(json.dumps(run, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
