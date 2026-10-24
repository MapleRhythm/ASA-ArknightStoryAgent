"""Input-only prompt encoder: audited data -> full-input gate -> train -> native46."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import annotation_protocol as p
import train_curated_deep_20261019 as data_audit

ROOT = Path(__file__).resolve().parent
CODE = ("train_curated_shallow_20261019.py", "train_curated_deep_20261019.py",
        "train_pilot.py", "memory_gate.py", "backend.py")


def prepare(data, source):
    original = data_audit.verify(source)
    data.mkdir(mode=0o700)
    for name in ("train.jsonl", "val.jsonl"):
        shutil.copy2(source / name, data / name)
    shutil.copytree(source / "review_snapshot", data / "review_snapshot")
    manifest = dict(original, evidence_count_per_example=20, truncated_documents=0,
                    train_val_story_overlap=[], train_val_source_family_overlap=[],
                    train_val_document_overlap=[], train_val_question_overlap=[],
                    review_required_before_training=False,
                    input_layer_training_data_copy_of=str(source.resolve()))
    p.atomic_json(data / "manifest.json", manifest)
    return data_audit.verify(data)


def evaluate(data, out):
    import torch
    from backend import PromptEncoder, load_backend
    from tokenizers import Tokenizer
    data_audit.verify(data)
    trained = out / "trained"
    run = json.loads((trained / "run.json").read_text())
    assert run["training_complete"] and run["backbone_bytewise_unchanged"]
    assert run["data_sha256"] == {s: p.file_sha(data / f"{s}.jsonl") for s in ("train", "val")}
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    rows = data_audit.read_rows(data / "val.jsonl")
    model, manifest = load_backend(ROOT / "artifact", dtype=torch.bfloat16)
    model.eval()
    encoder = PromptEncoder.load(trained / "final_adapter")
    encoder.eval()
    eos = [tokenizer.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
    eos = [t for t in eos if t is not None]
    assert eos
    path = out / "generated_native46.jsonl"
    with path.open("x") as stream:
        for row in rows:
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            assert len(ids) + encoder.length + 1024 <= 32768
            started = time.monotonic()
            with torch.no_grad():
                embeddings = encoder.inject(model.model.embed_tokens(
                    torch.tensor([ids], device="cuda")))
                generated = model.generate(
                    inputs_embeds=embeddings,
                    attention_mask=torch.ones(embeddings.shape[:2], device="cuda", dtype=torch.long),
                    max_new_tokens=1024, do_sample=False, use_cache=True,
                    eos_token_id=eos, pad_token_id=eos[0])
            tokens = generated[0].tolist()
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            item = dict(id=row["id"], question=row["question"], raw_text=raw,
                        parsed=parsed, valid_json=parsed is not None, token_ids=tokens,
                        prompt_sha256=p.sha(row["prompt"]), evidence_count=20,
                        prompt_tokens=len(ids), prefix_tokens=encoder.length,
                        truncated=False, output_limit_reached=len(tokens) >= 1024
                        and tokens[-1] not in eos,
                        elapsed_seconds=time.monotonic() - started)
            stream.write(p.compact(item) + "\n")
            stream.flush()
            print(p.compact(dict(event="generation", id=row["id"],
                                 valid_json=item["valid_json"])), flush=True)
    p.atomic_json(out / "generation_receipt.json", dict(
        completed=True, count=len(rows), output_sha256=p.file_sha(path),
        no_input_truncation=True, answer_validation_enabled=False,
        semantic_accuracy_not_yet_reviewed=True, historical_development_evidence=True,
        blind_test=False, end_to_end=False,
        adapter_sha256=p.file_sha(trained / "final_adapter/prompt_encoder.safetensors"),
        source_sha256=manifest["source_sha256"]))


def workflow(data, out):
    from run_ptuning_v2_hybrid import environment
    assert not out.exists()
    out.mkdir(mode=0o700)
    environment(out)
    status = dict(stage="preflight", pid=os.getpid(), started_unix=time.time(),
                  method="p_tuning_external_residual_mlp_input_only",
                  data_sha256=data_audit.verify(data)["data_sha256"],
                  code_sha256={n: p.file_sha(ROOT / n) for n in CODE},
                  production_modified=False)
    path = out / "workflow.json"

    def save(stage, **kwargs):
        status.update(stage=stage, **kwargs)
        p.atomic_json(path, status)
    save("preflight")
    commands = [
        ("memory_gate", [sys.executable, "-u", str(ROOT / "memory_gate.py"),
                         "--data", str(data), "--out", str(out / "memory_gate.json"), "--offload"]),
        ("training", [sys.executable, "-u", str(ROOT / "train_pilot.py"),
                      "--data", str(data), "--out", str(out / "trained"),
                      "--epochs", "1", "--lr", "0.00002", "--offload",
                      "--skip-initial-validation", "--validation-limit", "0",
                      "--gate", str(ROOT / "full_gate_bfloat16_bounded_20261013.json"),
                      "--memory-gate", str(out / "memory_gate.json")]),
        ("generation", [sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
                        "--data", str(data), "--out", str(out)]),
    ]
    try:
        for stage, command in commands:
            if stage == "training":
                gate = json.loads((out / "memory_gate.json").read_text())
                assert gate["completed"] and gate["activation_offload"]
                assert gate["data_sha256"] == status["data_sha256"]
                assert gate["prompt_gradient_norm"] > 0
            data_audit.verify(data)
            assert status["code_sha256"] == {n: p.file_sha(ROOT / n) for n in CODE}
            with (out / f"{stage}.log").open("x") as log:
                proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT)
                save(stage, child_pid=proc.pid, command=command)
                code = proc.wait()
            assert code == 0, f"{stage} exited {code}; inspect stage log"
        save("complete", finished_unix=time.time(), semantic_accuracy_not_yet_reviewed=True)
    except BaseException as exc:
        save("failed", failed_stage=status["stage"], error=str(exc), failed_unix=time.time())
        raise


def launch(data, out, gpu):
    data_audit.verify(data)
    assert not out.exists()
    free_info = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, free = free_info[0].strip(), int(free_info[1])
    assert free >= 22000
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow",
               "--data", str(data.resolve()), "--out", str(out.resolve())]
    receipt = dict(command=command, gpu=gpu, gpu_uuid=uuid, free_gpu_mib=free)
    reservation = out.with_suffix(".launch.json")
    with reservation.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="8",
               PYTORCH_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    with out.with_suffix(".log").open("x") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(pid=proc.pid, launched_at=p.datetime.now(p.timezone.utc).isoformat())
    p.atomic_json(reservation, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "launch", "workflow", "evaluate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    data = args.data.resolve()
    if args.command == "prepare":
        assert args.source
        print(p.compact(prepare(data, args.source.resolve())))
    else:
        assert args.out
        out = args.out.resolve()
        if args.command == "launch":
            print(p.compact(launch(data, out, args.gpu)))
        elif args.command == "workflow":
            workflow(data, out)
        else:
            evaluate(data, out)


if __name__ == "__main__":
    main()
