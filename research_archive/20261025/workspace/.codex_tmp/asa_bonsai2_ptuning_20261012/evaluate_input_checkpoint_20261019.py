"""Snapshot a completed input-only checkpoint and evaluate without touching training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def launch(args):
    assert not args.out.exists()
    run = json.loads((args.source_run / "trained/run.json").read_text())
    assert run["data_sha256"] == {
        split: sha(args.data / (split + ".jsonl")) for split in ("train", "val")}
    events = [json.loads(line) for line in
              (args.source_run / "trained/progress.jsonl").read_text().splitlines()
              if line.strip()]
    # One step behind the last logged step: save follows logging in this trainer.
    last_step = max(r["step"] for r in events if r.get("event") == "optimizer_step")
    step = last_step - 1
    assert step > 0
    source = args.source_run / "trained" / f"checkpoint-{step}"
    free = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    assert int(free[1]) >= 22000, free
    args.out.mkdir(mode=0o700)
    adapter = args.out / "adapter_snapshot"
    adapter.mkdir()
    hashes = {}
    for name in ("prompt_encoder.safetensors", "adapter_config.json"):
        before = sha(source / name)
        shutil.copy2(source / name, adapter / name)
        assert before == sha(source / name) == sha(adapter / name)
        hashes[name] = before
    receipt = dict(
        kind="intermediate_checkpoint_native46_not_final_training_result",
        source_run=str(args.source_run), source_checkpoint=str(source),
        checkpoint_step=step, expected_training_steps=run["train_examples"] * run["epochs"],
        source_training_complete_at_snapshot=run["training_complete"],
        source_run_sha256=sha(args.source_run / "trained/run.json"),
        adapter_sha256=hashes, data=str(args.data), data_sha256=run["data_sha256"],
        gpu_uuid=free[0].strip(), launched_unix=time.time(),
        script_sha256=sha(__file__), backend_sha256=sha(ROOT / "backend.py"),
        original_training_unmodified=True, new_training_performed=False,
        answer_validation_enabled=False, input_truncation=False, evidence_count=20,
        historical_development_questions=True, blind_test=False, end_to_end=False,
    )
    save(args.out / "snapshot_receipt.json", receipt)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
               "--source-run", str(args.source_run), "--data", str(args.data),
               "--out", str(args.out)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=receipt["gpu_uuid"],
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    with (args.out / "worker.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(pid=process.pid, command=command)
    save(args.out / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def worker(args):
    import torch
    from tokenizers import Tokenizer
    from backend import PromptEncoder, load_backend

    receipt = json.loads((args.out / "snapshot_receipt.json").read_text())
    state = dict(stage="loading", pid=os.getpid(), completed=0, started_unix=time.time(),
                 checkpoint_step=receipt["checkpoint_step"])
    status_path = args.out / "status.json"
    save(status_path, state)
    try:
        assert sha(__file__) == receipt["script_sha256"]
        assert sha(ROOT / "backend.py") == receipt["backend_sha256"]
        assert torch.cuda.mem_get_info()[0] >= 20_500 * 1024**2
        torch.cuda.set_per_process_memory_fraction(0.95)
        for name, digest in receipt["adapter_sha256"].items():
            assert sha(args.out / "adapter_snapshot" / name) == digest
        assert sha(args.data / "val.jsonl") == receipt["data_sha256"]["val"]
        rows = [json.loads(line) for line in (args.data / "val.jsonl").read_text().splitlines()]
        assert len(rows) == len({r["id"] for r in rows}) == 46
        tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
        encoded = []
        for row in rows:
            assert len(row["documents"]) == row["evidence_count"] == 20 and not row["truncated"]
            assert all(d["text"] in row["prompt"] and
                       hashlib.sha256(d["text"].encode()).hexdigest() == d["sha256"]
                       for d in row["documents"])
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            assert ids and len(ids) + 16 + 1024 <= 32768
            encoded.append((row, ids))
        model, manifest = load_backend(ROOT / "artifact", dtype=torch.bfloat16)
        model.eval()
        encoder = PromptEncoder.load(args.out / "adapter_snapshot")
        encoder.eval()
        eos = [tokenizer.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")]
        assert all(isinstance(t, int) for t in eos)
        state.update(stage="generating", expected=len(encoded))
        save(status_path, state)
        path = args.out / "generated_native46.jsonl"
        with path.open("x") as stream:
            for row, ids in encoded:
                tick = time.monotonic()
                with torch.no_grad():
                    embedded = encoder.inject(model.model.embed_tokens(
                        torch.tensor([ids], device="cuda")))
                    generated = model.generate(
                        inputs_embeds=embedded,
                        attention_mask=torch.ones(embedded.shape[:2], device="cuda", dtype=torch.long),
                        max_new_tokens=1024, do_sample=False, use_cache=True,
                        eos_token_id=eos, pad_token_id=eos[0])
                tokens = generated[0].tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = None
                item = dict(
                    id=row["id"], question=row["question"], raw_text=raw, parsed=parsed,
                    valid_json=parsed is not None, token_ids=tokens,
                    prompt_sha256=hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    evidence_count=20, prompt_tokens=len(ids), prefix_tokens=encoder.length,
                    truncated=False, output_limit_reached=len(tokens) >= 1024 and tokens[-1] not in eos,
                    checkpoint_step=receipt["checkpoint_step"],
                    elapsed_seconds=time.monotonic() - tick)
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                stream.flush()
                state.update(completed=state["completed"] + 1, last_id=row["id"],
                             updated_unix=time.time())
                save(status_path, state)
                print(json.dumps({k: item[k] for k in
                                  ("id", "valid_json", "elapsed_seconds")}), flush=True)
        save(args.out / "generation_receipt.json", dict(
            completed=True, count=len(encoded), output_sha256=sha(path),
            no_input_truncation=True, answer_validation_enabled=False,
            semantic_accuracy_not_yet_reviewed=True,
            checkpoint_step=receipt["checkpoint_step"],
            intermediate_checkpoint_not_final_model=True,
            adapter_sha256=receipt["adapter_sha256"]["prompt_encoder.safetensors"],
            source_sha256=manifest["source_sha256"],
            historical_development_evidence=True, blind_test=False, end_to_end=False))
        state.update(stage="complete", finished_unix=time.time())
        save(status_path, state)
    except BaseException as exc:
        state.update(stage="failed", error=repr(exc), finished_unix=time.time())
        save(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("launch", "worker"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    for name in ("source_run", "data", "out"):
        setattr(args, name, getattr(args, name).resolve())
    (launch if args.command == "launch" else worker)(args)


if __name__ == "__main__":
    main()
