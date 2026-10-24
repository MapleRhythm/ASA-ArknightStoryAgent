"""Isolated reviewed-subset data builder and existing deep-prefix workflow."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import annotation_protocol as p
import glm_review_original_20261018 as review_source
import run_ptuning_v2_hybrid as deep

ROOT = Path(__file__).resolve().parent
SNAPSHOTS = (
    "READER_ERROR_V3_SPOTCHECK_20261017.json",
    "READER_ERROR_V3_INDIVIDUAL_REVIEW_20261017.json",
    "GLM_ORIGINAL_REVIEW_SUMMARY_20261019.json",
)
EXCLUDE_UNCERTAIN_READ_SCOPE = {200, 202, 203, 204, 205, 206}
KIND = "reviewed_subset_deep_prefix_20261019_v1"


def read_rows(path):
    return list(p.json_rows(path))


def select(rows, snapshots):
    previous, individual, glm = snapshots
    assert individual["source_sha256"] == glm["source_sha256"] == review_source.SOURCE_SHA
    assert glm["all186_api_reviewed"] and glm["full20_prompt_and_original_target_integrity_verified"]
    cases = previous["cases"] + individual["cases"]
    by_position = {i: r for i, r in enumerate(rows, 1)}
    # The early spotcheck numbers its 15 sampled cases, not source positions.
    assert {c["id"] for c in previous["cases"]} <= {r["id"] for r in rows}
    for case in individual["cases"] + glm["cases"]:
        assert by_position[case["position"]]["id"] == case["id"]
    blocked = {c["id"] for c in cases if c["judgment"] != "pass"}
    blocked |= {c["id"] for c in glm["cases"] if c["state"] != "accepted"}
    blocked |= {by_position[i]["id"] for i in EXCLUDE_UNCERTAIN_READ_SCOPE}
    good = {c["id"] for c in cases if c["judgment"] == "pass"}
    good |= {c["id"] for c in glm["cases"] if c["state"] == "accepted"}
    good -= blocked
    selected = [r for r in rows if r["id"] in good]
    assert len(selected) == 367
    return selected


def verify(data):
    from targeted_pt_v2 import verify_merged
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["kind"] == KIND and manifest["ready_for_training"]
    snapshot = data / "review_snapshot"
    for name, digest in manifest["review_snapshot_sha256"].items():
        assert p.file_sha(snapshot / name) == digest
    source = snapshot / "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"
    assert p.file_sha(source) == review_source.SOURCE_SHA
    rows = read_rows(source)
    selected = select(rows, [json.loads((snapshot / n).read_text()) for n in SNAPSHOTS])
    base = Path(manifest["base"])
    original_manifest = verify_merged(base)
    assert original_manifest["data_sha256"] == manifest["base_data_sha256"]
    train, val = {s: read_rows(data / f"{s}.jsonl") for s in ("train", "val")}.values()
    assert train == read_rows(base / "train.jsonl") + selected
    assert (data / "val.jsonl").read_bytes() == (base / "val.jsonl").read_bytes()
    assert len(train) == manifest["train"] == 1117
    assert len(val) == manifest["val"] == 46
    assert len({r["id"] for r in train + val}) == len(train + val)
    base_questions = {p.normalized_question(r["question"]) for r in read_rows(base / "train.jsonl")}
    new_questions = [p.normalized_question(r["question"]) for r in selected]
    assert len(set(new_questions)) == len(new_questions)
    assert not base_questions & set(new_questions)
    p.assert_split({"train": train, "val": val})
    for split, values in (("train", train), ("val", val)):
        assert p.file_sha(data / f"{split}.jsonl") == manifest["data_sha256"][split]
        for original in values:
            row = dict(original)
            row["teacher_messages"], _ = p.prompt_for(
                row["question"], row["round"], row["documents"])
            p.check_row(row)
            p.check_target(json.loads(row["target"]), row)
    return manifest


def build(data, base):
    from targeted_pt_v2 import verify_merged
    base_manifest = verify_merged(base)
    source = ROOT / "READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl"
    assert p.file_sha(source) == review_source.SOURCE_SHA
    selected = select(read_rows(source), [json.loads((ROOT / n).read_text()) for n in SNAPSHOTS])
    assert base_manifest["train"] == 750 and base_manifest["val"] == 46
    data.mkdir(mode=0o700)
    snapshot = data / "review_snapshot"
    snapshot.mkdir()
    for path in [source] + [ROOT / n for n in SNAPSHOTS]:
        shutil.copy2(path, snapshot / path.name)
    with (data / "train.jsonl").open("x") as stream:
        for row in read_rows(base / "train.jsonl") + selected:
            stream.write(p.compact(row) + "\n")
    shutil.copy2(base / "val.jsonl", data / "val.jsonl")
    manifest = dict(
        kind=KIND, base=str(base.resolve()), train=1117, val=46, new_train=367,
        ready_for_training=True, authorization="user_requested_training_and_evaluation",
        source_sha256=review_source.SOURCE_SHA,
        base_data_sha256=base_manifest["data_sha256"],
        data_sha256={s: p.file_sha(data / f"{s}.jsonl") for s in ("train", "val")},
        review_snapshot_sha256={f.name: p.file_sha(f) for f in snapshot.iterdir()},
        new_targeted_counts=dict(Counter(r["targeted_kind"] for r in selected)),
        original_labels_unchanged=True, revised_labels_used=False,
        selection="union of assistant pass and GLM pass, excluding every known fail, "
                  "needs_review, GLM flag, and positions200,202-206 with uncertain read scope",
        all486_semantically_reviewed=False, all_retained_labels_semantically_certified=False,
        independent_human_gold=False, experimental_training_authorized=True,
        val_bytewise_unchanged=True, input_truncation=False, production_modified=False,
    )
    p.atomic_json(data / "manifest.json", manifest)
    return verify(data)


def load_data(data, length):
    from tokenizers import Tokenizer
    manifest = verify(data)
    tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
    rows = {}
    for split in ("train", "val"):
        rows[split] = []
        for row in read_rows(data / f"{split}.jsonl"):
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
            target = tokenizer.encode(row["target"] + "<|im_end|>", add_special_tokens=False).ids
            assert prompt and target and len(prompt) + len(target) + length <= 32768
            rows[split].append((row, prompt, target))
    return manifest, tokenizer, rows


def launch(data, out, gpu):
    verify(data)
    assert not out.exists()
    info = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, free = info[0].strip(), int(info[1])
    assert free >= 19500
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "workflow",
               "--data", str(data.resolve()), "--out", str(out.resolve())]
    receipt = dict(gpu=gpu, gpu_uuid=uuid, free_gpu_mib=free, command=command,
                   production_modified=False, other_jobs_stopped=False)
    reservation = out.with_suffix(".launch.json")
    with reservation.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="8",
               TOKENIZERS_PARALLELISM="false",
               PYTORCH_ALLOC_CONF="expandable_segments:True")
    with out.with_suffix(".log").open("x") as log:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(pid=proc.pid, launched_at=p.datetime.now(p.timezone.utc).isoformat())
    p.atomic_json(reservation, receipt)
    return receipt


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "workflow":
        # Reuse the tested gate/train/evaluate implementation without changing
        # historical scripts or pretending its old 750-row audit applies here.
        sys.argv.pop(1)
        deep.load_data = load_data
        deep.CODE_FILES += ("train_curated_deep_20261019.py",)
        deep.main()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify", "launch"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=ROOT /
                        "targeted_pt_v2_20261014_attempt02/merged_data")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.data.resolve(), args.base.resolve())
    elif args.command == "verify":
        result = verify(args.data.resolve())
    else:
        assert args.out is not None
        result = launch(args.data.resolve(), args.out.resolve(), args.gpu)
    print(p.compact(result), flush=True)


if __name__ == "__main__":
    main()
