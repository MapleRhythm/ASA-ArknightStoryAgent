"""Assemble a new persistent release without touching existing runtime files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = "0adb413f5ba8b40423341cfd0f1e17f2624684c43e9ff8c7035f2ba115426dd1"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--baseline-config", type=Path, required=True)
    args = p.parse_args()
    assert sha(args.adapter / "adapter_model.safetensors") == EXPECTED
    assert not args.destination.exists(), "Never overwrite an existing release"
    args.destination.mkdir(parents=True)
    for name in ("scripts", "src", "configs", "tests"):
        shutil.copytree(args.source / name, args.destination / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(args.source / "PROTOCOL_EXX.md", args.destination / "PROTOCOL_EXX.md")
    (args.destination / "model/lora").mkdir(parents=True)
    target = args.destination / "model/lora/exx_sft_v55_20260911"
    target.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "README.md"):
        shutil.copy2(args.adapter / name, target / name)
    assets = args.canonical / "release/ASA-ArknightStoryAgent"
    for relative in ("model/qwen3.5-4b", "model/reranker", "model/embeddings", "indexes"):
        (args.destination / relative).symlink_to((assets / relative).resolve(), target_is_directory=True)
    config = json.loads(args.baseline_config.read_text())
    config["generator"]["vllm"].update(
        lora_path="model/lora/exx_sft_v55_20260911", gpu_memory_utilization=0.62,
        enforce_eager=True, max_num_batched_tokens=4096,
    )
    config["generator"]["max_tokens"] = 1536
    config["inference"]["answer_grounding_mode"] = "evidence_id"
    (args.destination / "configs/runtime_exx_official.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    files = {}
    for directory in ("src", "scripts", "configs"):
        for path in sorted((args.destination / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path.relative_to(args.destination))] = sha(path)
    model_files = {str(path.relative_to(args.destination)): sha(path)
                   for path in sorted(target.iterdir()) if path.is_file()}
    base = args.destination / "model/qwen3.5-4b"
    for path in sorted(base.iterdir()):
        if path.is_file() and (path.suffix == ".safetensors" or path.name in
                              ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")):
            model_files[str(path.relative_to(args.destination))] = sha(path)
    lock = {"release": "exx-20260915-v1", "adapter_sha256": EXPECTED,
            "protocol": "grounded_action_exx_v1", "baseline_config_sha256": sha(args.baseline_config),
            "files": files, "model_files": model_files, "quality_gate": "pending_20_question_review"}
    (args.destination / "RELEASE_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"release": str(args.destination), "files": len(files), "adapter": EXPECTED}))


if __name__ == "__main__":
    main()
