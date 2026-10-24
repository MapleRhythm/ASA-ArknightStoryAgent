#!/usr/bin/env python3
"""Locked Exx entrypoint. Explicit experimental overrides use the generic CLI."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_release():
    lock = json.loads((ROOT / "RELEASE_LOCK.json").read_text())
    for relative, expected in lock["files"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Release file changed: {relative}")
    for relative, expected in lock["model_files"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Release model changed: {relative}")
    config = json.loads((ROOT / "configs/runtime_exx_official.json").read_text())
    assert config["inference"]["answer_grounding_mode"] == "evidence_id"
    assert config["inference"]["prompt_evidence_require_full_documents"] is True
    assert config["inference"]["prompt_evidence_top_k"] == 20
    return lock


def main():
    lock = verify_release()
    if sys.argv[1:] == ["--verify-only"]:
        print(json.dumps({"release": lock["release"], "verified": True,
                          "adapter_sha256": lock["adapter_sha256"]}, ensure_ascii=False))
        return
    forbidden = ("--runtime-config", "--lora-path", "--disable-lora", "--base-model",
                 "--answer-grounding-mode", "--backend", "--gguf-model",
                 "--prompt-evidence-top-k", "--prompt-evidence-max-chars-per-doc",
                 "--prompt-conclusion-evidence-max-total-chars", "--enable-crag-refinement",
                 "--enable-evidence-pinning", "--enable-mmr", "--enable-pyramid-order",
                 "--enable-web-context", "--ctx-size", "--max-tokens")
    if any(arg.split("=")[0] in forbidden for arg in sys.argv[1:]):
        raise SystemExit("Official model/protocol/evidence settings are locked; use run_cpu_inference.py for experiments.")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(ROOT / "src"))
    sys.argv = [str(ROOT / "scripts/run_cpu_inference.py"),
                "--runtime-config", str(ROOT / "configs/runtime_exx_official.json"),
                *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
