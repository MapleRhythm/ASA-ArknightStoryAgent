from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from asa_arknight_story_agent.inference.model_runtime.output_cleaning import sanitize_generation_output


class LlamaCppRunner:
    backend_name = "llama.cpp"

    def __init__(
        self,
        *,
        llama_cli_path: Path,
        gguf_model_path: Path,
        lora_path: Path | None = None,
        threads: int | None = None,
        ctx_size: int = 12000,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
        gpu_layers: str | int | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        ubatch_size: int | None = None,
        flash_attn: str | None = None,
    ) -> None:
        self.llama_cli_path = llama_cli_path
        self.gguf_model_path = gguf_model_path
        self.lora_path = lora_path
        self.threads = threads or max(1, os.cpu_count() or 1)
        self.ctx_size = ctx_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.gpu_layers = gpu_layers
        self.device = device
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.flash_attn = flash_attn

    def _has_gpu_backend(self) -> bool:
        bin_dir = self.llama_cli_path.parent
        for pattern in ("libggml-cuda*", "libggml-vulkan*", "libggml-hip*", "libggml-sycl*"):
            if any(bin_dir.glob(pattern)):
                return True
        return False

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "generator_backend": self.backend_name,
            "gguf_model_path": str(self.gguf_model_path),
            "base_model_path": None,
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "release_lora_artifact": "model/lora/asa-arknightstoryagent-4b-lora",
            "release_lora_artifact_type": "LoRA adapter",
            "recommended_runtime_model": "model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf",
            "runtime_mode": "merged_gguf" if not self.lora_path else "base_gguf_plus_lora_gguf",
            "llama_device": self.device,
            "gpu_layers": self.gpu_layers,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(
                "llama.cpp CLI not found: "
                f"{self.llama_cli_path}\n"
                "Please pass the real `--llama-cli` path, for example `/abs/path/to/llama.cpp/build/bin/llama-cli`."
            )
        if not self.gguf_model_path.exists():
            raise FileNotFoundError(
                "GGUF model not found: "
                f"{self.gguf_model_path}\n"
                "Please pass the real `--gguf-model` path to a converted GGUF file.\n"
                "Recommended runtime artifact in this repo: "
                "`model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf`."
            )
        if self.lora_path is not None and not self.lora_path.exists():
            raise FileNotFoundError(
                "LoRA path not found: "
                f"{self.lora_path}\n"
                "Please pass the real `--lora-path` directory or omit this option."
            )
        if self.lora_path is not None and self.lora_path.is_dir():
            raise FileNotFoundError(
                "llama.cpp does not load Hugging Face LoRA directories directly: "
                f"{self.lora_path}\n"
                "Use a GGUF LoRA adapter file, or omit `--lora-path` and run the merged GGUF "
                "`model/gguf/qwen3.5-4b-lora-merged-q4_k_m.gguf`."
            )
        if self.device and self.device.lower() not in {"cpu", "none"} and not self._has_gpu_backend():
            raise RuntimeError(
                "The selected llama.cpp binary does not include a GPU backend.\n"
                f"Binary: {self.llama_cli_path}\n"
                "Current build appears CPU-only, so generation will be extremely slow.\n"
                "Rebuild llama.cpp with CUDA/HIP/Vulkan support, or switch to the `vllm` backend."
            )
        cmd = [
            str(self.llama_cli_path),
            "-m",
            str(self.gguf_model_path),
            "--no-warmup",
            "--no-display-prompt",
            "--simple-io",
            "--no-perf",
            "--no-conversation",
            "--no-jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "-t",
            str(self.threads),
            "-c",
            str(self.ctx_size),
            "-n",
            str(max_tokens if max_tokens is not None else self.max_tokens),
            "--temp",
            str(temperature if temperature is not None else self.temperature),
            "--top-p",
            str(top_p if top_p is not None else self.top_p),
            "--repeat-penalty",
            str(repeat_penalty if repeat_penalty is not None else self.repeat_penalty),
            "-p",
            prompt,
        ]
        if self.device:
            cmd.extend(["--device", self.device])
        if self.gpu_layers is not None:
            cmd.extend(["--gpu-layers", str(self.gpu_layers)])
        if self.batch_size is not None:
            cmd.extend(["--batch-size", str(self.batch_size)])
        if self.ubatch_size is not None:
            cmd.extend(["--ubatch-size", str(self.ubatch_size)])
        if self.flash_attn is not None:
            cmd.extend(["--flash-attn", self.flash_attn])
        if self.lora_path:
            cmd.extend(["--lora", str(self.lora_path)])

        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "llama.cpp inference failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{completed.stderr.strip()}\n"
                f"stdout:\n{completed.stdout.strip()}"
            )
        return sanitize_generation_output(completed.stdout, prompt)
