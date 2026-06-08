from __future__ import annotations

from pathlib import Path
from typing import Any

from asa_arknight_story_agent.inference.model_runtime.output_cleaning import sanitize_generation_output


class VllmRunner:
    backend_name = "vllm"

    def __init__(
        self,
        *,
        base_model_path: Path,
        lora_path: Path | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 12000,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repeat_penalty: float = 1.05,
        dtype: str = "auto",
        max_num_batched_tokens: int | None = None,
        enforce_eager: bool = False,
    ) -> None:
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.tensor_parallel_size = max(1, tensor_parallel_size)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.dtype = dtype
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enforce_eager = enforce_eager
        self._llm = None
        self._lora_request = None
        self._engine_init_error: Exception | None = None

    def describe_runtime(self) -> dict[str, Any]:
        return {
            "generator_backend": self.backend_name,
            "gguf_model_path": None,
            "base_model_path": str(self.base_model_path),
            "lora_path": str(self.lora_path) if self.lora_path else None,
            "tokenizer_path": str(self.lora_path)
            if self.lora_path and (self.lora_path / "tokenizer_config.json").exists()
            else str(self.base_model_path),
            "release_lora_artifact": "model/lora/asa-arknightstoryagent-4b-lora",
            "release_lora_artifact_type": "LoRA adapter",
            "recommended_runtime_model": str(self.base_model_path),
            "runtime_mode": "base_hf" if not self.lora_path else "base_hf_plus_lora_vllm",
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enforce_eager": self.enforce_eager,
        }

    def _ensure_engine(self):
        if self._llm is not None:
            return self._llm, self._lora_request
        if self._engine_init_error is not None:
            raise RuntimeError("vLLM engine initialization previously failed.") from self._engine_init_error
        if not self.base_model_path.exists():
            raise FileNotFoundError(
                "Base model path not found for vLLM: "
                f"{self.base_model_path}\n"
                "Please pass a real `--base-model` path, for example `model/qwen3.5-4b`."
            )
        if self.lora_path is not None and not self.lora_path.exists():
            raise FileNotFoundError(
                "LoRA path not found for vLLM: "
                f"{self.lora_path}\n"
                "Please pass a real LoRA adapter directory or omit `--lora-path`."
            )
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed in the current environment. "
                "Run `bash scripts/setup_gpu_reranker_qwen35_4b.sh`, or install vLLM for your CUDA version."
            ) from exc

        try:
            tokenizer_path = (
                self.lora_path
                if self.lora_path and (self.lora_path / "tokenizer_config.json").exists()
                else self.base_model_path
            )
            llm_kwargs: dict[str, Any] = {
                "model": str(self.base_model_path),
                "tokenizer": str(tokenizer_path),
                "trust_remote_code": True,
                "enable_lora": self.lora_path is not None,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "dtype": self.dtype,
                "disable_log_stats": True,
                "enforce_eager": self.enforce_eager,
            }
            if self.max_num_batched_tokens is not None:
                llm_kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
            self._llm = LLM(**llm_kwargs)
            if self.lora_path is not None:
                self._lora_request = LoRARequest("asa_lora", 1, str(self.lora_path))
            return self._llm, self._lora_request
        except Exception as exc:
            self._engine_init_error = exc
            raise

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        llm, lora_request = self._ensure_engine()
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM is not installed in the current environment. "
                "Run `bash scripts/setup_gpu_reranker_qwen35_4b.sh`, or install vLLM for your CUDA version."
            ) from exc

        sampling_params = SamplingParams(
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            repetition_penalty=repeat_penalty if repeat_penalty is not None else self.repeat_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            skip_special_tokens=False,
        )
        outputs = llm.generate(
            [prompt],
            sampling_params,
            use_tqdm=False,
            lora_request=lora_request,
        )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no generation output.")
        return sanitize_generation_output(outputs[0].outputs[0].text, prompt)
