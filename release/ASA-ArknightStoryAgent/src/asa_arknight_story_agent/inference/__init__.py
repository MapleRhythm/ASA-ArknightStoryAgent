from __future__ import annotations

from typing import Any

from .model_runtime.runners import LlamaCppRunner, VllmRunner
from .pipeline.types import HypothesisDocument, InferenceResult

__all__ = ["CPUInferencePipeline", "HypothesisDocument", "InferenceResult", "LlamaCppRunner", "VllmRunner"]


def __getattr__(name: str) -> Any:
    if name == "CPUInferencePipeline":
        from . import cpu_pipeline

        value = getattr(cpu_pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
