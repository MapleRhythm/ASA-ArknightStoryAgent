from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    def __init__(self, model_path: Path, device: str = "cpu", max_length: int = 1024) -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, documents: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        for offset in range(0, len(documents), batch_size):
            batch_docs = documents[offset : offset + batch_size]
            inputs = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits
            flat = logits.view(-1).detach().cpu().float().tolist()
            scores.extend(float(item) for item in flat)
        return scores
