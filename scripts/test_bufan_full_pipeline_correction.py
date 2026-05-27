#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from goldenglow.config import EMBEDDING_MODEL_DIR, MINIRAG_GRAPH_PATH, RERANKER_MODEL_DIR, QueryConfig
from goldenglow.inference.cpu_pipeline import CPUInferencePipeline
from goldenglow.retrieval.hybrid import ArknightsHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DriftConclusionGenerator:
    backend_name = "fake-drift-conclusion"

    def __init__(self) -> None:
        self.max_tokens = 4096
        self.calls: list[str] = []

    def describe_runtime(self) -> dict[str, object]:
        return {"generator_backend": self.backend_name, "runtime_mode": "test"}

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
    ) -> str:
        del max_tokens, temperature, top_p, repeat_penalty
        self.calls.append(prompt)
        if "task: user_question_hypothesis_generation" in prompt:
            return json.dumps(
                {
                    "question": "真龙为什么要启动不反？",
                    "intent": "plot_reasoning",
                    "query_type": "causality",
                    "entities": ["真龙", "不反"],
                    "keywords": ["真龙", "启动", "不反", "岁陵", "危机", "代价", "性命"],
                    "expected_answer_type": "原因/动机",
                    "dialogue_context": "",
                },
                ensure_ascii=False,
            )
        if "请根据以下信息生成当前阶段结论" in prompt:
            return json.dumps(
                {
                    "question": "真龙为什么要启动不反？",
                    "next_action": "answer_directly",
                    "answer": "真龙启动不反是为了证明自己的能力，并推动大炎的发展。",
                    "missing_slots": [],
                    "clarification_question": "",
                    "follow_up_hypothesis": None,
                },
                ensure_ascii=False,
            )
        return "现有证据不足以确认。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-pipeline correction regression for the hard 不反 case.")
    parser.add_argument("--runtime-config", type=Path, default=PROJECT_ROOT / "api-mode" / "runtime_api.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--with-reranker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    retrieval_cfg = runtime_config.get("retrieval", {})
    inference_cfg = runtime_config.get("inference", {})
    minirag_mode_weights = retrieval_cfg.get("minirag_mode_weights") or {}
    if not isinstance(minirag_mode_weights, dict):
        minirag_mode_weights = {}

    configured_minirag = Path(str(retrieval_cfg.get("minirag_index_path") or MINIRAG_GRAPH_PATH))
    minirag_index_path = configured_minirag if configured_minirag.is_absolute() else PROJECT_ROOT / configured_minirag
    reranker_model = RERANKER_MODEL_DIR if args.with_reranker else None
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=EMBEDDING_MODEL_DIR,
        reranker_model_path=reranker_model,
        minirag_index_path=minirag_index_path,
        device=args.device,
    )
    generator = DriftConclusionGenerator()
    pipeline = CPUInferencePipeline(
        retriever=retriever,
        generator=generator,
        query_config=QueryConfig(
            dense_top_k=int(retrieval_cfg.get("dense_top_k", 120)),
            sparse_top_k=int(retrieval_cfg.get("sparse_top_k", 120)),
            minirag_top_k=int(retrieval_cfg.get("minirag_top_k", 120)),
            fusion_top_k=int(retrieval_cfg.get("fusion_top_k", 80)),
            rerank_top_k=int(retrieval_cfg.get("rerank_top_k", 32)),
            minirag_weight=float(retrieval_cfg.get("minirag_weight", 0.35)),
            minirag_mode_weights={str(key): float(value) for key, value in minirag_mode_weights.items()},
            minirag_fusion_mode=str(retrieval_cfg.get("minirag_fusion_mode", "score")),
            reranker_candidate_top_k=int(retrieval_cfg.get("reranker_candidate_top_k", 120)),
            rerank_batch_size=int(retrieval_cfg.get("rerank_batch_size", 4)),
        ),
        max_retrieval_rounds=1,
        prompt_evidence_top_k=int(inference_cfg.get("prompt_evidence_top_k", 12)),
        enable_mmr=bool(inference_cfg.get("enable_mmr", True)),
        mmr_lambda=float(inference_cfg.get("mmr_lambda", 0.72)),
        enable_pyramid_order=bool(inference_cfg.get("enable_pyramid_order", True)),
        enable_crag_refinement=False,
        answer_grounding_mode=str(inference_cfg.get("answer_grounding_mode", "weak")),
    )
    result = pipeline.run("真龙为什么要启动不反？")
    answer = result.answer
    conclusion_prompt = next((prompt for prompt in generator.calls if "请根据以下信息生成当前阶段结论" in prompt), "")
    checks = {
        "has_suiling_crisis": "岁陵里的那场危机" in answer or ("岁陵" in answer and "危机" in answer),
        "has_life_cost": "真龙本人的性命" in answer or ("性命" in answer and "代价" in answer),
        "removed_background_drift": "证明自己的能力" not in answer and "推动大炎的发展" not in answer,
        "prompt_has_suiling_crisis": "岁陵" in conclusion_prompt and "危机" in conclusion_prompt,
        "prompt_has_life_cost": "性命" in conclusion_prompt and "代价" in conclusion_prompt,
    }
    print(
        json.dumps(
            {
                "answer": answer,
                "checks": checks,
                "generator_calls": len(generator.calls),
                "top_evidence": [
                    {
                        "id": item.get("id"),
                        "stage_code": item.get("stage_code"),
                        "anchors": [
                            anchor
                            for anchor in ("岁陵", "危机", "性命", "代价", "不反", "太尉", "莫佚")
                            if anchor in str(item.get("evidence_chain_text") or item.get("clean_text") or "")
                        ],
                    }
                    for item in result.evidence[:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
