"""CPU-only prompt/token verification; no retriever or generation model is loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    support = runpy.run_path(str(args.repo / "tests/test_full_evidence_contract.py"))
    mod, norm = support["module"], support["norm"]
    cfg = json.loads((args.repo / "configs/runtime_inference_gpu.json").read_text())
    inf, gen = cfg["inference"], cfg["generator"]
    from transformers import AutoTokenizer
    tokenizer_path = args.repo / gen["vllm"]["lora_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, trust_remote_code=True,
    )
    rows = [json.loads(line) for line in args.source.read_text().splitlines()]
    assert len(rows) == 6
    with args.output.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            hyp = mod.HypothesisDocument(**row["retrieval_trace"][-1]["hypothesis"])
            obj = mod.CPUInferencePipeline(
                retriever=types.SimpleNamespace(),
                generator=types.SimpleNamespace(max_tokens=gen["max_tokens"]),
                query_config=types.SimpleNamespace(max_retrieval_queries=0),
                prompt_evidence_top_k=inf["prompt_evidence_top_k"],
                prompt_evidence_require_full_documents=inf["prompt_evidence_require_full_documents"],
                prompt_evidence_max_chars_per_doc=inf["prompt_evidence_max_chars_per_doc"],
                prompt_conclusion_evidence_max_total_chars=inf["prompt_conclusion_evidence_max_total_chars"],
                enable_crag_refinement=inf["enable_crag_refinement"],
                conclusion_prompt_mode=inf["conclusion_prompt_mode"],
            )
            nested = [
                {"document": item, "evidence_chain_text": item.get("evidence_chain_text")}
                for item in row["evidence"]
            ]
            selected = obj.prepare_prompt_evidence(row["question"], hyp, nested)
            assert len(selected) == 20
            common = dict(
                prompt_evidence=selected,
                evidence_max_chars_per_doc=obj.prompt_evidence_max_chars_per_doc,
                evidence_max_total_chars=obj.prompt_conclusion_evidence_max_total_chars,
            )
            conclusion = mod.build_conclusion_prompt(
                row["question"], hyp, nested, row["retrieval_trace"],
                len(row["retrieval_trace"]), 2, 20, prompt_mode=inf["conclusion_prompt_mode"],
                **common,
            )
            direct = mod.build_answer_prompt(
                row["question"], hyp, nested, prompt_evidence_top_k=20, **common,
            )
            followup = mod.build_follow_up_hypothesis_prompt(
                question=row["question"], current_hypothesis=hyp, evidence=nested,
                unresolved_points=[], retrieval_trace=row["retrieval_trace"],
                previous_conclusion=types.SimpleNamespace(next_action="retrieve_more"),
                current_round=2, max_retrieval_rounds=2, prompt_evidence_top_k=20,
                prompt_evidence=selected, require_full_prompt_evidence=True,
            )
            records = []
            for stage, prompt, budget in [
                ("conclusion", conclusion, 1536), ("direct_answer", direct, 1536),
                ("follow_up_hypothesis", followup, 384),
            ]:
                for item in selected:
                    original = norm(mod.strip_internal_evidence_meta(item["document"]["clean_text"]))
                    assert original in norm(prompt), (index, stage, item["document"]["id"])
                tokens = tokenizer.encode(prompt, add_special_tokens=True)
                record = {
                    "stage": stage, "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens": len(tokens), "output_budget": budget,
                    "context_limit": gen["vllm"]["max_model_len"],
                    "fits": len(tokens) + budget <= gen["vllm"]["max_model_len"],
                    "complete_documents_verified": len(selected),
                }
                records.append(record)
            result = {
                "question": row["question"], "top_k": 20,
                "tokenizer_path": str(tokenizer_path),
                "evidence_ids": [item["document"]["id"] for item in selected],
                "stages": records,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({
                "question_index": index,
                "stages": [{k: v for k, v in x.items() if k != "prompt"} for x in records],
                "vicka_key_sentence_visible": "最好的朋友" in conclusion if index == 4 else None,
            }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
