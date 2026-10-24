"""Build prompts with one frozen 241 code version and fixed selected passages."""
import argparse
import dataclasses
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def candidate(value, index):
    doc_id = str(value.get("id") or "").strip()
    if not doc_id:
        raise ValueError("candidate lacks corpus ID")
    document = {key: value.get(key) for key in
                ("id", "activity_name", "story_name", "stage_code", "avg_tag", "source_path")}
    document["clean_text"] = str(value.get("clean_text") or "")
    document["search_text"] = str(value.get("search_text") or document["clean_text"])
    result = {"document": document, "doc_index": value.get("doc_index", index)}
    for key in ("score", "dense_score", "sparse_score", "minirag_score", "fusion_score",
                "rerank_score", "evidence_chain_score", "evidence_chain_model_score",
                "evidence_chain_roles", "evidence_chain_member_ids", "evidence_chain_text",
                "supplemental_source", "prompt_prefer_clean_text"):
        if value.get(key) is not None:
            result[key] = value[key]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("before", "after"), required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    source = run / args.variant / "src"
    sys.path.insert(0, str(source))
    from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt

    manifest = json.loads((run / "manifest.json").read_text())
    for relative, item in manifest["source_files"].items():
        if relative.startswith(args.variant + "/"):
            if hashlib.sha256((run / relative).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("frozen code or config hash changed")
    config = json.loads((run / args.variant / "configs/runtime_gpu_reranker_qwen35_4b.json").read_text())
    inf = config["inference"]
    rows = json.loads((run / "selected_input.json").read_text())
    output = []
    for row in rows:
        evidence = [candidate(value, i) for i, value in enumerate(row["selected_evidence"])]
        hyp = HypothesisDocument(**row["hypothesis"])
        packing = []
        kwargs = {}
        if "evidence_packing_trace" in inspect.signature(build_minimal_conclusion_prompt).parameters:
            kwargs["evidence_packing_trace"] = packing
        prompt, brief = build_minimal_conclusion_prompt(
            question=row["question"], current_hypothesis=hyp,
            evidence=evidence, prompt_evidence=evidence, current_round=1,
            max_retrieval_rounds=inf["max_retrieval_rounds"],
            prompt_evidence_top_k=inf["prompt_evidence_top_k"],
            evidence_max_chars_per_doc=inf["prompt_evidence_max_chars_per_doc"],
            evidence_max_total_chars=inf["prompt_conclusion_evidence_max_total_chars"],
            grounding_mode=inf["answer_grounding_mode"], **kwargs,
        )
        match = re.fullmatch(
            r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\n(.*)",
            prompt, re.DOTALL,
        )
        if not match:
            raise ValueError("unexpected runtime chat template")
        blocks = evidence_id_text_map_from_prompt(brief)
        if args.variant == "after":
            id_map = {entry["evidence_id"]: entry["source_id"] for entry in packing
                      if entry["status"] == "included"}
        else:
            if len(blocks) != len(evidence):
                raise ValueError("legacy mapping requires all selected blocks")
            id_map = {eid: item["document"]["id"] for eid, item in zip(blocks, evidence)}
        if list(id_map) != list(blocks):
            raise ValueError("E-ID mapping differs from actual rendered blocks")
        output.append({
            "id": f'{row["id"]}::{args.variant}::runtime_raw',
            "source_id": row["id"], "question": row["question"],
            "question_family": row["question_family"], "renderer": args.variant,
            "template": "runtime_raw", "system": match[1], "user": match[2],
            "assistant_prefix": match[3],
            "runtime_prompt": prompt, "runtime_prompt_sha256": text_sha(prompt),
            "evidence_brief": brief, "evidence_sha256": text_sha(brief),
            "evidence_id_to_corpus_id": id_map, "packing_trace": packing,
            "hypothesis": dataclasses.asdict(hyp), "prompt_evidence": evidence,
            "evaluation_only": True, "training_ready": False,
        })
    report = {
        "version": "asa_canonical_quote_matrix_v1", "evaluation_only": True,
        "training_ready": False, "question_count": len(output), "variant": args.variant,
        "source_manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "module_path": inspect.getfile(build_minimal_conclusion_prompt),
        "rows": output,
    }
    with (run / f"{args.variant}.matrix.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "variant": args.variant, "rows": len(output),
        "mean_evidence_chars": sum(len(row["evidence_brief"]) for row in output) / len(output),
        "max_evidence_chars": max(len(row["evidence_brief"]) for row in output),
        "skipped_blocks": sum(entry["status"] == "budget_skip" for row in output for entry in row["packing_trace"]),
    }))


if __name__ == "__main__":
    main()
