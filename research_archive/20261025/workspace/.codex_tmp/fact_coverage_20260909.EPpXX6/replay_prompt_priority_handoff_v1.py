"""Replay six saved final32 suffix states, never recomputing model scores."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot


def main():
    source, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert mode in {"baseline", "candidate", "canonical"} and not output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(source))
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.inference.evidence import prompt_ordering
    from asa_arknight_story_agent.inference.evidence.prompt_preparation import prepare_prompt_evidence
    from asa_arknight_story_agent.inference.evidence.prompt_similarity import dedupe_prompt_evidence_candidates
    from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text, prompt_evidence_score
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    assert Path(prompt_ordering.__file__).resolve() == (
        source / "asa_arknight_story_agent/inference/evidence/prompt_ordering.py").resolve()
    source_before, canonical_before = snapshot(source), snapshot(SOURCE)
    handoff_path = HERE / "susie_prompt_handoff_v1.json"
    prepath = HERE / "susie_prompt_handoff_preflight_v1.json"
    pairpath = HERE / "fact_coverage_pairs116_v1.matrix.json"
    configpath = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    frozen = {str(p): sha(p) for p in (handoff_path, prepath, pairpath, configpath, Path(__file__))}
    old, pre = load(handoff_path), load(prepath)
    cfg = load(configpath)
    assert cfg == pre["config"]
    inf = cfg["inference"]
    pipeline = SimpleNamespace(**{name: inf[name] for name in [
        "enable_evidence_pinning", "enable_mmr", "enable_crag_refinement", "enable_pyramid_order",
        "prompt_evidence_top_k", "mmr_lambda"]},
        web_context_config=SimpleNamespace(force_prompt_evidence=inf["web_context"]["force_prompt_evidence"]))
    assert not any(getattr(pipeline, n) for n in [
        "enable_evidence_pinning", "enable_mmr", "enable_crag_refinement", "enable_pyramid_order"])
    query = pre["question"]
    row = next(r for r in load(pairpath)["rows"]
               if r["question"] == query and r["renderer"] == "original_prompt_v1")
    hypothesis = HypothesisDocument(**row["hypothesis"])
    tokenizer_path = ROOT / "model/qwen3.5-4b"
    assets = {str(p): sha(p) for p in tokenizer_path.iterdir()
              if p.is_file() and p.suffix in {".json", ".jinja", ".model", ".txt"}}
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    baseline = None if mode == "baseline" else load(HERE / "prompt_priority_handoff_baseline_v1.json")
    if baseline is not None:
        assert baseline["baseline_reproduced_before_comparison"]
        assert baseline["inputs_sha256"][str(handoff_path)] == sha(handoff_path)
    phrase = "她依然没有放手"
    conditions = []
    for index, condition in enumerate(old["conditions"]):
        evidence = deepcopy(condition["final32"])
        unchanged_copy = deepcopy(evidence)
        selected = prepare_prompt_evidence(
            pipeline=pipeline, question=query, hypothesis=hypothesis, evidence=evidence)
        assert evidence == unchanged_copy
        if mode != "baseline":
            ranked = sorted(enumerate(evidence), key=lambda p: (-prompt_evidence_score(p[1]), p[0]))
            expected = dedupe_prompt_evidence_candidates([r for _, r in ranked])[:inf["prompt_evidence_top_k"]]
            assert selected == expected
            scores = [prompt_evidence_score(r) for r in selected]
            assert scores == sorted(scores, reverse=True)
        packing = []
        prompt, brief = build_conclusion_prompt(
            question=query, current_hypothesis=hypothesis, evidence=evidence, retrieval_trace=[],
            current_round=row["current_round"], max_retrieval_rounds=row["max_retrieval_rounds"],
            prompt_evidence_top_k=inf["prompt_evidence_top_k"], prompt_evidence=selected,
            evidence_max_chars_per_doc=inf["prompt_evidence_max_chars_per_doc"],
            evidence_max_total_chars=inf["prompt_conclusion_evidence_max_total_chars"],
            prompt_mode=inf["conclusion_prompt_mode"], grounding_mode=inf["answer_grounding_mode"],
            evidence_packing_trace=packing)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if mode == "baseline":
            assert selected == condition["selected"] and packing == condition["packing_trace"]
            assert prompt == condition["runtime_prompt"] and brief == condition["evidence_brief"]
            assert len(ids) == condition["input_tokens"]
        old_selected = condition["selected"]
        old_ids = [r["document"]["id"] for r in old_selected]
        new_ids = [r["document"]["id"] for r in selected]
        details = lambda rows: [
            {"document_id": r["document"]["id"], "score": prompt_evidence_score(r),
             "text_sha256": digest(best_prompt_text(r, prefer_direct=bool(r.get("prompt_prefer_clean_text")))),
             "text_chars": len(best_prompt_text(r, prefer_direct=bool(r.get("prompt_prefer_clean_text")))),
             "has_diagnostic_phrase": phrase in best_prompt_text(r)}
            for r in rows]
        conditions.append({
            "index": index, "scoring_condition": condition["scoring_condition"],
            "order_only_reserve6": condition["order_only_reserve6"],
            "question": query, "selected": selected, "selected_ids": new_ids,
            "selected_details": details(selected), "previous_selected_details": details(old_selected),
            "removed_document_ids": [i for i in old_ids if i not in new_ids],
            "added_document_ids": [i for i in new_ids if i not in old_ids],
            "unchanged_final32_sha256": digest(json.dumps(evidence, ensure_ascii=False, sort_keys=True)),
            "runtime_prompt": prompt, "runtime_prompt_sha256": digest(prompt),
            "evidence_brief": brief, "packing_trace": packing, "input_tokens": len(ids),
            "input_token_ids_sha256": digest(json.dumps(ids)),
            "fits_configured_output_budget": len(ids) + cfg["generator"]["max_tokens"] <= cfg["generator"]["ctx_size"],
            "previous_target_phrase_visible": condition["target_phrase_in_visible_prompt"],
            "target_phrase_visible": phrase in brief,
            "prompt_changed": prompt != condition["runtime_prompt"],
            "new_answer": None,
        })
    assert len(conditions) == 6
    if baseline is not None:
        assert all(a["unchanged_final32_sha256"] == b["unchanged_final32_sha256"]
                   for a, b in zip(conditions, baseline["conditions"]))
    assert snapshot(source) == source_before and snapshot(SOURCE) == canonical_before
    assert all(sha(Path(p)) == h for p, h in {**frozen, **assets}.items())
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "runtime_source": str(source), "runtime_source_sha256": source_before,
        "canonical_source_sha256": canonical_before,
        "inputs_sha256": frozen, "tokenizer_assets_sha256": assets,
        "conditions": conditions, "baseline_reproduced_before_comparison": True,
        "baseline_replay_sha256": None if baseline is None else sha(HERE / "prompt_priority_handoff_baseline_v1.json"),
        "all_inputs_unchanged": True, "new_model_calls": 0, "independent_questions": 1,
        "new_full_rag_recall": None, "new_certified_answer_accuracy": None,
        "training_ready": False, "phrase_only_used_for_diagnosis": True,
        "limitations": [
            "One previously inspected development question, six suffix conditions; not six independent questions.",
            "Saved final32 inputs only; no new sparse/dense/fusion, hypothesis, reranker or generator calls.",
            "A representative's higher score does not certify semantic relevance or preservation of every unique fact.",
            "Base-tokenizer budget check here is diagnostic, not an actual LoRA generator dispatch.",
        ],
    })
    print(json.dumps({
        "mode": mode,
        "conditions": [{k: r[k] for k in [
            "scoring_condition", "order_only_reserve6", "prompt_changed",
            "previous_target_phrase_visible", "target_phrase_visible",
            "removed_document_ids", "added_document_ids", "input_tokens",
            "fits_configured_output_budget"]} for r in conditions],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
