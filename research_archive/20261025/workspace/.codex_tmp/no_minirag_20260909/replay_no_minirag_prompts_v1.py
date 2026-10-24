"""Rebuild frozen prompts after deleting only the synthetic MiniRAG hint line."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys

from build_fact_coverage_pairs_v1 import HERE, ROOT, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

HINT_LINE = re.compile(r"(?m)^minirag_hints: [^\n]*\n")


def main():
    source, configpath, output = map(Path, sys.argv[1:4])
    assert not output.exists()
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(source))
    from transformers import AutoTokenizer
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    cfg = load(configpath)
    inf = cfg["inference"]
    paths = [
        HERE / "fact_coverage_pairs116_v1.matrix.json",
        HERE / "susie_handoff4_v1.matrix.json",
        configpath,
        Path(__file__),
    ]
    frozen, source_before = {str(p): sha(p) for p in paths}, snapshot(source)
    historical = [r for r in load(paths[0])["rows"] if r["renderer"] == "original_prompt_v1"]
    handoff = load(paths[1])["rows"]
    assert len(historical) == 58 and len(handoff) == 4
    rows = historical + handoff
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "model/qwen3.5-4b"), local_files_only=True, trust_remote_code=False)
    results = []
    for row in rows:
        assert len(HINT_LINE.findall(row["user"])) == 1
        expected_user = HINT_LINE.sub("", row["user"], count=1)
        expected_prompt = render_qwen_chat_prompt(
            row["system"], expected_user, assistant_prefix=row["assistant_prefix"])
        packing = []
        prompt, brief = build_conclusion_prompt(
            question=row["question"], current_hypothesis=HypothesisDocument(**row["hypothesis"]),
            evidence=row["prompt_evidence"], prompt_evidence=row["prompt_evidence"],
            retrieval_trace=[], current_round=row["current_round"],
            max_retrieval_rounds=row["max_retrieval_rounds"],
            prompt_evidence_top_k=inf["prompt_evidence_top_k"],
            evidence_max_chars_per_doc=inf["prompt_evidence_max_chars_per_doc"],
            evidence_max_total_chars=inf["prompt_conclusion_evidence_max_total_chars"],
            prompt_mode=inf["conclusion_prompt_mode"], grounding_mode=inf["answer_grounding_mode"],
            evidence_packing_trace=packing)
        assert brief == row["evidence_brief"]
        assert packing == row["packing_trace"]
        assert prompt == expected_prompt
        assert HINT_LINE.search(expected_user) is None
        assert "minirag_hints:" not in prompt
        old_ids = tokenizer.encode(row["runtime_prompt"], add_special_tokens=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(ids) < len(old_ids)
        results.append({
            "id": row["id"], "question": row["question"], "renderer": row["renderer"],
            "evidence_sha256": digest(brief), "packing_trace_sha256": digest(
                json.dumps(packing, ensure_ascii=False, sort_keys=True)),
            "old_prompt_sha256": digest(row["runtime_prompt"]), "new_prompt_sha256": digest(prompt),
            "old_tokens": len(old_ids), "new_tokens": len(ids), "removed_tokens": len(old_ids) - len(ids),
            "only_expected_hint_line_removed": True,
            "fits_configured_output_budget": len(ids) + cfg["generator"]["max_tokens"] <= cfg["generator"]["ctx_size"],
        })
    assert snapshot(source) == source_before and all(sha(Path(p)) == h for p, h in frozen.items())
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False,
        "states": 62, "independent_questions": 58, "historical_states": 58,
        "single_question_handoff_states": 4, "results": results,
        "all_evidence_and_packing_unchanged": True, "only_minirag_hint_line_removed": True,
        "all_fit_configured_output_budget": all(r["fits_configured_output_budget"] for r in results),
        "source_artifacts_sha256": frozen, "runtime_source_sha256": source_before,
        "new_model_calls": 0, "new_full_rag_recall": None, "new_answer_accuracy": None,
        "limitations": [
            "Prompt reconstruction only; it does not run retrieval or generation.",
            "The58 historical rows are development states, not a held-out production accuracy set.",
            "Four handoff rows belong to one repeated development question.",
            "Removal changes model input and requires fresh generation before any answer-quality claim.",
        ],
    })
    print(json.dumps({
        "states": len(results), "all_evidence_unchanged": True,
        "removed_tokens": {
            "min": min(r["removed_tokens"] for r in results),
            "max": max(r["removed_tokens"] for r in results),
        },
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
