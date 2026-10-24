"""Read-only structure/token audit of frozen evidence, not answer accuracy."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "after/src"))
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=True)
    source = json.loads(args.matrix.read_text())
    rows = []
    for row in source["rows"]:
        blocks = evidence_id_text_map_from_prompt(row["evidence_brief"])
        originals = {
            entry["evidence_id"]: strip_internal_evidence_meta(best_prompt_text(
                row["prompt_evidence"][entry["input_index"]],
                prefer_direct=bool(row["prompt_evidence"][entry["input_index"]].get("prompt_prefer_clean_text")),
            ))
            for entry in row["packing_trace"] if entry["status"] == "included"
        }
        if set(originals) != set(blocks):
            raise ValueError("visible block trace mismatch")
        contained = {}
        for eid, text in blocks.items():
            supersets = [other for other, body in blocks.items()
                         if eid != other and text in body and (
                             len(body) > len(text) or int(other[1:]) < int(eid[1:])
                         )]
            if supersets:
                contained[eid] = supersets
        kept = {eid: text for eid, text in blocks.items() if eid not in contained}
        # Exact substring containment, not inferred semantic equivalence.
        if not all(any(text in other for other in kept.values()) for text in blocks.values()):
            raise ValueError("dedup diagnostic lost an original visible body")
        dedup_brief = "\n".join(f"[{eid}] {text}" for eid, text in kept.items())
        if row["runtime_prompt"].count(row["evidence_brief"]) != 1:
            raise ValueError("evidence brief is not unique in prompt")
        dedup_prompt = row["runtime_prompt"].replace(row["evidence_brief"], dedup_brief, 1)
        before_tokens = len(tokenizer.encode(row["runtime_prompt"], add_special_tokens=True))
        after_tokens = len(tokenizer.encode(dedup_prompt, add_special_tokens=True))
        rows.append({
            "id": row["id"], "question": row["question"],
            "visible_blocks": len(blocks), "exact_contained_block_ids": contained,
            "diagnostic_kept_original_ids": list(kept),
            "original_body_newlines": sum(text.count("\n") for text in originals.values()),
            "visible_body_newlines": sum(text.count("\n") for text in blocks.values()),
            "input_tokens": before_tokens, "dedup_diagnostic_input_tokens": after_tokens,
            "input_tokens_without_added_specials": len(tokenizer.encode(row["runtime_prompt"], add_special_tokens=False)),
            "fits_ctx10000_reserve2048": before_tokens + 2048 <= 10000,
            "dedup_diagnostic_fits_ctx10000_reserve2048": after_tokens + 2048 <= 10000,
            "all_original_visible_bodies_contained_in_kept": True,
            "diagnostic_only": True,
        })
    report = {
        "version": "canonical_prompt_structure_v1", "evaluation_only": True, "training_ready": False,
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_path": str(args.tokenizer), "rows": rows,
        "summary": {
            "rows": len(rows),
            "rows_with_lost_newlines": sum(r["original_body_newlines"] > r["visible_body_newlines"] for r in rows),
            "rows_with_exact_contained_blocks": sum(bool(r["exact_contained_block_ids"]) for r in rows),
            "reserved_overflow_before": sum(not r["fits_ctx10000_reserve2048"] for r in rows),
            "reserved_overflow_after_dedup_diagnostic": sum(
                not r["dedup_diagnostic_fits_ctx10000_reserve2048"] for r in rows),
        },
        "limitations": [
            "Diagnostic only: this does not generate answers or mutate runtime packing.",
            "Removed diagnostic blocks are exact substrings elsewhere; no sentence compression/truncation.",
            "Original E IDs are kept for diagnostics, not a deployable renumbered generation prompt.",
            "Evidence already flattened in historical fixed79 prompts cannot recover original dialogue lines.",
        ],
    }
    with args.output.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
