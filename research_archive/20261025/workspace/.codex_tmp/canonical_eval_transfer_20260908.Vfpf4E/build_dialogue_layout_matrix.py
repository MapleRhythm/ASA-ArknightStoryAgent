"""Restore existing source line breaks without changing visible evidence text.

One intervention only: same selected blocks, IDs, source content, instructions,
schema and assistant prefix. No generated speaker labels or new evidence.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path


RECIPE = "original_dialogue_layout_v1"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def original_layout(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text.strip()


def transform(row, *, render, parse_blocks, original_text):
    before = parse_blocks(row["evidence_brief"])
    texts = {
        entry["evidence_id"]: original_layout(original_text(row["prompt_evidence"][entry["input_index"]]))
        for entry in row["packing_trace"] if entry["status"] == "included"
    }
    if set(texts) != set(before):
        raise ValueError("visible evidence mapping mismatch")
    for eid in before:
        if normalize_whitespace(texts[eid]) != normalize_whitespace(before[eid]):
            raise ValueError("original source differs beyond whitespace; refusing to introduce content")
    brief = "\n".join(f"[{eid}] {texts[eid]}" for eid in before)
    if len(brief) > 12000:
        raise ValueError("new layout exceeds canonical character budget; no blocks may be dropped")
    after = parse_blocks(brief)
    if set(after) != set(before) or any(
        normalize_whitespace(after[eid]) != normalize_whitespace(before[eid]) for eid in before
    ):
        raise ValueError("layout creates ambiguous E labels or changes visible content")
    if row["user"].count(row["evidence_brief"]) != 1:
        raise ValueError("evidence brief must occur exactly once")
    result = copy.deepcopy(row)
    result.update({
        "id": row["id"] + "::" + RECIPE, "renderer": RECIPE, "recipe": RECIPE,
        "parent_row_sha256": sha(json.dumps(row, ensure_ascii=False, sort_keys=True)),
        "evidence_brief": brief, "evidence_sha256": sha(brief),
        "user": row["user"].replace(row["evidence_brief"], brief, 1),
        "original_visible_blocks": len(before),
        "normalized_visible_content_unchanged": True,
        "source_body_linebreaks_before": sum(text.count("\n") for text in before.values()),
        "source_body_linebreaks_after": sum(text.count("\n") for text in after.values()),
    })
    used, count = 0, 0
    for entry in result["packing_trace"]:
        if entry["status"] == "included":
            rendered = f'[{entry["evidence_id"]}] {texts[entry["evidence_id"]]}'
            used += len(rendered) + bool(count)
            entry["block_chars"] = len(rendered)
            count += 1
        entry["budget_used_chars"] = used
    if used != len(brief):
        raise ValueError("layout packing trace does not match actual budget")
    result["runtime_prompt"] = render(
        result["system"], result["user"], assistant_prefix=result["assistant_prefix"],
    )
    result["runtime_prompt_sha256"] = sha(result["runtime_prompt"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "after/src"))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    from asa_arknight_story_agent.inference.evidence.texts import best_prompt_text
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    source = json.loads(args.input.read_text())
    if not source.get("evaluation_only") or any(r["template"] != "runtime_raw" for r in source["rows"]):
        raise ValueError("expected frozen runtime-raw evaluation")
    def text(item):
        return strip_internal_evidence_meta(best_prompt_text(
            item, prefer_direct=bool(item.get("prompt_prefer_clean_text")),
        ))
    rows = [transform(row, render=render_qwen_chat_prompt,
                      parse_blocks=evidence_id_text_map_from_prompt, original_text=text)
            for row in source["rows"]]
    report = {
        "version": RECIPE, "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "limitations": [
            "Only existing source line breaks restored; no speaker attribution or content synthesized.",
            "All visible source bodies, IDs and instructions verified unchanged except whitespace.",
            "Development experiment motivated by error analysis, not independent held-out validation.",
        ],
    }
    with args.output.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "rows": len(rows), "all_content_equivalent": all(r["normalized_visible_content_unchanged"] for r in rows),
        "rows_with_restored_layout": sum(r["source_body_linebreaks_after"] > r["source_body_linebreaks_before"] for r in rows),
    }))


if __name__ == "__main__":
    main()
