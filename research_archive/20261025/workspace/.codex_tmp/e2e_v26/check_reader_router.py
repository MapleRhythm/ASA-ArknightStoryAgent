from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from selector_protocol import CASE_IDS, blocks, reader_prompt, validate_selection


HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    obj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            obj.update(chunk)
    return obj.hexdigest()


def main():
    output = HERE / "postflight_v1.json"
    if output.exists():
        raise ValueError("postflight already exists")
    preflight, generated = load(HERE / "preflight.json"), load(HERE / "generations.json")
    assert generated["preflight_sha256"] == sha(HERE / "preflight.json")
    assert generated["postflight_files_unchanged"]
    assert [row["id"] for row in generated["rows"]] == CASE_IDS
    assert len(preflight["rows"]) == len(generated["rows"]) == 8
    assert all(sha(Path(path)) == expected for path, expected in preflight["sha256"].items())
    sys.path.insert(0, str(HERE / "src"))
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.generation.conclusion_generation import (
        parse_conclusion_output, normalize_minimal_conclusion_output,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(preflight["tokenizer"], trust_remote_code=True)
    summary = {kind: {"status": Counter(), "action": Counter()} for kind in ("selector", "control", "routed")}
    calls = []
    mechanical = []
    for source, row in zip(preflight["rows"], generated["rows"], strict=True):
        assert source["id"] == row["id"]
        selected = None
        if row["selector"]["status"] == "ok":
            selected = validate_selection(row["selector"]["payload"], source["evidence_brief"])
            assert selected == row["selector"]["selected_evidence"]
            original_blocks = dict(blocks(source["evidence_brief"]))
            assert all(original_blocks[eid] == text for eid, text in blocks(selected))
        expected_prompts = {
            "selector": source["selector_prompt"],
            "control": source["control_prompt"],
        }
        if selected:
            expected_prompts["routed"] = reader_prompt(source["empty_reader_prompt"], selected)
        for call in row["calls"]:
            assert call["actual_prompt"] == expected_prompts[call["kind"]]
            assert call["seed"] == source["sampling_seed"]
            assert call["prompt_sha256"] == digest(call["actual_prompt"])
            ids = tokenizer.encode(call["actual_prompt"], add_special_tokens=True)
            assert len(ids) == call["prompt_tokens"]
            assert digest(json.dumps(ids)) == call["prompt_token_ids_sha256"]
            assert len(ids) + 2048 <= preflight["runtime"]["max_model_len"]
            expected_lora = None if call["kind"] == "selector" else 1
            assert call["lora_id"] == call["returned_lora_id"] == expected_lora
            calls.append(call)
        assert [call["kind"] for call in row["calls"]] == [
            "selector", *[kind for kind in source["reader_order"] if kind in expected_prompts],
        ]
        record = {"id": row["id"], "selected_ids": row["selector"].get("payload", {}).get("evidence_ids")}
        for kind in summary:
            value = row[kind]
            summary[kind]["status"].update([value["status"]])
            if kind == "selector":
                continue
            if value["status"] != "ok":
                continue
            evidence = source["evidence_brief"] if kind == "control" else selected
            hyp = HypothesisDocument(**source["hypothesis"])
            normalized = normalize_minimal_conclusion_output(value["raw_output"], "minimal")
            assert normalized == value["normalized_output"]
            parsed = parse_conclusion_output(
                normalized, question=row["question"], current_hypothesis=hyp, max_round_reached=True,
            )
            assert asdict(parsed) == value["parsed"]
            checked = validate_conclusion_grounding(
                question=row["question"], hypothesis=hyp, evidence=[], conclusion=parsed,
                max_round_reached=True, mode="quote", evidence_prompt_text=evidence,
            )
            assert asdict(checked) == value["grounded"]
            summary[kind]["action"].update([checked.next_action])
            record[kind] = {
                "action": checked.next_action, "answer": checked.answer,
                "warnings": checked.grounding_warnings,
                "raw_action": parsed.next_action,
            }
        mechanical.append(record)
    assert len(calls) == generated["actual_call_count"]
    report = {
        "evaluation_only": True, "training_ready": False, "rows": 8,
        "preflight_sha256": sha(HERE / "preflight.json"),
        "generations_sha256": sha(HERE / "generations.json"),
        "summary": summary, "calls": len(calls),
        "finish_reasons": Counter(call["finish_reason"] for call in calls),
        "whole_original_blocks_verified": True, "selector_reason_not_sent_to_reader": True,
        "no_old_answers_or_labels_sent": True, "token_ids_and_lora_echo_verified": True,
        "source_and_model_hashes_unchanged": True,
        "mechanical_outcomes": mechanical,
        "semantic_review_complete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": summary, "calls": len(calls),
                      "finish_reasons": report["finish_reasons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
