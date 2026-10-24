"""Paired development probe: ordinary resampling vs short quote-copy feedback.

Selection uses only mechanical grounding failure, never semantic labels.
No prior candidate answer, gold, judge feedback or hidden source enters prompts.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys


FEEDBACK = (
    "retry_notice: 上一次拟回答未通过引用校验，未展示给用户。本次仅使用上方相同的evidence_brief重新回答。"
    "先选能直接支持必要结论的连续原文，再写fact与final_answer；quote逐字复制，不改写、不拼接，不自行加入省略号。"
    "evidence_id必须是该段quote实际所在的[E编号]。每段quote不超过80字，每个fact最多2段，总引文不超过400字。"
    "只保留回答问题必需且有证据的事实，仍须说明问题要求的原因或关系；不能把题干当证据。"
    "证据不足时保留原有retrieve_more/abstain规则。保持原JSON schema，不输出核对步骤。"
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def selected_ids(rows):
    if len(rows) != len({r["id"] for r in rows}):
        raise ValueError("duplicate analysis IDs")
    return {
        r["id"] for r in rows
        if r.get("status") == "ok"
        and r["parsed"]["next_action"] == "answer_directly"
        and r["grounded"]["next_action"] != "answer_directly"
    }


def paired_rows(source, selected, render):
    if len(source) != len({r["id"] for r in source}):
        raise ValueError("duplicate matrix IDs")
    if not selected <= {r["id"] for r in source}:
        raise ValueError("selected ID absent from matrix")
    result = []
    for row in source:
        if row["id"] not in selected:
            continue
        seed = int(digest(row["id"])[:8], 16) % (2**31)
        # Alternate variant order without changing each pair's explicit seed.
        variants = ("retry_control_v1", "retry_quote_feedback_v1")
        if seed % 2:
            variants = tuple(reversed(variants))
        for recipe in variants:
            target = copy.deepcopy(row)
            target["parent_record_id"] = row["id"]
            target["parent_row_sha256"] = digest(json.dumps(row, ensure_ascii=False, sort_keys=True))
            target["id"] = row["id"] + "::" + recipe
            target["renderer"] = recipe
            target["sampling_seed"] = seed
            target["evaluation_only"] = True
            target["training_ready"] = False
            if recipe == "retry_quote_feedback_v1":
                target["user"] = row["user"] + "\n" + FEEDBACK
                target["runtime_prompt"] = render(
                    row["system"], target["user"], assistant_prefix=row["assistant_prefix"],
                )
                target["runtime_prompt_sha256"] = digest(target["runtime_prompt"])
            result.append(target)
    return result


if __name__ == "__main__":
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "after/src"))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    matrix_path = run / "fixed79.current_quote.matrix.json"
    analysis_path = run / "fixed79_configured_lora_ctx12288/validation_after_v2.json"
    source = json.loads(matrix_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    assert source["evaluation_only"] and len(source["rows"]) == len(analysis["results"]) == 79
    assert {r["id"] for r in source["rows"]} == {r["id"] for r in analysis["results"]}
    assert analysis["matrix_sha256"] == hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    selected = selected_ids(analysis["results"])
    assert len(selected) == 34
    rows = paired_rows(source["rows"], selected, render_qwen_chat_prompt)
    assert len(rows) == len({r["id"] for r in rows}) == 68
    report = {
        "version": "same_evidence_retry_paired68_v1", "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "source_analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feedback_text": FEEDBACK, "feedback_sha256": digest(FEEDBACK),
        "original_records": 79, "selected_rejected_records": 34,
        "selection": "parsed answer_directly and grounded not answer_directly; no semantic labels used",
        "rows": rows,
        "limitations": [
            "Development cohort selected from Transformers failures; not unseen or representative E2E.",
            "Both variants use native vLLM sampling with identical explicit seed per source case.",
            "Control is an ordinary additional generation, not the original Transformers answer.",
            "All visible evidence, question, hypothesis, round, schema and adapter remain unchanged.",
            "No previous answer, judge result, gold or hidden source is placed into the prompt.",
            "Additional accepted answers are not correct answers until source-reviewed.",
        ],
    }
    with (run / "same_evidence_retry_paired68_v1.matrix.json").open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"original_records": 79, "selected_records": 34, "paired_generations": len(rows)}))
