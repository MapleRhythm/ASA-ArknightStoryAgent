"""Paired question/role-focused prompt probe on the fixed nine development cases."""
import copy
import hashlib
import json
from pathlib import Path
import sys


FOCUS = (
    "answer_focus_check: 作答前核对问题究竟要求谁、什么关系、什么原因或哪些异同。"
    "答案必须填入这些具体信息，不能只复述问题中的泛称。"
    "分清说话者、被转述的人、行为主体和行为对象，不把寻找者当作被寻找者，不把转述他人的观点当成本人的观点。"
    "保留原文中的否定、条件、时间和明确更正；比较题同时回答共同点和不同点。"
    "只写当前可见证据支持且回答本题所必需的事实，不添加背景猜测。"
    "原文只能确认部分信息时明确保留缺失，不用模糊的‘或’填补身份或原因。"
    "保持原JSON字段和引用规则，不输出本核对步骤；证据不足时沿用retrieve_more/abstain规则。"
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def build_rows(source, render):
    if len(source) != len({r["id"] for r in source}):
        raise ValueError("Duplicate source IDs")
    rows = []
    for source_row in source:
        seed = int(digest(source_row["id"])[:8], 16) % (2**31)
        variants = ("focus_control_v1", "question_focus_v1")
        if seed % 2:
            variants = tuple(reversed(variants))
        for variant in variants:
            row = copy.deepcopy(source_row)
            row["parent_record_id"] = source_row["id"]
            row["parent_row_sha256"] = digest(json.dumps(source_row, ensure_ascii=False, sort_keys=True))
            row["id"] += "::" + variant
            row["renderer"] = variant
            row["sampling_seed"] = seed
            row["evaluation_only"] = True
            row["training_ready"] = False
            if variant == "question_focus_v1":
                row["user"] += "\n" + FOCUS
                row["runtime_prompt"] = render(row["system"], row["user"], assistant_prefix=row["assistant_prefix"])
                row["runtime_prompt_sha256"] = digest(row["runtime_prompt"])
            rows.append(row)
    return rows


if __name__ == "__main__":
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "inferred_integrity_candidate_v2/src"))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    matrix = run / "after.matrix.json"
    source = json.loads(matrix.read_text())
    assert source["evaluation_only"] and len(source["rows"]) == 9
    rows = build_rows(source["rows"], render_qwen_chat_prompt)
    assert len(rows) == 18
    original = {r["id"]: r for r in source["rows"]}
    for row in rows:
        parent = original[row["parent_record_id"]]
        for key in (
            "question", "hypothesis", "system", "assistant_prefix", "evidence_brief",
            "evidence_sha256", "prompt_evidence", "packing_trace", "evidence_id_to_corpus_id",
        ):
            assert parent[key] == row[key], key
    report = {
        "version": "question_focus_paired18_v1", "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "focus_prompt": FOCUS, "rows": rows,
        "selection": "All nine existing strict development cases, no new cherry-picking by outcome.",
        "limitations": [
            "These are repeatedly examined development cases, not a held-out test.",
            "Prompt is informed by observed error types; no case-specific answer, gold, judge or hidden source is sent.",
            "Within each pair only the generic question/role-focus instruction changes; sampling seed is equal.",
            "This is not the earlier citation-feedback retry; no previous failure is asserted to the model.",
            "No new retrieval, model training, service restart or production prompt change.",
        ],
    }
    with (run / "question_focus_paired18_v1.matrix.json").open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_cases": 9, "paired_generations": 18, "production_changed": False}))
