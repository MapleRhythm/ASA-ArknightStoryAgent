#!/usr/bin/env python3
"""Freeze evaluation-only Exx inputs; never create training targets."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PREV = REPO / "experiments/paper_idea1_alternative_proof_20260909/phase2_deep_audit"
sys.path.insert(0, str(REPO / "src"))
from asa_arknight_story_agent.inference.generation.exx_prompt import EXX_SYSTEM_PROMPT, render_exx_user_prompt


def sha(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def scope_rows():
    rows = []
    for world, (a, b, c) in enumerate([
        ("林舟", "周宁", "陈禾"), ("顾川", "许岚", "沈青"),
        ("方予", "陆安", "江月"), ("叶秋", "苏禾", "唐宁"),
    ]):
        families = [
            ("reported_owner", f"记录中明确持有“{c}支持提案”这一看法的是谁？只回答观点持有人，不把转述者当成持有人。",
             f"{a}转述：{b}认为{c}支持提案。{a}没有表达自己的看法。",
             f"{b}的看法是{c}支持提案；{a}只是复述这一看法，并未表示赞同。",
             f"{a}说听到有人认为{c}支持提案，但不知道这个人是谁；{b}没有发表意见。",
             f"{a}认为{c}支持提案；{b}只是复述这一看法，并未表示赞同。",
             b, a),
            ("negative_approval", f"记录所说的这一次申请，{a}是否同意了{b}的申请？",
             f"这一次，{a}没有同意{b}的申请。",
             f"{b}的这次申请未获{a}同意。",
             f"记录没有提供{a}是否同意{b}这次申请的信息。",
             f"这一次，{a}同意了{b}的申请。",
             f"{a}没有同意{b}的申请。", f"{a}同意了{b}的申请。"),
            ("argument_direction", f"在记录的昨晚那次救援中，是谁救了{c}？",
             f"昨晚那次救援中，{b}救了{c}；{a}负责登记。",
             f"{c}昨晚那次获救，是被{b}救下的；记录员是{a}。",
             f"记录提及{c}昨晚获救，但没有说明谁救了{c}；{a}和{b}均被提及但角色不明。",
             f"昨晚那次救援中，{a}救了{c}；{b}负责登记。",
             b, a),
            ("temporal_scope", "按记录，今日队长是谁？",
             f"昨日队长是{a}；今日队长换成了{b}。",
             f"{b}今日担任队长，{a}昨日担任该职务且今日已卸任。",
             f"昨日队长是{a}；记录没有提供今日队长是谁的信息，只提到{b}今日到场。",
             f"昨日队长是{b}；今日队长换成了{a}。",
             b, a),
            ("negated_saying", f"{a}是否说过“{b}反对提案”这句话？回答是否说过，不判断{b}的实际立场。",
             f"{a}从未说过“{b}反对提案”这句话。",
             f"关于“{b}反对提案”这句话，记录确认{a}没有说过。",
             f"记录只说{b}未表明立场，没有提供{a}是否说过“{b}反对提案”的信息。",
             f"{a}曾明确说过“{b}反对提案”这句话。",
             f"{a}没有说过这句话。", f"{a}说过这句话。"),
            ("intention_completion", f"截至记录时，{a}是否已经把档案交给了{b}？",
             f"{a}计划把档案交给{b}，但截至记录时尚未交付。",
             f"截至记录时，档案还没由{a}交到{b}手中，交付仅是计划。",
             f"{a}曾计划把档案交给{b}；记录没有提供截至现在是否交付的信息。",
             f"截至记录时，{a}已经把档案交给了{b}。",
             f"{a}尚未把档案交给{b}。", f"{a}已经把档案交给{b}。"),
        ]
        for family, q, proof_a, proof_b, broken, changed, answer, changed_answer in families:
            filler_a = f"记录补充：{c}带着一本蓝色笔记本。"
            filler_b = f"记录补充：{c}另带一支黑色钢笔。"
            for state, docs, status, expected, valid in [
                ("proof_a", [proof_a, filler_b], "sufficient", answer, [["E1"]]),
                ("proof_b", [filler_a, proof_b], "sufficient", answer, [["E2"]]),
                ("dual", [proof_a, proof_b], "sufficient", answer, [["E1"], ["E2"]]),
                ("scope_missing", [broken, filler_b], "insufficient", None, []),
                ("answer_changed", [changed, filler_b], "sufficient", changed_answer, [["E1"]]),
            ]:
                question = "以下为虚构记录，仅依据当前可见记录回答。" + q
                hypothesis = {"question": question, "query_type": "fact", "entities": [],
                              "keywords": [], "expected_answer_type": "short_text"}
                for round_value in ("1/2", "2/2"):
                    reference = {
                        "status": status,
                        "action": "answer_directly" if status == "sufficient" else
                                  "retrieve_more" if round_value == "1/2" else "abstain",
                        "answer": expected, "valid_support_sets": valid,
                        "label_authority": "agent-authored explicit controlled semantics, not human gold",
                        "exclude_from_training": True,
                        "semantics": "question answerability, not fixed-claim polarity",
                    }
                    evidence = "\n".join(f"[E{i}]\n{text}" for i, text in enumerate(docs, 1))
                    prompt = render_exx_user_prompt(question=question, hypothesis=hypothesis,
                                                    round_value=round_value, evidence_text=evidence)
                    messages = [{"role": "system", "content": EXX_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}]
                    rows.append({
                        "id": f"scope_{world}_{family}_{state}_{round_value[0]}",
                        "task_type": "grounded_action_generation", "system": EXX_SYSTEM_PROMPT,
                        "conversations": [{"from": "human", "value": prompt}],
                        "diagnostic": {
                            "evaluation_only": True, "suite": "scope_controlled",
                            "case": f"scope_{world}_{family}", "family": family,
                            "factors": {"state": state, "round": round_value},
                            "reference": reference, "reference_sha256": sha(reference),
                            "prompt_messages_sha256": sha(messages),
                            "input_fields": {"question": question, "hypothesis": hypothesis,
                                             "round": round_value, "evidence_ids": ["E1", "E2"],
                                             "documents": [
                                                 {"text": text, "text_sha256": sha(text)}
                                                 for text in docs]},
                            "limitations": "Six templates with renamed entities; A/B paraphrases, not independent natural multihop proofs. Lengths not exactly matched.",
                        }})
    return rows


def main():
    rows = scope_rows()
    for name in ("natural_controls", "synthetic_controls"):
        for row in load(PREV / f"data/{name}.json"):
            row["diagnostic"]["suite"] = name
            rows.append(row)
    if len(rows) != len({row["id"] for row in rows}):
        raise ValueError("duplicate ID")
    # Frozen technical smoke includes both sufficient and insufficient states.
    smoke_ids = [0, 6, 7, 8]
    smoke = [rows[i] for i in smoke_ids]
    for row in rows:
        diag = row["diagnostic"]
        messages = [{"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["conversations"][0]["value"]}]
        if sha(messages) != diag["prompt_messages_sha256"] or sha(diag["reference"]) != diag["reference_sha256"]:
            raise ValueError(f"content binding failure: {row['id']}")
        if not diag["reference"]["exclude_from_training"]:
            raise ValueError("not evaluation-only")
    write(ROOT / "input/eval_matrix.json", rows)
    write(ROOT / "input/smoke4.json", smoke)
    write(ROOT / "INPUT_MANIFEST.json", {
        "counts": dict(Counter(row["diagnostic"]["suite"] for row in rows)),
        "rows": len(rows), "matrix_sha256": hashlib.sha256((ROOT / "input/eval_matrix.json").read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_inputs": {name: hashlib.sha256((PREV / f"data/{name}.json").read_bytes()).hexdigest()
                         for name in ("natural_controls", "synthetic_controls")},
        "nonthinking": True, "train": False,
        "note": "Development mechanism experiment, not held-out project accuracy or training improvement",
    })
    print(json.dumps(load(ROOT / "INPUT_MANIFEST.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
