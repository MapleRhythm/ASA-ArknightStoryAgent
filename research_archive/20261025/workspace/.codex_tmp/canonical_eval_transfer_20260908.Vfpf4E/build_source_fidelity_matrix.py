"""One declared prompt intervention, with identical evidence/model/protocol.

This is a development experiment, not a held-out production improvement claim.
No gold, reviewer feedback, reference answers, or example names enter prompts.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


RECIPE = "source_fidelity_v1"
RULES = (
    "source_fidelity_rules: hypothesis仅用于理解检索意图，不是已证实的事实；只以evidence_brief为事实依据。"
    "先辨清说话者、被评价者、转述对象和代词指向；别人转述的评价不等于转述者自己的立场，问话中的身份不属于问话者。"
    "不得把名字、年龄、生死状态、行为动机从一个人物移到另一个人物。"
    "保留原文的否定、条件、时间、目的和保护对象；不要把有条件的表述改成无条件结论。"
    "不能仅凭两件事相邻出现就添加因果或主仆等关系；没有明确依据时用原文能支持的具体关系。"
    "问题若含错误前提，应据证据澄清；需要推断时，把有限推断与直接事实分开，并给出所依赖的线索。"
    "回答必须覆盖问题各子问；问原因时写明原因，问身份时写明所找身份，问差异时分别说明双方。"
    "不要用引用原文替代final_answer中的必要解释。"
    "输出前逐条核对fact与其引用在主体、对象、条件和含义上是否一致；quote是对应E块连续原文的复制，不拼接、不改写。"
    "只写当前证据能支持且与问题有关的内容；证据足够时直接回答，不能仅因材料较长就继续检索。"
    "保持原有JSON schema，不输出这些核对步骤，不新增字段。"
)


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def transform(row, render):
    result = copy.deepcopy(row)
    result["parent_row_sha256"] = sha_text(json.dumps(row, ensure_ascii=False, sort_keys=True))
    result["id"] = row["id"] + "::" + RECIPE
    result["renderer"] = RECIPE
    result["user"] = row["user"] + "\n" + RULES
    result["runtime_prompt"] = render(row["system"], result["user"], assistant_prefix=row["assistant_prefix"])
    result["runtime_prompt_sha256"] = sha_text(result["runtime_prompt"])
    result["recipe"] = RECIPE
    result["recipe_sha256"] = sha_text(RULES)
    result["evaluation_only"] = True
    result["training_ready"] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "after/src"))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    source = json.loads(args.input.read_text())
    if not source["evaluation_only"] or any(r["template"] != "runtime_raw" for r in source["rows"]):
        raise ValueError("requires frozen runtime-raw evaluation matrix")
    rows = [transform(row, render_qwen_chat_prompt) for row in source["rows"]]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate input IDs")
    report = {
        "version": RECIPE, "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recipe_text": RULES, "recipe_sha256": sha_text(RULES), "rows": rows,
        "limitations": [
            "Generic prompt intervention informed by error analysis; development, not unseen holdout.",
            "Only instruction suffix changes; evidence, question, hypothesis, schema and assistant prefix remain.",
            "No reference answers, judge comments, gold or new retrieved material enter the prompt.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "recipe": RECIPE,
                      "source_matrix_sha256": report["source_matrix_sha256"]}))


if __name__ == "__main__":
    main()
