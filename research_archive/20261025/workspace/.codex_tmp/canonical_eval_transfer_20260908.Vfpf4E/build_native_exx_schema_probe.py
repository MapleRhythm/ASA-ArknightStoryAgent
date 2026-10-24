"""Only add explicit JSON object schema to the frozen native Exx prompt."""
import copy
import hashlib
import json
from pathlib import Path
import sys


SCHEMA = (
    'answer_directly_schema: {"next_action":"answer_directly","supported_facts":[{"fact":"","evidence_ids":["E1"]}]}\n'
    'abstain_schema: {"next_action":"abstain","reason":"现有证据不足以确认。"}\n'
    "schema_rules: supported_facts的每个元素必须是包含fact和evidence_ids的对象，不能是字符串；"
    "fact填写实际事实文本，evidence_ids填写当前有效E编号的数组。以上只规定字段和类型，不是参考答案。"
)


def transform(row, render):
    result = copy.deepcopy(row)
    result["parent_record_id"] = row["id"]
    result["id"] += "::explicit_schema_v1"
    result["renderer"] = "native_exx_explicit_schema_v1"
    result["user"] += "\n" + SCHEMA
    result["runtime_prompt"] = render(result["system"], result["user"], assistant_prefix=result["assistant_prefix"])
    result["runtime_prompt_sha256"] = hashlib.sha256(result["runtime_prompt"].encode()).hexdigest()
    return result


if __name__ == "__main__":
    run = Path(__file__).resolve().parent
    sys.path.insert(0, str(run / "quote_integrity_candidate_v1/src"))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    path = run / "strict9_native_exx_v1.matrix.json"
    parent = json.loads(path.read_text())
    rows = [transform(row, render_qwen_chat_prompt) for row in parent["rows"]]
    assert len(rows) == 9
    for old, new in zip(parent["rows"], rows):
        for field in ("system", "assistant_prefix", "question", "hypothesis", "evidence_brief",
                      "evidence_sha256", "prompt_evidence", "evidence_id_to_corpus_id", "grounding_mode"):
            assert old[field] == new[field]
    result = {
        "version": "native_exx_explicit_schema_v1", "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_example": SCHEMA, "rows": rows,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Only a JSON object shape example is added; no factual example or gold is provided.",
            "Same 9 development cases, evidence, model, validator, assistant prefix and 512-token generation budget.",
            "Improved schema compliance would not by itself establish improved semantic correctness.",
        ],
    }
    with (run / "strict9_native_exx_explicit_schema_v1.matrix.json").open("x") as output:
        output.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": 9, "only_explicit_object_schema_added": True}))
