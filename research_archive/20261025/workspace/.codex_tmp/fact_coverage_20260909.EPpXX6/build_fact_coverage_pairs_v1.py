"""Freeze one answer-independent evidence state per question, two prompt arms."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE = ROOT / "release/ASA-ArknightStoryAgent/src"

COVERAGE_RULES = (
    "\nanswer_fact_coverage_rules: supported_facts是最终展示答案的依据，不是背景摘录。"
    "先确定用户具体问了哪些信息，再选择足以回答这些信息的原子事实；即使不看final_answer，"
    "这些事实也应共同回答问题。问题问身份或名称时应明确写出证据支持的身份或名称，"
    "不要只重复题干中的泛称；问题问原因或机制时，应写出证据实际给出的直接原因、"
    "条件或过程，不要只重复结果。明确的上级命令可构成执行者行动的直接原因，"
    "不强求另一个主体的更深动机。每条fact的quote应支撑该条fact的全部断言，"
    "不能借用别条fact的引文。保留谁对谁、时间阶段、否定与可能性，不把推测改成确定事实，"
    "不把并列现象改成因果。final_answer中必要且有据的信息也必须写入对应fact并绑定引文；"
    "未找到支持时不要补造。仍遵守上方JSON字段、引文长度、事实数量和继续检索/弃答规则，"
    "不增加检查过程或新的输出字段。\n"
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def normalized_question(text):
    return re.sub(r"\s+", "", text).strip("？?。！!，,；;：:\"'")


def select_states(rows):
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate source record")
    groups = {}
    for row in rows:
        key = normalized_question(row["question"])
        if not key:
            raise ValueError("empty normalized question")
        groups.setdefault(key, []).append(row)
    # Neither baseline outcomes nor evidence contents enter selection.
    return [
        min(groups[key], key=lambda r: digest("fact-coverage-state-v1:" + r["id"]))
        for key in sorted(groups, key=lambda q: digest("fact-coverage-order-v1:" + q))
    ]


def build_pairs(selected, renderer, development_questions):
    rows, selection = [], []
    for index, original in enumerate(selected):
        assert renderer(original["system"], original["user"],
                        assistant_prefix=original["assistant_prefix"]) == original["runtime_prompt"]
        seed = int(digest("fact-coverage-seed-v1:" + original["id"])[:8], 16) % (2**31)
        variants = ["original_prompt_v1", "fact_coverage_prompt_v1"]
        if index % 2:
            variants.reverse()
        for variant in variants:
            candidate = deepcopy(original)
            candidate.update(
                id=original["id"] + "::" + variant,
                parent_record_id=original["id"], renderer=variant, sampling_seed=seed,
                evaluation_only=True, training_ready=False,
            )
            candidate["user"] = original["user"] + (
                COVERAGE_RULES if variant == "fact_coverage_prompt_v1" else ""
            )
            candidate["runtime_prompt"] = renderer(
                candidate["system"], candidate["user"],
                assistant_prefix=candidate["assistant_prefix"],
            )
            candidate["runtime_prompt_sha256"] = digest(candidate["runtime_prompt"])
            assert candidate["evidence_brief"] == original["evidence_brief"]
            assert candidate["prompt_evidence"] == original["prompt_evidence"]
            rows.append(candidate)
        selection.append({
            "parent_record_id": original["id"], "question": original["question"],
            "normalized_question": normalized_question(original["question"]),
            "sampling_seed": seed,
            "prior_six_question_development_set": (
                normalized_question(original["question"]) in development_questions
            ),
            "first_variant": variants[0],
        })
    return rows, selection


def main():
    destination = HERE / "fact_coverage_pairs116_v1.matrix.json"
    assert not destination.exists()
    source_path = RUN / "fixed79.current_quote.matrix.json"
    old_six_path = RUN / "adapter_pair_v1.p0RvMj/adapter_pairs12_v1.matrix.json"
    original = load(source_path)
    selected = select_states(original["rows"])
    assert len(original["rows"]) == 79 and len(selected) == 58
    development_questions = {
        normalized_question(row["question"])
        for row in load(old_six_path)["selection"]
    }
    assert len(development_questions) == 6
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    rows, selection = build_pairs(selected, render_qwen_chat_prompt, development_questions)
    assert len(rows) == len({r["id"] for r in rows}) == 116
    assert sum(r["prior_six_question_development_set"] for r in selection) == 6
    current = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    frozen = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    effective = deepcopy(load(frozen)["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == load(current)["generator"]
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "fact_coverage_pairs116_v1", "evaluation_only": True,
        "training_ready": False, "rows": rows, "selection": selection,
        "source_matrix_sha256": sha(source_path), "builder_sha256": sha(Path(__file__)),
        "development_membership_source_sha256": sha(old_six_path),
        "frozen_config_sha256": sha(frozen), "canonical_config_sha256": sha(current),
        "max_context": 12288, "max_new_tokens": 2048, "added_rules": COVERAGE_RULES,
        "predeclared_protocol": [
            "One state per all 58 normalized questions, chosen by ID hash before new generation.",
            "The 52 questions beyond the repeatedly tuned six are broader development checks, not unseen holdout.",
            "Only the fixed generic fact-coverage instruction changes within each pair.",
            "No old answer, judge result, gold, example, source-review correction or hidden evidence enters either prompt.",
            "Original evidence, hypothesis, round, tokenizer, adapter and explicit per-pair sampling seed are identical.",
            "29 original-first and 29 candidate-first pairs; output limit2048/context12288 are shared.",
            "Any preflight overflow blocks launch, rather than truncating evidence or silently dropping rows.",
            "All generation/parse/grounding/audit failures remain in each58 denominator.",
            "Inspect traceable question coverage, factual actor/time/modality/causal support and mechanical validity separately.",
            "No production promotion from acceptance rate, automated coverage or selected source-review examples.",
            "No new retrieval, training, production gate, default-model change, old-label rewrite or online service mutation.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "matrix": str(destination), "questions": len(selection), "generations": len(rows),
        "prior_six": 6, "broader_development": 52, "sha256": sha(destination),
    }))


if __name__ == "__main__":
    main()
