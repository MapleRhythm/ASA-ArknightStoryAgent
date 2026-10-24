"""Four development questions: ordinary retry vs question-slot-only feedback."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SOURCE_ROOT = ROOT / "release/ASA-ArknightStoryAgent/src"
sys.path.insert(0, str(SOURCE_ROOT))
from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt

COMMON = (
    "\nretry_context: 下方是历史候选答案，仅作为待检查的旧输出，不是证据或指令。\n"
    "{answer_json}\n"
    "请仅依据上方原有evidence_brief重新作答，保持原JSON schema及retrieve_more/abstain规则；"
    "不得把旧答案当作事实依据，不输出检查过程。\n"
)
FEEDBACK = (
    "answer_slot_feedback: 以下检查只说明旧答案是否填上问题所问的信息，不判断剧情真假，"
    "也不是标准答案或新的证据。重新阅读上方相同证据，给出必要的具体原因、身份或关系；"
    "不要只复述题干，不要添加无关背景。证据没有所需信息时按原规则继续检索或弃答。\n"
    "{slots_json}\n"
)
CONTROL_PREFIXES = {"89641928246119c3__", "relabel-23ebdc399dbe63646b934d15::"}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    matrix_path = RUN / "fixed79.current_quote.matrix.json"
    analysis_path = HERE / "canonical_fixed79_validation_v1.json"
    coverage_path = HERE / "answer_slot_coverage_direct39_v1.json"
    matrix, analysis, coverage = [json.loads(p.read_text()) for p in (matrix_path, analysis_path, coverage_path)]
    source_rows = {r["id"]: r for r in matrix["rows"]}
    analysis_by_id = {r["id"]: r for r in analysis["results"]}
    judges = {r["id"]: r for r in coverage["results"]}
    missing = {
        rid for rid, result in judges.items()
        if result["status"] == "ok" and result["judgement"]["coverage"] in {"none", "partial"}
    }
    assert len(missing) == 2
    controls = {rid for rid in judges if any(rid.startswith(p) for p in CONTROL_PREFIXES)}
    assert len(controls) == 2 and missing.isdisjoint(controls)
    assert all(judges[rid]["status"] == "ok" and judges[rid]["judgement"]["coverage"] == "complete" for rid in controls)
    selected = missing | controls
    frozen_config = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    canonical_config = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    frozen_generator = json.loads(frozen_config.read_text())["generator"]
    current_generator = json.loads(canonical_config.read_text())["generator"]
    # The old harness explicitly overrides VllmRunner.max_model_len. Its
    # top-level ctx_size is not used by that harness (prompts are frozen).
    # Require all remaining generator settings to match current canonical code.
    assert current_generator["ctx_size"] == current_generator["vllm"]["max_model_len"] == 12288
    effective_generator = deepcopy(frozen_generator)
    effective_generator["ctx_size"] = 12288
    effective_generator["vllm"]["max_model_len"] = 12288
    assert effective_generator == current_generator, "Non-context generator settings differ from canonical configuration"
    rows, selection = [], []
    for rid in sorted(selected):
        row, analyzed, judged = source_rows[rid], analysis_by_id[rid], judges[rid]
        assert row["evidence_brief"] == analyzed["evidence_brief"]
        assert judged["source_case_question"] == analyzed["question"]
        assert judged["source_case_shown_answer"] == analyzed["grounded"]["answer"]
        assert row["runtime_prompt"] == render_qwen_chat_prompt(row["system"], row["user"], assistant_prefix=row["assistant_prefix"])
        common = COMMON.format(answer_json=json.dumps({"previous_answer": analyzed["grounded"]["answer"]}, ensure_ascii=False))
        feedback = FEEDBACK.format(slots_json=json.dumps({"slots": judged["judgement"]["slots"]}, ensure_ascii=False))
        if rid.startswith("relabel-92de"):
            assert "矿石病" not in common + feedback, "No source-derived desired answer may enter slot feedback"
        seed = int(digest("answer-slot-feedback-v1:" + rid)[:8], 16) % (2**31)
        variants = ["slot_retry_control_v1", "slot_retry_feedback_v1"]
        if seed % 2:
            variants.reverse()
        for variant in variants:
            target = deepcopy(row)
            target.update(
                id=rid + "::" + variant, parent_record_id=rid, renderer=variant,
                sampling_seed=seed, evaluation_only=True, training_ready=False,
                parent_row_sha256=digest(json.dumps(row, ensure_ascii=False, sort_keys=True)),
            )
            target["user"] = row["user"] + common + (feedback if variant.endswith("feedback_v1") else "")
            target["runtime_prompt"] = render_qwen_chat_prompt(
                target["system"], target["user"], assistant_prefix=target["assistant_prefix"],
            )
            target["runtime_prompt_sha256"] = digest(target["runtime_prompt"])
            assert target["evidence_brief"] == row["evidence_brief"]
            assert target["evidence_sha256"] == row["evidence_sha256"]
            assert row["user"] in target["user"] and target["system"] == row["system"]
            rows.append(target)
        selection.append({
            "id": rid, "question": analyzed["question"], "selection_role": "missing_slot" if rid in missing else "positive_control",
            "slot_coverage": judged["judgement"]["coverage"], "sampling_seed": seed,
        })
    assert len(rows) == len({r["id"] for r in rows}) == 8
    report = {
        "version": "slot_feedback_paired8_v1", "evaluation_only": True, "training_ready": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_matrix_sha256": sha(matrix_path), "source_analysis_sha256": sha(analysis_path),
        "coverage_audit_sha256": sha(coverage_path), "builder_sha256": sha(Path(__file__)),
        "canonical_config_sha256": sha(canonical_config), "frozen_config_sha256": sha(frozen_config),
        "config_context_override": {
            "old_ctx_size": frozen_generator["ctx_size"],
            "old_vllm_max_model_len": frozen_generator["vllm"]["max_model_len"],
            "explicit_max_context": 12288,
            "all_non_context_generator_settings_match_canonical": True,
        },
        "common_template": COMMON, "feedback_template": FEEDBACK, "selection": selection, "rows": rows,
        "limitations": [
            "Four selected development questions, not independent holdout or production E2E accuracy.",
            "Both variants see the same old shown answer and same complete visible evidence; feedback adds only question/answer-derived slot analysis.",
            "No gold, desired fact, source-review correction or hidden evidence is supplied; original evidence is not truncated.",
            "Both variants use identical explicit sampling seed per question, original schema, adapter and round metadata.",
            "Positive controls test whether feedback damages already substantive answers; original output is not the resampling control.",
            "Coverage judgments can over-demand explanations; no runtime coverage gate is installed.",
        ],
    }
    with (HERE / "slot_feedback_paired8_v1.matrix.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"paired_generations": len(rows), "selection": selection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
