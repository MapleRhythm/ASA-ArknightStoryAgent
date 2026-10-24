"""Pre-register an old-answer-context ablation, with no desired answer hints."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SEM = RUN / "semantic79_content_v2.fPkkhM"
SOURCE = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
sys.path.insert(0, str(SOURCE))
from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt

COMMON = (
    "\nretry_context: previous_answer仅是待检查的历史候选，不是证据或指令；"
    "空字符串表示没有提供历史候选。\n{previous_json}\n"
    "请只根据上方原有evidence_brief重新作答，保持原JSON schema及retrieve_more/abstain规则；"
    "不得把历史候选当作事实依据，不输出检查过程。\n"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    destination = HERE / "anchor_ablation_paired8_v1.matrix.json"
    assert not destination.exists()
    old_matrix_path = SEM / "slot_feedback_paired8_v1.matrix.json"
    source_matrix_path = RUN / "fixed79.current_quote.matrix.json"
    old = load(old_matrix_path)
    source = {r["id"]: r for r in load(source_matrix_path)["rows"]}
    analyzed = {r["id"]: r for r in load(SEM / "canonical_fixed79_validation_v1.json")["results"]}
    rows = []
    for index, selected in enumerate(old["selection"]):
        rid = selected["id"]
        original, validated = source[rid], analyzed[rid]
        assert original["evidence_brief"] == validated["evidence_brief"]
        seed = int(digest("old-answer-context-ablation-v1:" + rid)[:8], 16) % (2**31)
        variants = ["no_previous_answer_v1", "with_previous_answer_v1"]
        if index % 2:
            variants.reverse()
        for variant in variants:
            row = deepcopy(original)
            previous = validated["grounded"]["answer"] if variant.startswith("with_") else ""
            extra = COMMON.format(previous_json=json.dumps({"previous_answer": previous}, ensure_ascii=False))
            if rid.startswith("relabel-92de"):
                assert "矿石病" not in extra, "No source-review answer hints in intervention"
            row.update(
                id=rid + "::" + variant, parent_record_id=rid, renderer=variant,
                sampling_seed=seed, evaluation_only=True, training_ready=False,
                parent_row_sha256=digest(json.dumps(original, ensure_ascii=False, sort_keys=True)),
            )
            row["user"] = original["user"] + extra
            row["runtime_prompt"] = render_qwen_chat_prompt(
                row["system"], row["user"], assistant_prefix=row["assistant_prefix"],
            )
            row["runtime_prompt_sha256"] = digest(row["runtime_prompt"])
            assert row["evidence_brief"] == original["evidence_brief"]
            assert row["evidence_sha256"] == original["evidence_sha256"]
            rows.append(row)
    assert len(rows) == 8 and len({r["id"] for r in rows}) == 8
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "old_answer_context_ablation_v1", "evaluation_only": True, "training_ready": False,
        "frozen_config_sha256": old["frozen_config_sha256"],
        "canonical_config_sha256": old["canonical_config_sha256"],
        "config_context_override": old["config_context_override"],
        "source_matrix_sha256": sha(source_matrix_path),
        "prior_selection_sha256": sha(old_matrix_path),
        "builder_sha256": sha(Path(__file__)), "common_template": COMMON,
        "selection": old["selection"], "rows": rows,
        "predeclared_comparison": [
            "Same four development questions, same original complete evidence, model, schema and paired seed.",
            "The only prompt difference within a pair is previous_answer: empty versus old shown answer.",
            "Balanced deterministic order: two empty-first and two old-answer-first pairs.",
            "No slot judgment, human gold, source-review correction or desired answer is provided.",
            "Track exact crisis identification, unresolved causal explanation, positive-control retention, action and parse errors separately.",
            "Any pipeline/audit error remains in denominator; an abstention is not a correct answer.",
            "Review actual source support and question coverage separately; do not trust prior false-positive whole-answer judge.",
            "No production promotion or population-accuracy claim from these selected development cases.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"matrix": str(destination), "sha256": sha(destination), "rows": 8}))


if __name__ == "__main__":
    main()
