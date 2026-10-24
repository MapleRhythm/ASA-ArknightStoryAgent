"""Remove only generator-visible hypothesis/hints; preserve all visible evidence."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, RUN, SOURCE, ROOT, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

MATRIX = HERE / "without_reasoning_hints58_v1.matrix.json"
PREFLIGHT = HERE / "without_reasoning_hints58_preflight_v1.json"


def remove_reasoning_context(row):
    prefix, body = row["user"].split("\nevidence_brief:\n", 1)
    lines = prefix.splitlines()
    assert len(lines) == 4
    assert lines[0] == "task: conclusion_generation"
    assert lines[1] == "question: " + row["question"]
    assert lines[2].startswith("hypothesis: ")
    assert json.loads(lines[2][len("hypothesis: "):]) == row["hypothesis"]
    assert lines[3] == f'round: {row["current_round"]}/{row["max_retrieval_rounds"]}'
    evidence = row["evidence_brief"]
    assert body.startswith(evidence + "\nminirag_hints: ")
    hints, suffix = body[len(evidence) + 1:].split("\n", 1)
    assert hints.startswith("minirag_hints: ") and suffix.startswith("output_schema: ")
    user = "\n".join([lines[0], lines[1], lines[3], "evidence_brief:", evidence, suffix])
    return user, {"hypothesis_line": lines[2], "hints_line": hints}


def main():
    assert not MATRIX.exists() and not PREFLIGHT.exists()
    original_matrix = HERE / "fact_coverage_pairs116_v1.matrix.json"
    original_rows = [r for r in load(original_matrix)["rows"] if r["renderer"] == "original_prompt_v1"]
    assert len(original_rows) == 58
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    rows, removals = [], []
    for original in original_rows:
        row = deepcopy(original)
        user, removed = remove_reasoning_context(original)
        row.update(
            id=original["parent_record_id"] + "::without_reasoning_hints_v1",
            renderer="without_reasoning_hints_v1", user=user,
            runtime_prompt=render_qwen_chat_prompt(original["system"], user,
                                                   assistant_prefix=original["assistant_prefix"]),
        )
        row["runtime_prompt_sha256"] = digest(row["runtime_prompt"])
        assert row["evidence_brief"] in row["runtime_prompt"]
        assert row["evidence_sha256"] == digest(row["evidence_brief"])
        for key in ("question", "hypothesis", "prompt_evidence", "sampling_seed", "system",
                    "current_round", "max_retrieval_rounds", "assistant_prefix"):
            assert row[key] == original[key]
        rows.append(row)
        removals.append({"id": row["id"], **removed})
    assert len({r["id"] for r in rows}) == 58
    name = "凯尔希·思衡托"
    keeper = next(r for r in original_rows if "保存者" in r["question"])
    altered = next(r for r in rows if "保存者" in r["question"])
    assert name in keeper["hypothesis"]["keywords"]
    assert name not in keeper["question"] + keeper["evidence_brief"]
    assert name in keeper["runtime_prompt"] and name not in altered["runtime_prompt"]
    save(MATRIX, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "version": "without_reasoning_hints58_v1",
        "rows": rows, "removed_metadata_not_sent_to_model": removals,
        "source_matrix_sha256": sha(original_matrix), "builder_sha256": sha(Path(__file__)),
        "predeclared_protocol": [
            "Same58 evidence states/system/schema/model/adapter/tokenizer/output budget/per-row seeds as original4B.",
            "Remove the generator's hypothesis JSON line and synthetic minirag_hints line only; no evidence removed or added.",
            "Both fields are removed together: a reasoning-context ablation, not an isolated single-field causal test.",
            "The downstream validator/follow-up still receives the original hypothesis; this is not a whole-loop cleanup.",
            "No old answer/judge/desired answer/source-review correction is inserted; prior generation inputs remain untouched.",
            "These are development states, not fresh holdout or fresh retrieval; retained old baseline is not interleaved with this run.",
            "All errors/no-answer outcomes stay in58; no rerun or prompt repair based on results.",
            "No default prompt/model change, training, production mutation or automatic promotion.",
        ],
    })
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "cpu_only_builder": True,
        "matrix_sha256": sha(MATRIX), "source_hashes": snapshot(SOURCE),
        "source_matrix_sha256": sha(original_matrix), "builder_sha256": sha(Path(__file__)),
        "observed_unanchored_keyword": {
            "parent_record_id": keeper["parent_record_id"], "term": name,
            "in_question_or_visible_evidence": False, "in_original_hypothesis_and_hints": True,
            "in_candidate_generator_prompt": False,
            "not_a_certified_global_falsehood": True,
        },
        "hint_builder_sha256": sha(SOURCE / "asa_arknight_story_agent/inference/retrieval/minirag_prompt_hints.py"),
        "new_generations": 0, "training_ready": False,
    })
    print(json.dumps({"states": 58, "no_evidence_removed": True,
                      "unanchored_term_removed_from_generator_input": True}), flush=True)


if __name__ == "__main__":
    main()
