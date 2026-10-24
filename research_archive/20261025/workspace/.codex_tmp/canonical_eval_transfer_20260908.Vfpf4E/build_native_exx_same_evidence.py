"""Native supported-fact/E-ID protocol on identical frozen visible evidence.

Uses canonical runtime's existing Exx template. This is a development probe,
not permission to switch the default or treat E-ID lexical acceptance as truth.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def transform(row, render_user, render_chat, system):
    result = copy.deepcopy(row)
    result["parent_record_id"] = row["id"]
    result["id"] = row["id"] + "::native_exx_same_evidence_v1"
    result["renderer"] = "native_exx_same_evidence_v1"
    result["grounding_mode"] = "evidence_id"
    result["system"] = system
    result["assistant_prefix"] = ""
    result["user"] = render_user(
        question=row["question"], hypothesis=row["hypothesis"],
        round_value=f'{row.get("current_round", 1)}/{row.get("max_retrieval_rounds", 2)}',
        evidence_text=row["evidence_brief"],
    )
    result["runtime_prompt"] = render_chat(system, result["user"], assistant_prefix="")
    result["runtime_prompt_sha256"] = digest(result["runtime_prompt"])
    result["parent_row_sha256"] = digest(json.dumps(row, ensure_ascii=False, sort_keys=True))
    result["evaluation_only"] = True
    result["training_ready"] = False
    return result


if __name__ == "__main__":
    run = Path(__file__).resolve().parent
    root = run / "quote_integrity_candidate_v1/src"
    sys.path.insert(0, str(root))
    from asa_arknight_story_agent.inference.generation.exx_prompt import EXX_SYSTEM_PROMPT, render_exx_user_prompt
    from asa_arknight_story_agent.inference.generation.chat_prompt_format import render_qwen_chat_prompt
    matrix_path = run / "after.matrix.json"
    source = json.loads(matrix_path.read_text())
    assert source["evaluation_only"] and len(source["rows"]) == 9
    rows = [
        transform(row, render_exx_user_prompt, render_qwen_chat_prompt, EXX_SYSTEM_PROMPT)
        for row in source["rows"]
    ]
    for original, new in zip(source["rows"], rows):
        for key in ("evidence_brief", "evidence_sha256", "prompt_evidence", "evidence_id_to_corpus_id",
                    "question", "hypothesis", "packing_trace"):
            assert new[key] == original[key]
    report = {
        "version": "native_exx_same_evidence_v1", "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canonical_exx_template_sha256": hashlib.sha256(
            (root / "asa_arknight_story_agent/inference/generation/exx_prompt.py").read_bytes()
        ).hexdigest(),
        "rows": rows,
        "limitations": [
            "The same 9 repeatedly reviewed development questions; not a held-out test.",
            "Input evidence and mapping are byte-identical; no retrieval or new source is introduced.",
            "This changes protocol, assistant prefix, validator and native output budget (512 vs quote 2048).",
            "It is an alternative operational recipe, not an isolated formatting-effect estimate.",
            "E-ID checks are lexical and cannot establish semantic entailment.",
            "No default production protocol change is authorized by this experiment's completion.",
        ],
    }
    with (run / "strict9_native_exx_v1.matrix.json").open("x") as output:
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "evidence_and_mapping_unchanged": True, "production_changed": False}))
