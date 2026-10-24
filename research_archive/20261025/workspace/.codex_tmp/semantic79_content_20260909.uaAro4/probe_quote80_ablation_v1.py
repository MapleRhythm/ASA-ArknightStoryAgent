"""Offline counterfactual: remove only the individual 80-char quote ceiling.

All exact-visible-evidence checks, per-fact 160 and answer-total 400 limits,
fact content, parser behavior, original prompts and original outputs remain.
No files in the canonical repository are modified.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
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
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(RUN))
from analyze_canonical_grounding_v4 import matched_rows, round_state
from asa_arknight_story_agent.inference.generation.conclusion_generation import (
    normalize_minimal_conclusion_output, parse_conclusion_output,
)
from asa_arknight_story_agent.inference.grounding import quote_checks
from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument

ONLY_QUOTE80 = re.compile(r"^(?:supported|inferred)_fact_[1-9]\d*_ref_[1-9]\d*_quote_over_80$")
CASES = [
    ("fixed79", RUN / "fixed79.current_quote.matrix.json", RUN / "fixed79_configured_lora_ctx12288/predictions.json", 79),
    ("strict9", RUN / "after.matrix.json", RUN / "strict9_vllm_context_guard_ctx12288_stable_idle/predictions.json", 9),
    ("paired68", RUN / "same_evidence_retry_paired68_v1.matrix.json", RUN / "same_evidence_retry_paired68_ctx12288_v1/predictions.json", 68),
    ("focus18", RUN / "question_focus_paired18_v1.matrix.json", RUN / "question_focus_paired18_ctx12288_v1/predictions.json", 18),
    ("quote_sft9", RUN / "quote80_sft_adapter_probe9_v1.matrix.json", RUN / "quote80_sft_adapter_probe9_ctx12288_v1/predictions.json", 9),
    ("slot8", RUN / "semantic79_content_v2.fPkkhM/slot_feedback_paired8_v1.matrix.json", RUN / "semantic79_content_v2.fPkkhM/slot_feedback_paired8_ctx12288_v1/predictions.json", 8),
    ("anchor8", RUN / "anchor_ablation_v1.VHleHq/anchor_ablation_paired8_v1.matrix.json", RUN / "anchor_ablation_v1.VHleHq/anchor_ablation_paired8_ctx12288_v1/predictions.json", 8),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def remaining_issues(issues):
    return [item for item in issues if not ONLY_QUOTE80.fullmatch(item)]


def main():
    destination = HERE / "quote80_ablation_comparison_v1.json"
    assert not destination.exists()
    source_hashes = {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}
    original = quote_checks.validate_grounded_quotes

    def counterfactual(**kwargs):
        issues, warnings = original(**kwargs)
        return remaining_issues(issues), warnings

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_only": True,
        "training_ready": False, "new_generation": False, "production_changed": False,
        "source_hashes": source_hashes, "driver_sha256": sha(Path(__file__)),
        "removed_issue_pattern": ONLY_QUOTE80.pattern,
        "source_files": {}, "groups": [], "newly_answering_states": [],
        "limitations": [
            "Counterfactual protocol ablation, not a new generation or retrieval run.",
            "Different experiments/adapters and repeated questions are reported separately, not one accuracy denominator.",
            "All non-80 quote checks remain; quote/fact text is never shortened or rewritten.",
            "Mechanical admission is not semantic correctness; newly admitted answers require source review.",
        ],
    }
    for name, matrix_path, predictions_path, expected_count in CASES:
        assert len(load(matrix_path)["rows"]) == expected_count
        report["source_files"][str(matrix_path)] = sha(matrix_path)
        report["source_files"][str(predictions_path)] = sha(predictions_path)
        outputs = {"baseline": [], "quote160": []}
        changes = []
        for row, prediction in matched_rows(load(matrix_path), load(predictions_path)):
            current, maximum = round_state(row)
            item = {
                **{key: row[key] for key in (
                    "id", "source_id", "question", "renderer", "template", "evidence_brief",
                    "evidence_id_to_corpus_id", "packing_trace",
                )},
                "current_round": current, "max_retrieval_rounds": maximum,
                "raw_output": prediction["raw_output"],
                "finish_reason": prediction["finish_reason"],
                "input_tokens": prediction["prompt_token_count"],
                "generated_tokens": prediction["generated_token_count"],
            }
            try:
                normalized = normalize_minimal_conclusion_output(prediction["raw_output"], "minimal")
                hyp = HypothesisDocument(**row["hypothesis"])
                parsed = parse_conclusion_output(
                    normalized, question=row["question"], current_hypothesis=hyp,
                    max_round_reached=current >= maximum,
                )
                item["normalized_output"] = normalized
                item["parsed"] = asdict(parsed)
                baseline_issues, _ = original(
                    conclusion=deepcopy(parsed), evidence=deepcopy(row["prompt_evidence"]),
                    question=row["question"], evidence_prompt_text=row["evidence_brief"],
                )
                item["original_quote_issues"] = baseline_issues
                for variant in outputs:
                    result = deepcopy(item)
                    quote_checks.validate_grounded_quotes = original if variant == "baseline" else counterfactual
                    try:
                        grounded = validate_conclusion_grounding(
                            question=row["question"], hypothesis=hyp,
                            evidence=deepcopy(row["prompt_evidence"]), conclusion=deepcopy(parsed),
                            max_round_reached=current >= maximum, mode="quote",
                            evidence_prompt_text=row["evidence_brief"],
                        )
                        result["status"] = "ok"
                        result["grounded"] = asdict(grounded)
                        assert asdict(parsed) == item["parsed"]
                    except Exception as exc:
                        result.update(status="parse_or_validation_error", error=f"{type(exc).__name__}: {exc}")
                    finally:
                        quote_checks.validate_grounded_quotes = original
                    outputs[variant].append(result)
            except Exception as exc:
                item.update(status="parse_or_validation_error", error=f"{type(exc).__name__}: {exc}")
                for variant in outputs:
                    outputs[variant].append(deepcopy(item))
            baseline, candidate = outputs["baseline"][-1], outputs["quote160"][-1]
            before = baseline.get("grounded", {}).get("next_action", "pipeline_error")
            after = candidate.get("grounded", {}).get("next_action", "pipeline_error")
            if baseline.get("grounded") != candidate.get("grounded"):
                change = {
                    "id": row["id"], "question": row["question"], "before": before, "after": after,
                    "original_quote_issues": item.get("original_quote_issues"),
                    "remaining_quote_issues": remaining_issues(item.get("original_quote_issues", [])),
                }
                changes.append(change)
                if before != "answer_directly" and after == "answer_directly":
                    assert candidate["grounded"]["supported_facts"] == candidate["parsed"]["supported_facts"]
                    assert candidate["grounded"]["inferred_facts"] == candidate["parsed"]["inferred_facts"]
                    assert not change["remaining_quote_issues"]
                    report["newly_answering_states"].append({**change, "group": name})
        summaries = {}
        for variant, results in outputs.items():
            assert len(results) == expected_count
            summaries[variant] = {
                "states": len(results),
                "statuses": dict(Counter(r["status"] for r in results)),
                "actions": dict(Counter(r.get("grounded", {}).get("next_action", "pipeline_error") for r in results)),
            }
            save(HERE / f"{name}.{variant}_v1.json", {
                "evaluation_only": True, "training_ready": False, "new_generation": False,
                "protocol_ablation": variant, "matrix_sha256": sha(matrix_path),
                "predictions_sha256": sha(predictions_path),
                "runtime_source_root": str(SOURCE), "results": results,
                "summary": summaries[variant],
                "limitations": report["limitations"],
            })
        report["groups"].append({"name": name, "summary": summaries, "changes": changes})
        print(json.dumps({"group": name, "summary": summaries, "changed_states": len(changes)}, ensure_ascii=False), flush=True)
    assert quote_checks.validate_grounded_quotes is original
    assert source_hashes == {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*.py")}
    assert all(sha(Path(p)) == digest for p, digest in report["source_files"].items())
    save(destination, report)
    print(json.dumps({"newly_answering_states": report["newly_answering_states"], "production_changed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
