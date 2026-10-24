"""CPU-only subject/reference audit: all 24 v41 and all 32 v35 outputs."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V41 = HERE.parent / "routing_frozen_canonical_reader_20260910.v41"
V35 = HERE.parent / "reader_alias_groups_20260910.v35"
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
sys.path.insert(0, str(V41))
import run_canonical_reader_v2 as prior
shared = prior.shared
from subject_reference_diagnostic import diagnose


def rows():
    pre41 = shared.load(V41 / "preflight.json")
    gen41 = shared.load(V41 / "generations.json")
    index41 = {(r["id"], r["condition"]): r for r in pre41["conditions"]}
    for result in gen41["attempts"]:
        source = index41[(result["id"], result["condition"])]
        yield {
            "source_experiment": "v41", "pair_id": result["pair_id"], "id": result["id"],
            "condition": result["condition"], "question": source["question"],
            "hypothesis": source["hypothesis"], "evidence": source["visible_evidence"],
            "raw": result["sanitized_raw_output"],
            "recorded_action": result["conclusion"]["next_action"],
            "recorded_action_scope": "v41_current_canonical_first_round_verified",
            "max_round_reached": False,
        }
    pre35, gen35 = shared.load(V35 / "preflight.json"), shared.load(V35 / "generations.json")
    for source, result in zip(pre35["rows"], gen35["rows"], strict=True):
        assert source["pair_id"] == result["pair_id"]
        for kind in ("control", "candidate"):
            item = result["conditions"][kind]
            assert item["status"] == "ok"
            yield {
                "source_experiment": "v35", "pair_id": source["pair_id"], "id": source["id"],
                "condition": kind, "question": source["question"], "hypothesis": source["hypothesis"],
                "evidence": source["evidence_brief"], "raw": item["raw_output"],
                "recorded_action": item["grounded"]["next_action"],
                "recorded_action_scope": "historical_v35_standardized_last_round_empty_object_quote_check_not_replayed",
                "max_round_reached": True,
            }


def prepare():
    assert not (HERE / "preflight.json").exists()
    prior.verify(shared.load(V41 / "preflight.json"))
    shared.load(V41 / "checkpoint_v41.json")
    inputs = list(rows())
    assert len(inputs) == 56
    assert Counter(r["source_experiment"] for r in inputs) == {"v41": 24, "v35": 32}
    paths = [
        Path(__file__), HERE / "subject_reference_diagnostic.py", HERE / "test_subject_reference_diagnostic.py",
        V41 / "run_canonical_reader_v2.py", V41 / "preflight.json", V41 / "generations.json",
        V41 / "checkpoint_v41.json", V35 / "preflight.json", V35 / "generations.json",
        V35 / "checkpoint_v35.json", ROOT / "indexes/arknights_story/operator_aliases.json",
        *sorted(SRC.rglob("*.py")),
    ]
    shared.save(HERE / "preflight.json", {
        "version": "named_subject_reference_presence_diagnosis_v42",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False, "gpu_used": False,
        "protocol": [
            "Read-only diagnostic, not a validator or an answer-quality/accuracy metric.",
            "Parse every existing v41 and v35 raw output with current canonical normalizer/parser; no new model/retrieval calls.",
            "Name candidate must be >=2 characters, an exact existing hypothesis entity, an alias-catalog key, and mentioned in the question and fact.",
            "Check explicit alias presence within that fact's cited visible E blocks; unreferenced blocks do not silently become citations.",
            "No output, reference, E-ID, source, alias catalog, parsing rule, or production decision is modified.",
            "No inferred family/role links from the catalog; aliases express only existing spelling equivalence.",
            "All old failures/actions retained and labeled by their original experiment's validation context.",
            "Name absence can be caused by legitimate pronouns; name presence can be an unrelated sentence. Neither is semantic entailment.",
            "v35 eight known development cases broaden observations but are not a held-out validation set.",
            "Do not promote a reject rule without examining false positives and unaffected wrong relationship claims.",
        ],
        "sha256": {str(p): shared.sha(p) for p in paths},
        "inputs": inputs,
    })
    print({"inputs": 56, "gpu_used": False, "production_changed": False})


def run():
    pre = shared.load(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    sys.path.insert(0, str(SRC))
    from asa_arknight_story_agent.inference.generation.conclusion_generation import normalize_minimal_conclusion_output, parse_conclusion_output
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    catalog = shared.load(ROOT / "indexes/arknights_story/operator_aliases.json")
    results = []
    for row in pre["inputs"]:
        result = {key: row[key] for key in (
            "source_experiment", "pair_id", "id", "condition", "recorded_action", "recorded_action_scope",
        )}
        original = deepcopy(row)
        try:
            normalized = normalize_minimal_conclusion_output(row["raw"], "minimal")
            parsed = parse_conclusion_output(
                normalized, question=row["question"], current_hypothesis=HypothesisDocument(**row["hypothesis"]),
                max_round_reached=row["max_round_reached"],
            )
            result.update(status="parsed", parsed=asdict(parsed))
            diagnostics = []
            for group in ("supported_facts", "inferred_facts"):
                for item in diagnose(
                    question=row["question"], hypothesis=row["hypothesis"],
                    facts=getattr(parsed, group), visible_evidence=row["evidence"], catalog=catalog,
                ):
                    diagnostics.append({"group": group, **item})
            result.update(diagnostics=diagnostics, absence_flags=sum(
                r["named_subject_absent_in_available_cited_blocks"] for r in diagnostics
            ))
        except Exception as exc:
            result.update(status="parse_or_diagnostic_error", error=f"{type(exc).__name__}: {exc}")
        assert row == original
        results.append(result)
    assert len(results) == 56
    summary = {}
    for version in ("v41", "v35"):
        subset = [r for r in results if r["source_experiment"] == version]
        summary[version] = {
            "outputs": len(subset), "status": dict(Counter(r["status"] for r in subset)),
            "outputs_with_absence_flags": sum(bool(r.get("absence_flags")) for r in subset),
            "fact_subject_absence_flags": sum(r.get("absence_flags", 0) for r in subset),
        }
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    shared.save(HERE / "diagnosis_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "results": results, "summary_not_accuracy": summary,
        "production_changed": False, "gpu_used": False, "model_calls": 0, "retrieval_calls": 0,
        "training_ready": False, "semantic_review_complete": False,
    })
    print(summary)
    for row in results:
        if row.get("absence_flags"):
            print({key: row[key] for key in ("source_experiment", "pair_id", "condition", "recorded_action", "absence_flags")})


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_subject_diagnosis.py prepare|run")
    operations[sys.argv[1]]()
