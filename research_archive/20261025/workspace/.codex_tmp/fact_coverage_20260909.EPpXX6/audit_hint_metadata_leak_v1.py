"""Trace internal chain metadata into generated hints, without runtime edits."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from build_fact_coverage_pairs_v1 import HERE, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import save, snapshot

MARKERS = re.compile(r"\b(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES|CHAIN_MEMBER_[0-9]+|model_candidate)\b")


def main():
    output = HERE / "hint_metadata_leak_audit_v1.json"
    assert not output.exists()
    sys.path.insert(0, str(SOURCE))
    from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta
    from asa_arknight_story_agent.inference.retrieval.minirag_prompt_hints import render_minirag_hints_for_prompt
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    paths = [
        HERE / "fact_coverage_pairs116_v1.matrix.json",
        HERE / "susie_handoff4_v1.matrix.json",
        SOURCE / "asa_arknight_story_agent/inference/retrieval/minirag_prompt_hints.py",
        SOURCE / "asa_arknight_story_agent/inference/common/patterns.py",
        SOURCE / "asa_arknight_story_agent/inference/anchors/terms.py",
        Path(__file__),
    ]
    frozen, source = {str(p): sha(p) for p in paths}, snapshot(SOURCE)
    assert source == load(HERE / "checkpoint_v21_state.json")["runtime_source_sha256"]
    rows = [r for r in load(paths[0])["rows"] if r["renderer"] == "original_prompt_v1"]
    rows += load(paths[1])["rows"]
    assert len(rows) == 62
    results = []
    for row in rows:
        hypothesis = HypothesisDocument(**row["hypothesis"])
        visible = [row["prompt_evidence"][e["input_index"]]
                   for e in row["packing_trace"] if e["status"] == "included"]
        old_hints = render_minirag_hints_for_prompt(visible, hypothesis)
        marker = "\nminirag_hints: "
        assert row["user"].count(marker) == 1
        recorded_hints = row["user"].split(marker, 1)[1].split("\n", 1)[0]
        # Record mismatches explicitly; never label a diagnostic as faithful if
        # the saved prompt came from a different renderer/input state.
        new_visible = deepcopy(visible)
        cleaned_chains = []
        for index, item in enumerate(new_visible):
            chain = str(item.get("evidence_chain_text") or "")
            cleaned = re.sub(r"\[CHAIN_MEMBER_[0-9]+\]\s*", "", strip_internal_evidence_meta(chain))
            if cleaned != chain:
                cleaned_chains.append({
                    "index": index, "document_id": item["document"]["id"],
                    "before_sha256": digest(chain), "after_sha256": digest(cleaned),
                    "removed_markers": dict(Counter(MARKERS.findall(chain))),
                })
                item["evidence_chain_text"] = cleaned
        new_hints = render_minirag_hints_for_prompt(new_visible, hypothesis)
        # The candidate is hint-extractor input only. Source and visible
        # evidence sent to generation are not rewritten by this audit.
        results.append({
            "id": row["id"], "question": row["question"], "renderer": row["renderer"],
            "canonical_hint_reproduces_saved": old_hints == recorded_hints,
            "recorded_hints": recorded_hints, "canonical_hints": old_hints, "sanitized_hint_candidate": new_hints,
            "recorded_internal_markers": dict(Counter(MARKERS.findall(recorded_hints))),
            "canonical_internal_markers": dict(Counter(MARKERS.findall(old_hints))),
            "candidate_internal_markers": dict(Counter(MARKERS.findall(new_hints))),
            "hint_changed": old_hints != new_hints, "cleaned_chain_inputs": cleaned_chains,
            "original_evidence_brief_sha256": digest(row["evidence_brief"]),
            "hypothesis_sha256": digest(json.dumps(row["hypothesis"], ensure_ascii=False, sort_keys=True)),
        })
    assert snapshot(SOURCE) == source and all(sha(Path(p)) == h for p, h in frozen.items())
    summary = {
        "states": len(results), "historical_question_states": 58, "single_question_handoff_states": 4,
        "saved_hints_reproduced": sum(r["canonical_hint_reproduces_saved"] for r in results),
        "canonical_hints_with_internal_markers": sum(bool(r["canonical_internal_markers"]) for r in results),
        "sanitized_hints_with_internal_markers": sum(bool(r["candidate_internal_markers"]) for r in results),
        "changed_hint_states": sum(r["hint_changed"] for r in results),
    }
    save(output, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "source_artifacts_sha256": frozen,
        "runtime_source_sha256": source, "summary": summary, "results": results,
        "runtime_edited": False, "new_generation_calls": 0, "new_answer_accuracy": None,
        "source_findings": [
            "render_minirag_hints_for_prompt passes raw evidence_chain_text into anchor extraction.",
            "It synthesizes up to six adjacent-name '相关' relations from its entity list, not a retrieved graph edge list.",
            "The candidate strips only recognized bracketed chain metadata before hint extraction; evidence/gold/model inputs are not rewritten here.",
            "No claim that metadata filtering fixes semantic entity recognition, false relations or answer correctness.",
        ],
    })
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
