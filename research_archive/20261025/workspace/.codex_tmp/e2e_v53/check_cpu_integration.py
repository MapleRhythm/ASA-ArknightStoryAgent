"""Fresh CPU integration checks, explicitly not model/semantic evaluation."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import run_shadow as experiment
import span_protocol as protocol

HERE = Path(__file__).resolve().parent
shared = experiment.shared


def compute():
    pre = shared.load(HERE / "preflight.json")
    experiment.verify(pre)
    helpers = experiment.reader.runtime_imports()
    from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    rows = shared.load(HERE / "inputs.json")
    bindings, fixtures = [], []
    assert experiment.make_tasks(rows) == [
        {k: v for k, v in t.items() if k not in ("prompt_tokens", "token_ids_sha256", "context_fits")}
        for t in pre["tasks"]
    ]
    for row in rows:
        visible = row["canonical_kwargs"]["evidence_prompt_text"]
        packet = protocol.from_dict(row["packet"])
        assert packet == protocol.make_packet(visible)
        mapping = evidence_id_text_map_from_prompt(visible)
        assert mapping == dict(packet.blocks)
        for span in packet.spans:
            ref = protocol.references(packet, [span.span_id])[0]
            assert mapping[ref["evidence_id"]][span.start:span.end] == ref["quote"] == span.text
            bindings.append({"family": row["family"], **asdict(span)})
        # Predetermined source-bound positive fixture: this is NOT a generated
        # answer and not evidence that an LLM would select these references.
        witnesses = row["source_row"]["source_witnesses"]
        ids = []
        for witness in witnesses:
            start = witness["occurrences"][0]["start_codepoint"]
            end = witness["occurrences"][0]["end_codepoint"]
            assert mapping[witness["evidence_id"]][start:end] == witness["excerpt"]
            ids.extend(s.span_id for s in packet.spans if s.evidence_id == witness["evidence_id"]
                       and s.start < end and start < s.end)
        ids = list(dict.fromkeys(ids))
        claim = row["source_row"]["single_fact_projections_not_model_answers"][0]["input"]["supported_facts"][0]["fact"]
        draft = {"packet_id": packet.packet_id, "action": "answer",
                 "facts": [{"claim": claim, "span_ids": ids}], "unresolved": []}
        draft_before = deepcopy(draft)
        result = experiment.canonical_draft(row, draft)
        assert draft == draft_before and not result["quote_issues"]
        controls = []
        for label in ("invalid_evidence_id", "absent_quote"):
            proposed = deepcopy(result["constructed_before_validation"])
            if label == "invalid_evidence_id":
                proposed["supported_facts"][0]["evidence_refs"][0]["evidence_id"] = "E999999"
            else:
                sentinel = "v53_source_absent_integrity_sentinel"
                assert sentinel not in visible
                proposed["supported_facts"][0]["evidence_refs"][0]["quote"] += sentinel
            kwargs = deepcopy(row["canonical_kwargs"])
            kwargs["hypothesis"] = HypothesisDocument(**kwargs["hypothesis"])
            kwargs["conclusion"] = ConclusionResult(**proposed)
            before = deepcopy(kwargs)
            issues, warnings = validate_grounded_quotes(
                conclusion=deepcopy(kwargs["conclusion"]), evidence=deepcopy(kwargs["evidence"]),
                question=kwargs["question"], evidence_prompt_text=visible,
            )
            actual = validate_conclusion_grounding(**kwargs)
            assert issues and actual.next_action != "answer_directly" and claim not in actual.answer
            assert kwargs == before, "old validator input mutated"
            controls.append({"label": label, "input": proposed, "quote_issues": issues,
                             "quote_warnings": warnings, "canonical_result": asdict(actual)})
        # Schema/gate witnesses only. A synthetic verdict cannot label truth.
        synthetic = {"packet_id": packet.packet_id, "claim_id": "CPU_ONLY", "verdict": "supported",
                     "relevance": "answers", "subject": "", "relation": "", "object": "",
                     "span_ids": ids, "reason": "Synthetic protocol fixture, not a model judgment."}
        protocol.validate_verdict(synthetic, packet, "CPU_ONLY")
        assert protocol.semantic_pass(synthetic, ids)
        assert not protocol.semantic_pass({**synthetic, "relevance": "unrelated"}, ids)
        assert not protocol.semantic_pass({**synthetic, "verdict": "insufficient"}, ids)
        foreign = next(s.span_id for s in packet.spans if s.span_id not in ids)
        assert not protocol.semantic_pass({**synthetic, "span_ids": [foreign]}, ids)
        fixtures.append({
            "family": row["family"], "synthetic_only": True,
            "source_consistent_fixture_not_necessarily_answer_to_question": draft,
            "new_protocol_canonical_projection": result, "invalid_reference_controls": controls,
            "synthetic_gate_checks": 4,
        })
    experiment.verify(pre)
    return {
        "source_span_bindings": bindings, "synthetic_protocol_fixtures": fixtures,
        "span_binding_count": len(bindings), "synthetic_canonical_fixtures": len(fixtures),
        "invalid_reference_controls_rejected": sum(len(f["invalid_reference_controls"]) for f in fixtures),
        "synthetic_gate_checks": sum(f["synthetic_gate_checks"] for f in fixtures),
        "frozen_tasks_rebuilt_identically": len(pre["tasks"]), "new_llm_calls": 0,
        "retrieval_calls": 0, "gpu_used": False, "production_changed": False,
        "human_gold": False, "semantic_effectiveness_tested": False,
    }


def main():
    assert not (HERE / "cpu_integration_v1.json").exists()
    result = compute()
    shared.save(HERE / "cpu_integration_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": shared.sha(Path(__file__)),
        "preflight_sha256": shared.sha(HERE / "preflight.json"), **result,
    })
    print({k: v for k, v in result.items() if k not in ("source_span_bindings", "synthetic_protocol_fixtures")}, flush=True)


if __name__ == "__main__":
    main()
