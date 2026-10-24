"""Validate all new source bindings and old-gate rejection, without any LLM."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import run_shadow_v2 as wrapper

experiment, shared, protocol = wrapper.base, wrapper.shared, wrapper.protocol
HERE = Path(__file__).resolve().parent


def compute():
    pre = shared.load(HERE / "preflight.json")
    experiment.verify(pre)
    experiment.reader.runtime_imports()
    from asa_arknight_story_agent.inference.pipeline.types import ConclusionResult, HypothesisDocument
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding
    from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    rows = shared.load(HERE / "inputs.json")
    assert experiment.make_tasks(rows) == [
        {k: v for k, v in t.items() if k not in ("prompt_tokens", "token_ids_sha256", "context_fits")}
        for t in pre["tasks"]
    ]
    bindings, fixtures = [], []
    for row in rows:
        packet = protocol.from_dict(row["packet"])
        visible = row["canonical_kwargs"]["evidence_prompt_text"]
        assert packet == protocol.make_packet(visible)
        mapping = evidence_id_text_map_from_prompt(visible)
        assert mapping == dict(packet.blocks)
        for span in packet.spans:
            ref = protocol.references(packet, [span.span_id])[0]
            assert mapping[ref["evidence_id"]][span.start:span.end] == ref["quote"] == span.text
            assert len(ref["quote"].strip()) <= 80
            bindings.append({"family": row["family"], **asdict(span)})
        # Literal source quotation for protocol testing only, deliberately
        # not labeled an answer to the question or a model-selected citation.
        witness = row["source_row"]["source_witnesses"][0]
        offset = witness["occurrences"][0]["start_codepoint"]
        span = next(s for s in packet.spans if s.evidence_id == witness["evidence_id"] and s.start <= offset < s.end)
        claim = "原文写道：" + span.text.strip()
        draft = {"packet_id": packet.packet_id, "action": "answer",
                 "facts": [{"claim": claim, "span_ids": [span.span_id]}], "unresolved": []}
        result = experiment.canonical_draft(row, draft)
        assert not result["quote_issues"], result["quote_issues"]
        controls = []
        for label in ("invalid_evidence_id", "absent_quote"):
            proposed = deepcopy(result["constructed_before_validation"])
            if label == "invalid_evidence_id":
                proposed["supported_facts"][0]["evidence_refs"][0]["evidence_id"] = "E999999"
            else:
                sentinel = "v53_absent_sentinel"
                assert sentinel not in visible
                proposed["supported_facts"][0]["evidence_refs"][0]["quote"] = sentinel
            kwargs = deepcopy(row["canonical_kwargs"])
            kwargs["hypothesis"] = HypothesisDocument(**kwargs["hypothesis"])
            kwargs["conclusion"] = ConclusionResult(**proposed)
            before = deepcopy(kwargs)
            issues, warnings = validate_grounded_quotes(
                conclusion=deepcopy(kwargs["conclusion"]), evidence=deepcopy(kwargs["evidence"]),
                question=kwargs["question"], evidence_prompt_text=visible)
            actual = validate_conclusion_grounding(**kwargs)
            assert issues and actual.next_action != "answer_directly" and claim not in actual.answer
            assert kwargs == before
            controls.append({"label": label, "input": proposed, "quote_issues": issues,
                             "quote_warnings": warnings, "canonical_result": asdict(actual)})
        synthetic = {"packet_id": packet.packet_id, "claim_id": "CPU_ONLY", "verdict": "supported",
                     "relevance": "answers", "subject": "", "relation": "", "object": "",
                     "span_ids": [span.span_id], "reason": "Synthetic gate test, not a semantic judgment."}
        protocol.validate_verdict(synthetic, packet, "CPU_ONLY")
        assert protocol.semantic_pass(synthetic, [span.span_id])
        assert not protocol.semantic_pass({**synthetic, "relevance": "unrelated"}, [span.span_id])
        assert not protocol.semantic_pass({**synthetic, "verdict": "insufficient"}, [span.span_id])
        foreign = next(s.span_id for s in packet.spans if s.span_id != span.span_id)
        assert not protocol.semantic_pass({**synthetic, "span_ids": [foreign]}, [span.span_id])
        fixtures.append({"family": row["family"], "synthetic_only": True,
                         "new_protocol_canonical_projection": result, "invalid_reference_controls": controls,
                         "synthetic_gate_checks": 4})
    experiment.verify(pre)
    return {
        "source_span_bindings": bindings, "synthetic_protocol_fixtures": fixtures,
        "span_binding_count": len(bindings), "synthetic_canonical_fixtures": 3,
        "invalid_reference_controls_rejected": 6, "synthetic_gate_checks": 12,
        "frozen_tasks_rebuilt_identically": 21, "new_llm_calls": 0,
        "gpu_used": False, "production_changed": False, "human_gold": False,
        "semantic_effectiveness_tested": False,
    }


def main():
    assert not (HERE / "cpu_integration_v1.json").exists()
    result = compute()
    shared.save(HERE / "cpu_integration_v1.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": shared.sha(Path(__file__)),
        "preflight_sha256": shared.sha(HERE / "preflight.json"), **result})
    print({k: v for k, v in result.items() if k not in ("source_span_bindings", "synthetic_protocol_fixtures")}, flush=True)


if __name__ == "__main__":
    main()
