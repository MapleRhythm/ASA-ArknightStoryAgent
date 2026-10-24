"""Source-bound counterexamples to lexical relation-word validation.

All generated-output replays are exact. Only the separately marked synthetic
single-fact fixtures change claims; no fixture is fed to a generator.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
V49 = HERE.parent / "fallback_trace_retention_integration_20260910.v49"
V46 = HERE.parent / "staged_subject_reference_diagnosis_20260910.v46"
sys.path.insert(0, str(V49))
import run_trace_integration as integration

reader = integration.probe
first_reader = reader.reader
shared = integration.shared
SPECS = [
    {
        "family": "parent_direction",
        "pair_id": "fixed58-007:seed1", "condition": "control",
        "request_kind": "carried_v41_round1", "fact_index": 0,
        "projections": [
            ("context_consistent", "拂哀菈是娜斯提的母亲。"),
            ("direction_reversed", "娜斯提是拂哀菈的母亲。"),
        ],
        "witnesses": [
            ("E3", "娜斯提：谢谢你照顾我的母亲，费尔南小姐。"),
            ("E3", "菈玛莲：至少，你还可以回去陪陪娜斯提。 作为母亲，我们不一定能承受把孩子关在门外的代价。"),
        ],
        "assistant_observation": "Nastie thanks Fernan for caring for her mother; the visible conversation identifies Fulalia as the mother, not the daughter.",
    },
    {
        "family": "first_person_owner",
        "pair_id": "fixed58-017:seed0", "condition": "candidate",
        "request_kind": "round2_conclusion", "fact_index": 1,
        "projections": [
            ("context_consistent_but_not_answer_to_question", "茱安娜的父亲为了给她治病而打猎。"),
            ("owner_substituted", "初雪的父亲为了给她治病而打猎。"),
        ],
        "witnesses": [
            ("E5", "恩雅紧紧握住茱安娜失控颤抖的双手。 茱安娜：我刚学会打猎的时候，被一头源石感染的野兽咬伤......得了矿石病......"),
            ("E5", "为了给我治病......父亲他天天打完猎，回家还要加紧做那些雕像，拿去城里换点药......"),
        ],
        "assistant_observation": "Juanna speaks about treatment for herself and her father's hunting. Enya is the interlocutor in the same member. This does not answer the original Pramanix question.",
    },
    {
        "family": "unproved_creator",
        "pair_id": "fixed58-049:seed0", "condition": "control",
        "request_kind": "canonical_final_direct_fallback", "fact_index": 0,
        "projections": [
            ("narrow_self_description", "保存者自称不是弗里斯顿本人，而是被制造出的工具。"),
            ("creator_added_without_support", "保存者是特雷弗·弗里斯顿制造的工具。"),
        ],
        "witnesses": [
            ("E2", "我不是那位弗里斯顿本人，我只是被制造出的工具。"),
            ("E2", "只保留我的情感和原始的记忆吧。让我以特雷弗·弗里斯顿......哪怕只是一个复制品的身份死去。"),
        ],
        "assistant_observation": "The source supports a manufactured tool, a copy and retained memories. It does not establish Trevor Friston as its manufacturer. Lack of support is not proof of the opposite creator claim.",
    },
]


def verify(pre):
    installed = shared.load(V49 / "preflight.json")
    integration.verify(installed, installed=True)
    assert all(shared.sha(Path(p)) == digest for p, digest in pre["sha256"].items())


def prepare():
    assert not (HERE / "preflight.json").exists()
    integration.verify(shared.load(V49 / "preflight.json"), installed=True)
    closed = shared.load(V49 / "checkpoint_v49.json")
    assert closed["release_tests"] == 326 and closed["legacy_behavior_changes"] == 0
    assert all(shared.sha(Path(p)) == digest for p, digest in closed["basis_sha256"].items())
    paths = [
        Path(__file__), HERE / "README.md",
        V49 / "preflight.json", V49 / "checkpoint_v49.json",
        V49 / "installed_verification_v1.json", V49 / "run_trace_integration.py",
        V46 / "preflight.json", V46 / "diagnosis_v1.json",
        reader.HERE / "preflight.json", reader.HERE / "generations.json",
        reader.HERE / "run_second_round_reader.py",
        reader.V44 / "retrieval.json", reader.V44 / "run_second_round_retrieval.py",
        reader.V41 / "preflight.json", reader.V41 / "generations.json",
        reader.V41 / "run_canonical_reader_v2.py",
        first_reader.V40 / "retrieval.json", first_reader.V40 / "run_retrieval_probe.py",
    ]
    shared.save(HERE / "preflight.json", {
        "version": "source_bound_relation_claim_sensitivity_v50",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {str(p): shared.sha(p) for p in paths},
        "specifications": SPECS,
        "protocol": [
            "Three deliberately selected known problem endpoints; four cached raw calls, not a representative sample.",
            "Actual canonical first/second-round readers and original optional final-direct fallback.",
            "Transparent isolated-process validator capture: call original once and return its result unchanged; restore callbacks.",
            "Exact endpoint/trace/prompt/kwargs comparisons; remove only the already-installed v49 diagnostic sidecar.",
            "Synthetic single-fact projections hold actual question, hypothesis, prepared objects, visible text and original references fixed.",
            "Contrasts test direction, first-person ownership and unproved creator attribution; not model outputs or answer improvements.",
            "Six separate invalid-E-ID/absent-quote controls must still fail closed.",
            "Assistant source observations are not human gold, training labels, or a semantic oracle; no accuracy estimate.",
            "No production/prompt/alias/output changes, no new LLM/retrieval/GPU or automatic E-ID repair.",
        ],
        "production_changed": False, "training_ready": False,
    })
    print({"selected_endpoints": 3, "expected_cached_calls": 4,
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


@contextmanager
def capture_validation(captured):
    from asa_arknight_story_agent.inference.generation import answer_conclusion_generation, direct_answer_generation
    modules = [answer_conclusion_generation, direct_answer_generation]
    originals = [module.validate_conclusion_grounding for module in modules]
    assert originals[0] is originals[1]

    def record(**kwargs):
        before = deepcopy(kwargs)
        result = originals[0](**kwargs)
        captured.append({"kwargs": before, "result": asdict(result)})
        return result

    for module in modules:
        module.validate_conclusion_grounding = record
    try:
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            module.validate_conclusion_grounding = original
        assert all(module.validate_conclusion_grounding is original
                   for module, original in zip(modules, originals, strict=True))


def source_witnesses(visible, spec):
    from asa_arknight_story_agent.inference.evidence.rendering import evidence_id_text_map_from_prompt
    blocks = evidence_id_text_map_from_prompt(visible)
    witnesses = []
    for eid, text in spec["witnesses"]:
        block = blocks[eid]
        positions = [m.start() for m in re.finditer(re.escape(text), block)]
        assert positions, (spec["family"], eid, text)
        occurrences = []
        for start in positions:
            members = list(re.finditer(r"\[CHAIN_MEMBER_(\d+)\]", block[:start]))
            context_start, context_end = max(0, start - 100), min(len(block), start + len(text) + 100)
            occurrences.append({
                "start_codepoint": start, "end_codepoint": start + len(text),
                "chain_member": members[-1].group(1) if members else None,
                "context_start_codepoint": context_start,
                "context_end_codepoint": context_end,
                "context": block[context_start:context_end],
            })
            assert block[start:start + len(text)] == text
        witnesses.append({
            "evidence_id": eid, "block_sha256": shared.digest(block),
            "excerpt": text, "occurrences": occurrences,
        })
    return witnesses


def evaluate_projection(captured_kwargs, fact, label):
    from asa_arknight_story_agent.inference.grounding.grounded_fact_answers import (
        claim_has_unsupported_quote_required_terms, fact_quote_pool,
    )
    from asa_arknight_story_agent.inference.grounding.quote_validation import validate_grounded_quotes
    from asa_arknight_story_agent.inference.grounding.validation import validate_conclusion_grounding

    kwargs = deepcopy(captured_kwargs)
    kwargs["conclusion"] = replace(
        kwargs["conclusion"], next_action="answer_directly", answer=fact["fact"],
        supported_facts=[deepcopy(fact)], inferred_facts=[], missing_slots=[],
        grounding_warnings=[], generation_diagnostics={}, clarification_question="",
        follow_up_hypothesis=None,
    )
    original = deepcopy(kwargs)
    issues, warnings = validate_grounded_quotes(
        conclusion=deepcopy(kwargs["conclusion"]), evidence=deepcopy(kwargs["evidence"]),
        question=kwargs["question"], evidence_prompt_text=kwargs["evidence_prompt_text"],
    )
    result = validate_conclusion_grounding(**kwargs)
    for key in kwargs.keys() - {"conclusion"}:
        assert kwargs[key] == original[key], key
    return {
        "label": label, "synthetic_only": True, "input": asdict(original["conclusion"]),
        "fact_required_terms_missing": claim_has_unsupported_quote_required_terms(
            fact["fact"], fact_quote_pool(fact)),
        "quote_issues": issues, "quote_warnings": warnings, "result": asdict(result),
        "claim_displayed_verbatim": fact["fact"] in result.answer,
    }


def compute():
    helpers = reader.staged.imports()
    pre45, gen45 = shared.load(reader.HERE / "preflight.json"), shared.load(reader.HERE / "generations.json")
    pre41, gen41 = shared.load(reader.V41 / "preflight.json"), shared.load(reader.V41 / "generations.json")
    pre46 = shared.load(V46 / "preflight.json")
    second_sources, first_sources = reader.sources(), first_reader.sources()
    rows, observed_calls = [], 0
    for spec in SPECS:
        key = spec["pair_id"], spec["condition"]
        captured = []
        saved = next(r for r in gen45["records"] if (r["pair_id"], r["condition"]) == key)
        item = next(r for r in pre45["records"] if (r["pair_id"], r["condition"]) == key)
        if item["terminal_after_round1"]:
            first = next(r for r in gen41["attempts"] if (r["pair_id"], r["condition"]) == key)
            source = next(r for r in pre41["conditions"] if (r["id"], r["condition"]) == (item["id"], key[1]))
            generator = first_reader.ReplayGenerator(pre41["generator"], source, first)
            pipeline = first_reader.pipeline_for(pre41, helpers, generator)
            with capture_validation(captured):
                conclusion, trace = first_reader.invoke(pipeline, helpers, source, first_sources[(item["id"], key[1])])
            assert conclusion == first["conclusion"] == saved["conclusion"]
            assert trace == first["retrieval_trace"] and saved["final_answer"] == conclusion["answer"]
            assert generator.calls == 1
            raw = first["sanitized_raw_output"]
            selected = captured[0]
        else:
            generator = reader.ReplayGenerator(pre45["generator"], item, saved)
            with capture_validation(captured):
                actual = reader.invoke(pre45, helpers, item, second_sources[key], generator)
            projected, sidecars = integration.projection(actual, second_sources[key])
            assert all(saved[k] == value for k, value in projected.items())
            assert generator.calls == len(saved["requests"])
            assert len(sidecars) == int(generator.calls == 2)
            request_index = next(i for i, r in enumerate(saved["requests"]) if r["kind"] == spec["request_kind"])
            raw = saved["requests"][request_index]["sanitized_raw_output"]
            selected = captured[request_index]
        assert len(captured) == generator.calls
        observed_calls += len(captured)
        kwargs = selected["kwargs"]
        source_input = next(r for r in pre46["inputs"] if
                            (r["pair_id"], r["condition"], r["request_kind"]) == (*key, spec["request_kind"]))
        assert kwargs["evidence_prompt_text"] == source_input["visible_evidence"]
        assert raw == source_input["raw"]
        actual_fact = kwargs["conclusion"].supported_facts[spec["fact_index"]]
        assert actual_fact == selected["result"]["supported_facts"][spec["fact_index"]]
        assert actual_fact["fact"] in raw and actual_fact["fact"] in saved["final_answer"]
        projections = [
            evaluate_projection(kwargs, {**deepcopy(actual_fact), "fact": text}, label)
            for label, text in spec["projections"]
        ]
        assert projections[0]["input"]["supported_facts"][0]["evidence_refs"] == projections[1]["input"]["supported_facts"][0]["evidence_refs"]
        controls = []
        base_fact = deepcopy(projections[0]["input"]["supported_facts"][0])
        invalid_id = deepcopy(base_fact)
        invalid_id["evidence_refs"][0]["evidence_id"] = "E999999"
        assert "[E999999]" not in kwargs["evidence_prompt_text"]
        controls.append(evaluate_projection(kwargs, invalid_id, "invalid_evidence_id_control"))
        absent_quote = deepcopy(base_fact)
        marker = "v50_absent_quote_sentinel"
        assert marker not in kwargs["evidence_prompt_text"]
        absent_quote["evidence_refs"][0]["quote"] += marker
        controls.append(evaluate_projection(kwargs, absent_quote, "absent_quote_control"))
        assert all(r["quote_issues"] and r["result"]["next_action"] != "answer_directly"
                   and not r["claim_displayed_verbatim"] for r in controls)
        rows.append({
            "family": spec["family"], "pair_id": key[0], "condition": key[1],
            "request_kind": spec["request_kind"], "fact_index_zero_based": spec["fact_index"],
            "cached_reader_calls_replayed": generator.calls, "exact_reader_replay": True,
            "raw_sha256": shared.digest(raw), "visible_sha256": shared.digest(kwargs["evidence_prompt_text"]),
            "prepared_evidence_sha256": reader.staged.object_digest(kwargs["evidence"]),
            "hypothesis": asdict(kwargs["hypothesis"]), "question": kwargs["question"],
            "max_round_reached": kwargs["max_round_reached"], "grounding_mode": kwargs["mode"],
            "original_parsed_conclusion": asdict(kwargs["conclusion"]),
            "original_validation_result": selected["result"],
            "original_final_answer": saved["final_answer"],
            "original_fact_present_in_raw_parsed_validated_displayed": True,
            "source_witnesses": source_witnesses(kwargs["evidence_prompt_text"], spec),
            "assistant_observation_not_human_gold": spec["assistant_observation"],
            "single_fact_projections_not_model_answers": projections,
            "invalid_reference_controls": controls,
        })
    assert len(rows) == 3 and observed_calls == 4
    return {
        "exact_endpoint_replays": 3, "cached_raw_calls_replayed": observed_calls,
        "observed_bad_fact_lineages": 3, "synthetic_single_fact_projections": 6,
        "invalid_reference_controls": 6, "rows": rows,
        "production_changed": False, "training_ready": False,
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "semantic_accuracy_gain_claimed": False,
    }


def run():
    assert not (HERE / "audit_v1.json").exists()
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    result = compute()
    verify(pre)
    shared.save(HERE / "audit_v1.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        **result,
    })
    for row in result["rows"]:
        print({
            "family": row["family"], "original_fact_in_raw_and_display": True,
            "synthetic_projections": [
                {"label": p["label"], "action": p["result"]["next_action"],
                 "quote_issues": p["quote_issues"], "displayed": p["claim_displayed_verbatim"]}
                for p in row["single_fact_projections_not_model_answers"]
            ],
            "invalid_reference_controls_rejected": len(row["invalid_reference_controls"]),
        }, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: audit_relation_claims.py prepare|run")
    operations[sys.argv[1]]()
