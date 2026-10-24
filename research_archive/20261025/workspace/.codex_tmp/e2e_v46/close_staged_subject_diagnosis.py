"""Close unchanged diagnostic replays and one source-bound false-negative witness."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import re
import run_staged_subject_diagnosis as probe
from subject_reference_diagnostic import diagnose, evidence_blocks

HERE = Path(__file__).resolve().parent
shared = probe.shared


def main():
    assert not (HERE / "checkpoint_v46.json").exists()
    pre = shared.load(HERE / "preflight.json")
    result = shared.load(HERE / "diagnosis_v1.json")
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())
    probe.probe.staged.imports()
    from asa_arknight_story_agent.inference.generation.conclusion_generation import normalize_minimal_conclusion_output, parse_conclusion_output
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    catalog = shared.load(probe.probe.reader.ROOT / "indexes/arknights_story/operator_aliases.json")
    replays = 0
    for item, saved in zip(pre["inputs"], result["outputs"], strict=True):
        parsed = parse_conclusion_output(
            normalize_minimal_conclusion_output(item["raw"], "minimal"), question=item["question"],
            current_hypothesis=HypothesisDocument(**item["hypothesis"]), max_round_reached=item["max_round_reached"],
        )
        assert saved["status"] == "parsed" and asdict(parsed) == saved["parsed"]
        diagnostics = []
        for group in ("supported_facts", "inferred_facts"):
            diagnostics.extend({"group": group, **finding} for finding in diagnose(
                question=item["question"], hypothesis=item["hypothesis"], facts=getattr(parsed, group),
                visible_evidence=item["visible_evidence"], catalog=catalog,
            ))
        assert diagnostics == saved["diagnostics"]
        replays += 1
    assert replays == 33
    select = lambda r: (
        r["pair_id"] == "fixed58-017:seed0" and r["condition"] == "candidate"
        and r["request_kind"] == "round2_conclusion"
    )
    source = next(r for r in pre["inputs"] if select(r))
    saved = next(r for r in result["outputs"] if select(r))
    fact = saved["parsed"]["supported_facts"][1]
    wrong_claim = "初雪的父亲为了治病天天打猎，最后在天灾那天在车站去世。"
    assert fact["fact"] == wrong_claim and wrong_claim in saved["final_endpoint_answer"]
    finding = next(r for r in saved["diagnostics"] if r["fact_index"] == 2)
    assert not finding["named_subject_absent_in_available_cited_blocks"]
    assert finding["cited_subject_mentions"] == {"E5": ["恩雅"]}
    block = evidence_blocks(source["visible_evidence"])["E5"]
    quotes = [
        "恩雅紧紧握住茱安娜失控颤抖的双手。",
        "茱安娜：我刚学会打猎的时候，被一头源石感染的野兽咬伤......得了矿石病......",
        "为了给我治病......父亲他天天打完猎，回家还要加紧做那些雕像，拿去城里换点药......",
        "前几天，父亲的朋友告诉我......天灾的那天，父亲他......就在车站......",
    ]
    assert all(q in block for q in quotes)
    # Explicitly post-hoc diagnostic-only projection, never a model prompt or
    # replacement of source data: even the exact single member has both names.
    parts = re.split(r"\[CHAIN_MEMBER_(\d+)\]\s*", block)
    members = dict(zip(parts[1::2], parts[2::2], strict=True))
    local_member = members["6"]
    assert all(q in local_member for q in quotes)
    member_only_diagnostic = diagnose(
        question=source["question"], hypothesis=deepcopy(source["hypothesis"]),
        facts=[deepcopy(fact)], visible_evidence="[E5] " + local_member, catalog=catalog,
    )
    assert len(member_only_diagnostic) == 1
    assert not member_only_diagnostic[0]["named_subject_absent_in_available_cited_blocks"]
    assert member_only_diagnostic[0]["cited_subject_mentions"] == {"E5": ["恩雅"]}
    eligible = sum(bool(r["diagnostics"]) for r in result["outputs"])
    checks = sum(len(r["diagnostics"]) for r in result["outputs"])
    flags = sum(bool(r["absence_flags"]) for r in result["outputs"])
    assert (eligible, checks, flags) == (4, 7, 2)
    basis = [HERE / name for name in (
        "preflight.json", "diagnosis_v1.json", "prepare.log", "run_v1.log",
        "run_staged_subject_diagnosis.py", "close_staged_subject_diagnosis.py",
    )]
    shared.save(HERE / "checkpoint_v46.json", {
        "version": "staged_subject_diagnostic_closed_v46",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_parser_and_function_replays": replays,
        "raw_outputs": 33, "underlying_endpoints": 24,
        "eligible_raw_outputs": eligible, "fact_subject_checks": checks, "flagged_raw_outputs": flags,
        "summary_not_accuracy": result["summary_not_accuracy"],
        "source_bound_assistant_false_negative_observation": {
            "pair_id": source["pair_id"], "condition": source["condition"], "request_kind": source["request_kind"],
            "displayed_claim": wrong_claim, "diagnostic": finding,
            "evidence_id": "E5", "block_sha256": shared.digest(block), "source_excerpts": quotes,
            "observation": "Juanna speaks about her father's hunting/medicine/station death; Enya is a different participant in the same scene.",
            "not_human_gold": True,
        },
        "post_hoc_single_member_diagnostic_only_fixture": {
            "source_member": "E5/CHAIN_MEMBER_6",
            "member_sha256": shared.digest(local_member),
            "diagnostic": member_only_diagnostic,
            "model_input_modified": False,
            "observation": "Splitting only by CHAIN_MEMBER still does not disambiguate speaker from interlocutor.",
        },
        "basis_sha256": {str(p): shared.sha(p) for p in basis},
        "new_llm_calls": 0, "retrieval_calls": 0, "gpu_used": False,
        "production_changed": False, "training_ready": False,
        "promotion_decision": "diagnostic_only_no_name_presence_gate",
        "limitations": [
            "33 diagnostic parser returns are not33 strict-valid JSON model outputs; one actual fallback is malformed and remains rejected.",
            "Four eligible raw outputs and seven checks give narrow coverage, not a semantic pass for the other29 outputs.",
            "Both flags repeat carried first-round Pramanix errors; no new stage output receives an absence flag.",
            "Concrete false-negative witness is post-hoc and assistant-reviewed, not an independent false-negative rate estimate.",
            "Single-member projection is CPU diagnostic only and does not modify evidence, generation, actions, alias rules or production.",
        ],
    })
    print({"replays": replays, "eligible_raw_outputs": eligible, "checks": checks, "flagged_raw_outputs": flags,
           "real_source_false_negative_bound": True, "single_member_still_misses": True}, flush=True)


if __name__ == "__main__":
    main()
