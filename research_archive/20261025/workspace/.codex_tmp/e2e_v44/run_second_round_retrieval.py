"""Staged continuation: replay v41 round1 exactly, retrieve unresolved round2.

All 24 outcomes retained: ten terminal answers unchanged, fourteen retrievals.
No new LLM generation; this is not a newly sampled full end-to-end benchmark.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
V41 = HERE.parent / "routing_frozen_canonical_reader_20260910.v41"
V40 = HERE.parent / "routing_frozen_retrieval_20260910.v40"
sys.path.insert(0, str(V41))
import run_canonical_reader_v2 as reader
shared = reader.shared
retrieval = reader.retrieval


def imports():
    helpers = reader.runtime_imports()
    from asa_arknight_story_agent.inference.pipeline.actions import apply_terminal_conclusion
    from asa_arknight_story_agent.inference.pipeline.query_flow import prepare_next_round
    from asa_arknight_story_agent.inference.pipeline.trace import append_conclusion_to_step
    from asa_arknight_story_agent.inference.pipeline.constants import FOLLOW_UP_HYPOTHESIS_TASK_TYPE
    from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
    return {**helpers, **locals()}


def object_digest(value):
    return shared.digest(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def parent_data():
    pre = shared.load(V41 / "preflight.json")
    gen = shared.load(V41 / "generations.json")
    inputs = {(r["id"], r["condition"]): r for r in pre["conditions"]}
    outputs = {(r["pair_id"], r["condition"]): r for r in gen["attempts"]}
    evidence = reader.sources()
    return pre, inputs, outputs, evidence


def rehydrate(key, helpers, parents):
    """Reproduce all first-round state, checking that generation did not mutate evidence."""
    pre, inputs, outputs, retrieved = parents
    saved = outputs[key]
    item = inputs[(saved["id"], saved["condition"])]
    original = retrieved[(saved["id"], saved["condition"])]
    generator = reader.ReplayGenerator(pre["generator"], item, saved)
    pipeline = reader.pipeline_for(pre, helpers, generator)
    state = helpers["PipelineRunState"](
        evidence=deepcopy(original["evidence"]), retrieval_trace=[deepcopy(original["trace"])],
    )
    assert not state.scope_retention_enabled and not state.web_context_evidence
    assert not original["trace"]["retained_chapter_scope"]
    assert not original["trace"]["retained_storyline_scope"]
    assert original["trace"]["scope_retention_enabled"] is False
    hypothesis = helpers["HypothesisDocument"](**item["hypothesis"])
    conclusion = pipeline.generate_conclusion(
        item["question"], hypothesis, state.evidence, state.retrieval_trace, 1,
    )
    assert generator.calls == 1
    assert asdict(conclusion) == saved["conclusion"]
    assert state.retrieval_trace == saved["retrieval_trace"]
    assert state.evidence == original["evidence"], "No unrecorded first-round evidence mutation allowed"
    helpers["append_conclusion_to_step"](state.retrieval_trace[-1], conclusion)
    terminal = helpers["apply_terminal_conclusion"](
        state=state, conclusion=conclusion, round_index=1, max_retrieval_rounds=2,
    )
    queries = []
    if terminal:
        assert state.final_answer == saved["conclusion"]["answer"]
    else:
        assert conclusion.next_action == "retrieve_more"
        assert conclusion.follow_up_hypothesis is not None, "Do not silently generate a new follow-up planner"
        hypothesis, queries = helpers["prepare_next_round"](
            pipeline=pipeline, question=item["question"], current_hypothesis=hypothesis,
            conclusion=conclusion, state=state, step_record=state.retrieval_trace[-1],
            round_index=1, progress_callback=None,
        )
    assert generator.calls == 1
    return item["question"], state, hypothesis, queries, terminal


def round2_prompt(pre, helpers, state, hypothesis, question):
    pipeline, forbidden = retrieval.make_pipeline(pre, helpers, retriever=None)
    packed = pipeline.prepare_prompt_evidence(question, hypothesis, deepcopy(state.evidence))
    packing = []
    prompt, visible = helpers["build_conclusion_prompt"](
        question, hypothesis, state.evidence, state.retrieval_trace, 2, 2,
        pipeline.prompt_evidence_top_k, prompt_evidence=packed,
        evidence_max_chars_per_doc=pipeline.prompt_evidence_max_chars_per_doc,
        evidence_max_total_chars=pipeline.prompt_conclusion_evidence_max_total_chars,
        prompt_mode=pipeline.conclusion_prompt_mode, grounding_mode=pipeline.answer_grounding_mode,
        evidence_packing_trace=packing,
    )
    assert forbidden.calls == 0
    return {"planned_canonical_round2_prompt": prompt, "visible_evidence": visible,
            "prompt_evidence": packed, "evidence_packing": packing,
            "prompt_sha256": shared.digest(prompt)}


def prepare():
    assert not (HERE / "preflight.json").exists()
    parents, helpers = parent_data(), imports()
    pre41, _, outputs, _ = parents
    reader.verify(pre41)
    v40pre = shared.load(V40 / "preflight.json")
    retrieval.verify(v40pre)
    closed = shared.load(V41 / "checkpoint_v41.json")
    assert closed["canonical_method_replays"] == 24
    pre = {
        key: deepcopy(v40pre[key]) for key in
        ("assets", "query_config", "pipeline_config", "reranker_max_length", "sha256")
    }
    pre.update(
        version="staged_round2_retrieval_continuation_v44",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        production_changed=False, training_ready=False,
        verified_nvidia_mapping=retrieval.numeric_mapping(),
        protocol=[
            "All 24 v41 outcomes remain: ten terminal first-round answers carry unchanged, fourteen unresolved states advance.",
            "Rehydrate fresh PipelineRunState from v40 evidence by actually replaying the saved v41 raw through the canonical first-round method.",
            "Assert replayed conclusion, trace, evidence, and generated-call count exactly match; no lost hidden state assumptions.",
            "Canonical append_conclusion_to_step/apply_terminal_conclusion/prepare_next_round, not hand-written hypotheses or queries.",
            "No new planner call: all unresolved conclusions already contain their canonical follow-up hypothesis.",
            "Fourteen actual canonical round2 retrievals under frozen v40 indexes/embedding/reranker/config, no MiniRAG or web.",
            "Capture actual searches, full state, prepared objects and planned canonical round2 prompt. No LLM generation in v44.",
            "No first-round retry, no selection of only good terminal answers, no subject-reference diagnostic applied as a gate.",
            "This is staged continuation of one development run, not independent or newly sampled fresh E2E.",
            "Do not change budgets, bindings, source labels, defaults, services, training, or production code.",
            "Recheck GPU2 mapping and continuous >=60-second idle window inside the existing lock before retrieval model loading.",
        ],
        records=[],
    )
    for key, saved in outputs.items():
        question, state, hypothesis, queries, terminal = rehydrate(key, helpers, parents)
        pre["records"].append({
            "pair_id": key[0], "condition": key[1], "id": saved["id"], "question": question,
            "terminal_after_round1": terminal, "first_round_action": saved["conclusion"]["next_action"],
            "rehydrated_state_sha256": object_digest(asdict(state)),
            "round2_hypothesis": asdict(hypothesis), "round2_queries": queries,
            "carried_terminal_answer": state.final_answer if terminal else None,
        })
    assert len(pre["records"]) == 24
    assert sum(r["terminal_after_round1"] for r in pre["records"]) == 10
    for path in (
        Path(__file__), V41 / "run_canonical_reader_v2.py", V41 / "preflight.json",
        V41 / "generations.json", V41 / "postflight_v1.json", V41 / "checkpoint_v41.json",
        V40 / "retrieval.json", V40 / "checkpoint_v40.json",
    ):
        pre["sha256"][str(path)] = shared.sha(path)
    shared.save(HERE / "preflight.json", pre)
    print({
        "first_round_state_replays": 24, "terminal_carry": 10, "planned_round2_retrievals": 14,
        "new_llm_calls": 0, "preflight_sha256": shared.sha(HERE / "preflight.json"),
    }, flush=True)


def retrieve():
    pre = shared.load(HERE / "preflight.json")
    reader.verify(pre)
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    parents, helpers = parent_data(), imports()
    import torch
    torch.manual_seed(20260909)
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever
    assets = pre["assets"]
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=Path(assets["embedding"]), reranker_model_path=Path(assets["reranker"]),
        reranker_max_length=pre["reranker_max_length"], documents_path=Path(assets["documents"]),
        faiss_index_path=Path(assets["faiss"]), bm25_tokens_path=Path(assets["bm25"]),
        minirag_index_path=None, device="cuda:0",
    )
    assert retriever.minirag_index is None and retriever.reranker is not None
    pipeline, forbidden = retrieval.make_pipeline(pre, helpers, retriever)
    searches = []
    for method in ("dense_search", "sparse_search"):
        original = getattr(retriever, method)

        def traced(query, *args, _method=method, _original=original, **kwargs):
            receipt = {"method": _method, "query": query, "args": args, "kwargs": kwargs}
            searches.append(receipt)
            hits = _original(query, *args, **kwargs)
            receipt["returned_document_ids"] = [h["document"].get("id") for h in hits]
            return hits

        setattr(retriever, method, traced)
    results = []
    with (HERE / "retrieval.progress.jsonl").open("x", encoding="utf-8") as journal:
        for record in pre["records"]:
            start, before = time.monotonic(), len(searches)
            result = {k: record[k] for k in ("pair_id", "condition", "id", "terminal_after_round1")}
            result["status"] = "error"
            try:
                question, state, hypothesis, queries, terminal = rehydrate(
                    (record["pair_id"], record["condition"]), helpers, parents,
                )
                assert object_digest(asdict(state)) == record["rehydrated_state_sha256"]
                assert asdict(hypothesis) == record["round2_hypothesis"] and queries == record["round2_queries"]
                if terminal:
                    result.update(status="terminal_carried_unchanged", terminal_answer=state.final_answer)
                else:
                    graph, web = pipeline._retrieve_round_evidence(
                        question=question, hypothesis=hypothesis, pending_queries=queries,
                        round_index=2, state=state, progress_callback=None,
                    )
                    assert graph is web is None and retriever.minirag_index is None
                    state.retrieval_trace.append(pipeline._build_step_record(
                        round_index=2, pending_queries=queries,
                        hypothesis_task_type=helpers["FOLLOW_UP_HYPOTHESIS_TASK_TYPE"],
                        hypothesis=hypothesis, state=state,
                        minirag_expansion_record=graph, web_context_record=web,
                    ))
                    result.update(
                        status="round2_retrieval_ok", state_after_retrieval=asdict(state),
                        **round2_prompt(pre, helpers, state, hypothesis, question),
                    )
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result.update(seconds=time.monotonic() - start, actual_search_calls=searches[before:])
            result = retrieval.plain_record(result)
            results.append(result)
            journal.write(json.dumps(result, ensure_ascii=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            print({"completed": len(results), "total": 24, "pair_id": record["pair_id"],
                   "condition": record["condition"], "status": result["status"]}, flush=True)
    assert forbidden.calls == 0
    reader.verify(pre)
    shared.save(HERE / "retrieval.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "records": results, "new_llm_calls": 0, "production_changed": False, "training_ready": False,
    })


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    pre = shared.load(HERE / "preflight.json")
    assert retrieval.numeric_mapping() == pre["verified_nvidia_mapping"]
    reader.verify(pre)
    assert not (HERE / "idle_samples.json").exists()
    shared.HERE, shared.run = HERE, retrieve
    shared.idle_then_run()


def check():
    pre, result = shared.load(HERE / "preflight.json"), shared.load(HERE / "retrieval.json")
    reader.verify(pre)
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(line) for line in (HERE / "retrieval.progress.jsonl").read_text().splitlines()]
    assert journal == result["records"] and len(journal) == 24
    parents, helpers = parent_data(), imports()
    statuses = Counter()
    for record, saved in zip(pre["records"], journal, strict=True):
        assert (record["pair_id"], record["condition"]) == (saved["pair_id"], saved["condition"])
        question, state, hypothesis, queries, terminal = rehydrate(
            (record["pair_id"], record["condition"]), helpers, parents,
        )
        assert object_digest(asdict(state)) == record["rehydrated_state_sha256"]
        statuses[saved["status"]] += 1
        if saved["status"] == "terminal_carried_unchanged":
            assert terminal and saved["terminal_answer"] == record["carried_terminal_answer"] == state.final_answer
            assert saved["actual_search_calls"] == []
        elif saved["status"] == "round2_retrieval_ok":
            assert not terminal
            after = helpers["PipelineRunState"](**saved["state_after_retrieval"])
            assert after.retrieval_trace[0] == state.retrieval_trace[0]
            assert len(after.retrieval_trace) == 2
            assert len(saved["actual_search_calls"]) == 2 * len(queries)
            for index, query in enumerate(queries):
                for offset, method in enumerate(("dense_search", "sparse_search")):
                    call = saved["actual_search_calls"][index * 2 + offset]
                    assert call["method"] == method and call["query"] == query
            preview = round2_prompt(pre, helpers, after, hypothesis, question)
            assert all(saved[k] == v for k, v in preview.items())
        else:
            assert saved["error"]
    assert result["new_llm_calls"] == 0
    assert statuses == {"terminal_carried_unchanged": 10, "round2_retrieval_ok": 14}, statuses
    shared.save(HERE / "postflight_v1.json", {
        "version": "staged_round2_retrieval_postflight_v44",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "retrieval_sha256": shared.sha(HERE / "retrieval.json"),
        "status_counts_not_accuracy": statuses,
        "first_round_state_replays": 24, "new_llm_calls": 0,
        "terminal_answers_unchanged": True, "round2_queries_and_prompts_replayed": True,
        "production_changed": False, "training_ready": False,
        "round2_answers_generated": False,
    })
    print({"statuses": statuses, "first_round_replays": 24, "new_llm_calls": 0}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_second_round_retrieval.py prepare|run|check")
    operations[sys.argv[1]]()
