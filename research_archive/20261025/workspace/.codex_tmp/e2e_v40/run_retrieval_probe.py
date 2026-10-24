"""Paired first-round retrieval only: four field-recovery cases + two controls."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
RELEASE = ROOT / "release/ASA-ArknightStoryAgent"
SRC = RELEASE / "src"
V27 = HERE.parent / "reader_evidence_router_20260909.v27"
V39 = HERE.parent / "planner_intent_field_fallback_20260910.v39"
IDS = ["fixed58-007", "fixed58-010", "fixed58-017", "fixed58-023", "fixed58-046", "fixed58-049"]
sys.path.insert(0, str(V27))
import run_reader_router as shared
sys.path.insert(0, str(SRC))


class NoGeneration:
    calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("LLM generation is forbidden in this retrieval-only probe")


def imports():
    from asa_arknight_story_agent.config import QueryConfig, DOCUMENTS_PATH, FAISS_INDEX_PATH, BM25_TOKENS_PATH, EMBEDDING_MODEL_DIR
    from asa_arknight_story_agent.inference.cpu_pipeline import CPUInferencePipeline
    from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
    from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
    from asa_arknight_story_agent.inference.pipeline.constants import INITIAL_HYPOTHESIS_TASK_TYPE
    from asa_arknight_story_agent.inference.pipeline.query_flow import build_initial_queries
    from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import build_minimal_conclusion_prompt
    return locals()


def numeric_mapping():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader",
    ], text=True)
    rows = [[part.strip() for part in line.split(",")] for line in raw.splitlines()]
    target = next(row for row in rows if row[0] == "2")
    assert target[1] == shared.GPU_UUID and sorted(rows, key=lambda row: row[2])[2] == target
    return rows


def make_pipeline(pre, helpers, retriever):
    generator = NoGeneration()
    pipeline = helpers["CPUInferencePipeline"](
        retriever=retriever, generator=generator,
        query_config=helpers["QueryConfig"](**pre["query_config"]),
        **pre["pipeline_config"],
    )
    assert not pipeline.web_context_config.enabled
    return pipeline, generator


def preview(helpers, pipeline, question, hypothesis, evidence):
    packed = pipeline.prepare_prompt_evidence(question, hypothesis, deepcopy(evidence))
    prompt, visible = helpers["build_minimal_conclusion_prompt"](
        question=question, current_hypothesis=hypothesis, evidence=evidence,
        prompt_evidence=packed, current_round=1, max_retrieval_rounds=2,
        prompt_evidence_top_k=pipeline.prompt_evidence_top_k, grounding_mode="quote",
    )
    return {"prompt_evidence": packed, "preview_prompt": prompt, "evidence_brief": visible,
            "preview_prompt_sha256": shared.digest(prompt)}


def prepare():
    assert not (HERE / "preflight.json").exists()
    helpers = imports()
    cfg = shared.load(V27 / "runtime.json")
    retrieval = cfg["retrieval"]
    assert retrieval["enable_minirag"] is False and retrieval["enable_reranker"] is True
    assert cfg["inference"]["web_context"]["enabled"] is False
    query_fields = {field.name for field in fields(helpers["QueryConfig"])}
    query_config = asdict(helpers["QueryConfig"](**{k: v for k, v in retrieval.items() if k in query_fields}))
    assert query_config["minirag_top_k"] == 0 and query_config["minirag_weight"] == 0
    pipeline_config = {k: deepcopy(v) for k, v in cfg["inference"].items() if k != "web_context"}
    pipeline_config["web_context_config"] = deepcopy(cfg["inference"]["web_context"])
    assets = {
        "documents": str(helpers["DOCUMENTS_PATH"].resolve()),
        "faiss": str(helpers["FAISS_INDEX_PATH"].resolve()),
        "bm25": str(helpers["BM25_TOKENS_PATH"].resolve()),
        "embedding": str(helpers["EMBEDDING_MODEL_DIR"].resolve()),
        "reranker": str((RELEASE / retrieval["reranker_model_path"]).resolve()),
    }
    assert all(Path(path).exists() for path in assets.values())
    for key in ("embedding", "reranker"):
        assert (Path(assets[key]) / "config.json").exists()
    replay = shared.load(V39 / "replay_v1.json")
    checkpoint = shared.load(V39 / "checkpoint_v39.json")
    assert checkpoint["valid_payloads_identical"] == 54
    indexed = {row["id"]: row for row in replay["rows"]}
    rows = []
    for index, case_id in enumerate(IDS):
        source = indexed[case_id]
        control, candidate = source["control"], source["routing_frozen_candidate"]
        for key in control.keys() - {"entities", "keywords"}:
            assert control[key] == candidate[key]
        assert set(control["entities"]) <= set(candidate["entities"])
        assert set(control["keywords"]) <= set(candidate["keywords"])
        row = {
            "id": case_id, "question": source["question"],
            "control": deepcopy(control), "candidate": deepcopy(candidate),
            "order": ["control", "candidate"] if index % 2 == 0 else ["candidate", "control"],
            "no_intervention_control": control == candidate, "queries": {},
        }
        for kind in ("control", "candidate"):
            hypothesis = helpers["HypothesisDocument"](**row[kind])
            queries = helpers["build_initial_queries"](row["question"], hypothesis)
            assert queries and len(queries) <= 50 and all(isinstance(q, str) and q for q in queries)
            row["queries"][kind] = queries
        rows.append(row)
    assert [r["id"] for r in rows if r["no_intervention_control"]] == ["fixed58-017", "fixed58-023"]
    files = [
        *sorted(SRC.rglob("*.py")), Path(__file__), V27 / "runtime.json",
        V39 / "replay_v1.json", V39 / "checkpoint_v39.json",
        ROOT / "indexes/arknights_story/operator_aliases.json",
        *[Path(assets[key]) for key in ("documents", "faiss", "bm25")],
        V27 / "run_reader_router.py", V27 / "selector_protocol.py",
    ]
    for key in ("embedding", "reranker"):
        files.extend(p for p in sorted(Path(assets[key]).rglob("*"))
                     if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"})
    manifest = {str(path): shared.sha(path) for path in files}
    pre = {
        "version": "routing_frozen_field_recovery_first_retrieval_v40",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_changed": False, "training_ready": False,
        "assets": assets, "query_config": query_config, "pipeline_config": pipeline_config,
        "reranker_max_length": retrieval["reranker_max_length"],
        "sha256": manifest, "rows": rows, "verified_nvidia_mapping": numeric_mapping(),
        "protocol": [
            "Four v39 invalid-intent recovery cases and two byte-identical-hypothesis controls, 12 first-round retrieval conditions.",
            "Use ONLY the routing-frozen, old-cue-preserving candidate; do not use v39 basic repair or v37 model-only projection.",
            "Same actual embedding/reranker/indexes/config/current canonical source, alternating condition order.",
            "Build initial queries and call canonical _retrieve_round_evidence at round1 with a fresh PipelineRunState.",
            "Capture actual dense/sparse query calls, full retrieved objects, trace summary, prepared evidence and standardized minimal first-round preview.",
            "No LLM generation, answer validation, next retrieval round, API judge, training or promotion.",
            "MiniRAG index and graph calls forbidden; web context disabled; existing indexes read-only, never rebuilt.",
            "Preview is a named canonical minimal-renderer diagnostic, not a claim that all production reader routing paths use that exact prompt.",
            "No manually chosen target chapters, answer fragments or expected answers are provided to retrieval.",
            "Unchanged controls expose implementation variation; do not assume deterministic-by-seed retrieval or label ID changes as accuracy gains.",
            "No lost outcomes or retries; any failed retrieval condition stays recorded.",
            "Verify numeric GPU2 mapping and obtain a NEW continuous >=60s idle window under the existing lock before loading models.",
        ],
    }
    make_pipeline(pre, helpers, retriever=None)
    shared.save(HERE / "preflight.json", pre)
    print({"conditions": 12, "query_counts": {r["id"]: {k: len(v) for k, v in r["queries"].items()} for r in rows},
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


def verify(pre):
    assert all(shared.sha(Path(p)) == h for p, h in pre["sha256"].items())


def plain_record(value):
    def scalar(item):
        import numpy as np
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"unsupported record value {type(item)}")
    return json.loads(json.dumps(value, ensure_ascii=False, default=scalar, allow_nan=False))


def retrieve():
    pre = shared.load(HERE / "preflight.json")
    verify(pre)
    assert not (HERE / "retrieval.progress.jsonl").exists()
    idle = shared.load(HERE / "idle_samples.json")
    assert len(idle["samples"]) >= 13 and idle["elapsed_seconds"] >= 60
    shared.gpu_idle()
    import torch
    torch.manual_seed(20260909)
    helpers = imports()
    from asa_arknight_story_agent.retrieval.hybrid import ArknightsHybridRetriever
    assets = pre["assets"]
    retriever = ArknightsHybridRetriever.from_paths(
        embedding_model_path=Path(assets["embedding"]), reranker_model_path=Path(assets["reranker"]),
        reranker_max_length=pre["reranker_max_length"], documents_path=Path(assets["documents"]),
        faiss_index_path=Path(assets["faiss"]), bm25_tokens_path=Path(assets["bm25"]),
        minirag_index_path=None, device="cuda:0",
    )
    assert retriever.minirag_index is None and retriever.reranker is not None
    pipeline, generator = make_pipeline(pre, helpers, retriever)
    queries = []
    for method in ("dense_search", "sparse_search"):
        original = getattr(retriever, method)

        def traced(query, *args, _method=method, _original=original, **kwargs):
            record = {"method": _method, "query": query, "args": args, "kwargs": kwargs}
            queries.append(record)
            hits = _original(query, *args, **kwargs)
            record["returned_document_ids"] = [hit["document"].get("id") for hit in hits]
            return hits

        setattr(retriever, method, traced)
    results = []
    with (HERE / "retrieval.progress.jsonl").open("x", encoding="utf-8") as journal:
        for row in pre["rows"]:
            for kind in row["order"]:
                print({"starting": row["id"], "condition": kind}, flush=True)
                before = len(queries)
                item = {"id": row["id"], "condition": kind, "status": "error"}
                started = time.monotonic()
                try:
                    hypothesis = helpers["HypothesisDocument"](**row[kind])
                    actual_queries = pipeline._build_initial_queries(row["question"], hypothesis)
                    assert actual_queries == row["queries"][kind]
                    state = helpers["PipelineRunState"]()
                    graph, web = pipeline._retrieve_round_evidence(
                        question=row["question"], hypothesis=hypothesis, pending_queries=actual_queries,
                        round_index=1, state=state, progress_callback=None,
                    )
                    assert graph is web is None and retriever.minirag_index is None
                    item.update(
                        actual_hypothesis=asdict(hypothesis), actual_queries=actual_queries,
                        evidence=deepcopy(state.evidence),
                        trace=pipeline._build_step_record(
                            round_index=1, pending_queries=actual_queries,
                            hypothesis_task_type=helpers["INITIAL_HYPOTHESIS_TASK_TYPE"], hypothesis=hypothesis, state=state,
                            minirag_expansion_record=None, web_context_record=None,
                        ),
                        **preview(helpers, pipeline, row["question"], hypothesis, state.evidence),
                        status="ok",
                    )
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                item.update(actual_search_calls=queries[before:], seconds=time.monotonic() - started)
                item = plain_record(item)
                results.append(item)
                journal.write(json.dumps(item, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print({"completed": len(results), "id": row["id"], "condition": kind,
                       "status": item["status"], "seconds": item["seconds"],
                       "evidence_count": len(item.get("evidence", []))}, flush=True)
    assert generator.calls == 0
    verify(pre)
    shared.save(HERE / "retrieval.json", {
        "version": pre["version"], "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "conditions": results, "llm_generation_calls": generator.calls,
        "source_models_and_indexes_unchanged": True,
        "production_changed": False, "training_ready": False,
    })


def run():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2" and os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    pre = shared.load(HERE / "preflight.json")
    assert numeric_mapping() == pre["verified_nvidia_mapping"]
    verify(pre)
    assert not (HERE / "idle_samples.json").exists() and not (HERE / "retrieval.progress.jsonl").exists()
    shared.HERE, shared.run = HERE, retrieve
    shared.idle_then_run()


def check():
    pre = shared.load(HERE / "preflight.json")
    result = shared.load(HERE / "retrieval.json")
    verify(pre)
    helpers = imports()
    pipeline, generator = make_pipeline(pre, helpers, retriever=None)
    assert result["preflight_sha256"] == shared.sha(HERE / "preflight.json")
    journal = [json.loads(line) for line in (HERE / "retrieval.progress.jsonl").read_text().splitlines()]
    assert result["conditions"] == journal and len(journal) == 12
    source_rows = {row["id"]: row for row in pre["rows"]}
    expected_order = [(row["id"], kind) for row in pre["rows"] for kind in row["order"]]
    assert [(row["id"], row["condition"]) for row in journal] == expected_order
    counts = Counter()
    for row in journal:
        counts[row["status"]] += 1
        if row["status"] != "ok":
            assert row["error"]
            continue
        source = source_rows[row["id"]]
        kind = row["condition"]
        hypothesis = helpers["HypothesisDocument"](**source[kind])
        assert row["actual_hypothesis"] == source[kind]
        assert row["actual_queries"] == source["queries"][kind] == pipeline._build_initial_queries(source["question"], hypothesis)
        for index, query in enumerate(row["actual_queries"]):
            for offset, method in enumerate(("dense_search", "sparse_search")):
                call = row["actual_search_calls"][index * 2 + offset]
                assert call["method"] == method and call["query"] == query
        # Renderer replay requires no retrieval or model, using the actual packed objects.
        prompt, visible = helpers["build_minimal_conclusion_prompt"](
            question=source["question"], current_hypothesis=hypothesis, evidence=row["evidence"],
            prompt_evidence=row["prompt_evidence"], current_round=1, max_retrieval_rounds=2,
            prompt_evidence_top_k=10, grounding_mode="quote",
        )
        assert prompt == row["preview_prompt"] and shared.digest(prompt) == row["preview_prompt_sha256"]
        assert visible == row["evidence_brief"]
    assert generator.calls == result["llm_generation_calls"] == 0
    comparisons = []
    indexed = {(row["id"], row["condition"]): row for row in journal}
    for source in pre["rows"]:
        control, candidate = (indexed[(source["id"], kind)] for kind in ("control", "candidate"))
        record = {"id": source["id"], "no_intervention_control": source["no_intervention_control"]}
        if control["status"] == candidate["status"] == "ok":
            record.update(
                same_visible_evidence=control["evidence_brief"] == candidate["evidence_brief"],
                same_evidence_objects=control["evidence"] == candidate["evidence"],
                control_preview_sha256=control["preview_prompt_sha256"],
                candidate_preview_sha256=candidate["preview_prompt_sha256"],
            )
        comparisons.append(record)
    shared.save(HERE / "postflight_v1.json", {
        "version": "routing_frozen_first_retrieval_postflight_v40",
        "preflight_sha256": shared.sha(HERE / "preflight.json"),
        "retrieval_sha256": shared.sha(HERE / "retrieval.json"),
        "status_counts": counts, "comparisons_not_accuracy": comparisons,
        "actual_hypotheses_and_search_queries_verified": True, "preview_renderer_replayed": True,
        "llm_generation_calls": 0, "production_changed": False, "training_ready": False,
        "semantic_source_review_complete": False,
    })
    print({"status_counts": dict(counts), "comparisons_not_accuracy": comparisons}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": run, "check": check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_retrieval_probe.py prepare|run|check")
    operations[sys.argv[1]]()
