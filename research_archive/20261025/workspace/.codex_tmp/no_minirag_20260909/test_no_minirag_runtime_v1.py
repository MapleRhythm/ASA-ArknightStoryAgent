import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from asa_arknight_story_agent.config import QueryConfig
from asa_arknight_story_agent.inference.generation.direct_answer_prompt import build_answer_prompt
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import (
    build_minimal_conclusion_prompt,
)
from asa_arknight_story_agent.inference.pipeline.state import PipelineRunState
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.pipeline.retrieval_round import retrieve_round_evidence
from asa_arknight_story_agent.inference.retrieval import round as retrieval_round
from asa_arknight_story_agent.inference.retrieval.search import (
    search_minirag_queries,
    search_queries,
)


def hypothesis():
    return HypothesisDocument(
        question="夏栎如何救出苏茜？",
        intent="plot_reasoning",
        query_type="reasoning",
        entities=["夏栎", "苏茜"],
        keywords=["救出"],
        expected_answer_type="剧情事件描述",
    )


def evidence():
    return [{
        "document": {
            "id": "story/1",
            "clean_text": "夏栎抱住苏茜，告诉她一切已经结束。",
            "activity_name": "阴云火花",
            "story_name": "必有所偿",
        },
        "rerank_score": 3.0,
        "evidence_chain_text": (
            "[CHAIN_LEN=1] [CAUSAL_ORDER=model_candidate] "
            "[EVIDENCE_TYPES=(action|outcome)]\n"
            "[CHAIN_MEMBER_1] 夏栎抱住苏茜，告诉她一切已经结束。"
        ),
        "evidence_chain_roles": ["action", "outcome"],
    }]


def test_canonical_runtime_config_disables_every_minirag_entry():
    path = Path(os.environ["ASA_NO_MINIRAG_CONFIG"])
    retrieval = json.loads(path.read_text())["retrieval"]
    assert retrieval["enable_minirag"] is False
    assert retrieval["minirag_index_path"] is None
    assert retrieval["minirag_top_k"] == 0
    assert retrieval["minirag_weight"] == 0.0
    assert retrieval["minirag_mode_weights"] == {}
    assert retrieval["minirag_chapter_isolation"] is False
    assert retrieval["minirag_auto_second_retrieval"] is False
    assert retrieval["minirag_scope_seed_top_k"] == 0
    assert retrieval["minirag_expansion_query_top_k"] == 0


def test_query_config_defaults_are_graph_free():
    config = QueryConfig()
    assert config.minirag_top_k == 0
    assert config.minirag_weight == 0.0
    assert config.minirag_mode_weights == {}
    assert config.minirag_chapter_isolation is False
    assert config.minirag_auto_second_retrieval is False
    assert config.minirag_scope_seed_top_k == 0
    assert config.minirag_expansion_query_top_k == 0


class Retriever:
    def __init__(self):
        self.calls = []

    def dense_search(self, query, *, top_k):
        self.calls.append(("dense", query, top_k))
        return []

    def sparse_search(self, query, *, top_k, storyline_scope):
        self.calls.append(("sparse", query, top_k, storyline_scope))
        return []

    def minirag_search(self, *args, **kwargs):
        raise AssertionError("MiniRAG search must never be called")


@pytest.mark.parametrize("requested", [False, True])
def test_search_queries_never_calls_minirag_even_with_stale_true_flag(requested):
    retriever = Retriever()
    pipeline = SimpleNamespace(
        retriever=retriever,
        query_config=QueryConfig(dense_top_k=2, sparse_top_k=3, minirag_top_k=999),
    )
    dense, sparse, graph = search_queries(
        pipeline=pipeline, queries=["q1", "q2"], enable_minirag=requested,
        minirag_chapter_scope="stale", sparse_storyline_scope=None)
    assert dense == sparse == graph == []
    assert [c[0] for c in retriever.calls] == ["dense", "sparse", "dense", "sparse"]


def test_direct_minirag_search_entrypoint_is_inert():
    retriever = Retriever()
    pipeline = SimpleNamespace(retriever=retriever, query_config=QueryConfig())
    assert search_minirag_queries(
        pipeline=pipeline, queries=["q"], minirag_chapter_scope="stale") == []
    assert retriever.calls == []


def test_pipeline_round_never_enters_graph_scope_or_second_pass():
    calls = []
    class Pipeline:
        query_config = QueryConfig(
            minirag_top_k=999, minirag_weight=1.0,
            minirag_chapter_isolation=True, minirag_auto_second_retrieval=True)
        prompt_evidence_top_k = 10
        web_context_config = SimpleNamespace(enabled=False)
        def _retrieve_first_round_with_scoped_minirag_expansion(self, *args, **kwargs):
            raise AssertionError("graph-driven expansion must not be called")
        def _retrieve_round(self, *args, **kwargs):
            calls.append((args, kwargs))
            return [], [], evidence()
        def merge_raw_definition_evidence(self, question, hyp, rows):
            return rows
    state = PipelineRunState()
    graph_record, web_record = retrieve_round_evidence(
        pipeline=Pipeline(), question="q", hypothesis=hypothesis(), pending_queries=["q"],
        round_index=1, state=state, progress_callback=lambda _: pytest.fail("unexpected progress"))
    assert graph_record is web_record is None
    assert state.evidence == evidence()
    assert len(calls) == 1 and calls[0][1]["enable_minirag"] is False
    assert state.retained_chapter_scope is None
    assert state.scope_retention_enabled is False


def test_finalization_discards_injected_graph_hits(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        retrieval_round, "build_final_rerank_query",
        lambda question, hyp: (question, question, []))
    def fuse(*args, **kwargs):
        captured["graph"] = kwargs["minirag_hits"]
        return []
    monkeypatch.setattr(retrieval_round, "fuse_retrieval_hits", fuse)
    monkeypatch.setattr(retrieval_round, "rerank_and_rescue_hits", lambda *a, **k: [])
    pipeline = SimpleNamespace(
        retriever=object(),
        query_config=QueryConfig(enable_neighbor_expansion=False, minirag_weight=99.0),
    )
    assert retrieval_round.finalize_hits(
        pipeline=pipeline, question="q", hypothesis=hypothesis(),
        dense_hits=[], sparse_hits=[], minirag_hits=[{"forbidden": True}]) == []
    assert captured["graph"] == []


@pytest.mark.parametrize("builder", [build_minimal_conclusion_prompt, build_answer_prompt])
@pytest.mark.parametrize("mode", ["quote", "evidence_id"])
def test_generation_prompts_contain_no_minirag_hints(builder, mode):
    kwargs = {
        "question": "夏栎如何救出苏茜？",
        "evidence": evidence(),
        "prompt_evidence": evidence(),
        "prompt_evidence_top_k": 1,
        "grounding_mode": mode,
    }
    if builder is build_minimal_conclusion_prompt:
        kwargs.update(
            current_hypothesis=hypothesis(), current_round=2, max_retrieval_rounds=2)
    else:
        kwargs.update(hypothesis=hypothesis())
    prompt, brief = builder(**kwargs)
    assert "minirag_hints:" not in prompt
    assert "relations=" not in prompt
    assert "neighbors=" not in prompt
    assert "CHAIN_LEN" not in brief
    assert "夏栎抱住苏茜" in brief
    assert brief in prompt


def test_runtime_cli_guard_rejects_enable_and_returns_zeroed_contract():
    path = Path(os.environ["ASA_NO_MINIRAG_RUNNER"])
    sys.path.insert(0, os.environ["ASA_ROOT_SRC"])
    spec = importlib.util.spec_from_file_location("asa_no_minirag_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "enable_minirag": False, "minirag_top_k": 0, "minirag_weight": 0.0,
        "minirag_mode_weights": {}, "minirag_chapter_isolation": False,
        "minirag_auto_second_retrieval": False, "minirag_scope_seed_top_k": 0,
        "minirag_expansion_query_top_k": 0,
    }
    assert module.require_minirag_disabled(False) == expected
    with pytest.raises(SystemExit, match="disabled"):
        module.require_minirag_disabled(True)
