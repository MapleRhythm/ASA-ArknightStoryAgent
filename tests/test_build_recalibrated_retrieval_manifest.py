import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_recalibrated_retrieval_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_recalibrated_retrieval_manifest", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _task(task_id: str, question: str, *, doc_id: str = "doc-a") -> dict:
    return {
        "task_id": task_id,
        "question": question,
        "hypothesis": json.dumps(
            {
                "question": question,
                "query_type": "causality",
                "entities": ["甲"],
                "keywords": ["原因"],
            },
            ensure_ascii=False,
        ),
        "round": "1/2",
        "evidence": [{"label": "E1", "doc_id": doc_id, "text": "证据"}],
    }


def _preference(task_id: str, families: list[str]) -> dict:
    return {
        "kto_tag": True,
        "meta": json.dumps(
            {"task_id": task_id, "story_families": families},
            ensure_ascii=False,
        ),
    }


def _recalibrated(
    task_id: str,
    *,
    action: str = "answer_directly",
    evidence_ids: list[str] | None = None,
) -> dict:
    payload = {"next_action": action}
    if action == "answer_directly":
        payload["supported_facts"] = [
            {
                "fact": "甲行动是为了保护同伴。",
                "evidence_ids": evidence_ids or ["E1"],
            }
        ]
    return {
        "id": f"relabel-{task_id}",
        "conversations": [
            {"from": "human", "value": "task: grounded_action_generation"},
            {
                "from": "gpt",
                "value": json.dumps(payload, ensure_ascii=False),
            },
        ],
    }


def test_manifest_uses_recalibrated_citations_and_corpus_ids() -> None:
    rows, audit = MODULE.build_manifest_rows(
        tasks=[_task("task-a", "甲为什么行动？", doc_id="story/a#chunk-1")],
        preferences=[_preference("task-a", ["activities/act-a"])],
        recalibrated=[_recalibrated("task-a")],
    )

    assert len(rows) == 1
    assert rows[0]["gold_evidence_ids"] == ["story/a#chunk-1"]
    assert rows[0]["gold_supported_facts"] == [
        {
            "fact": "甲行动是为了保护同伴。",
            "gold_evidence_ids": ["story/a#chunk-1"],
            "source_evidence_ids": ["E1"],
        }
    ]
    assert rows[0]["question_family"] == "activities/act-a"
    assert rows[0]["query_type"] == "causality"
    assert audit["rows_written"] == 1
    assert audit["supported_fact_count"] == 1
    assert audit["multi_citation_fact_count"] == 0


def test_non_answers_and_unresolved_citations_are_excluded() -> None:
    rows, audit = MODULE.build_manifest_rows(
        tasks=[
            _task("retrieve", "需要继续检索吗？"),
            _task("bad-eid", "错误引用？"),
        ],
        preferences=[
            _preference("retrieve", ["family-a"]),
            _preference("bad-eid", ["family-b"]),
        ],
        recalibrated=[
            _recalibrated("retrieve", action="retrieve_more"),
            _recalibrated("bad-eid", evidence_ids=["E9"]),
        ],
    )

    assert rows == []
    assert audit["exclusions"]["recalibrated_not_answer_directly"] == 1
    assert audit["exclusions"]["unresolved_evidence_id"] == 1


def test_multi_citation_fact_keeps_its_fact_group() -> None:
    task = _task("task-a", "甲为什么行动？", doc_id="story/a#chunk-1")
    task["evidence"].append(
        {
            "label": "E2",
            "doc_id": "story/a#chunk-2",
            "text": "补充证据",
        }
    )
    rows, audit = MODULE.build_manifest_rows(
        tasks=[task],
        preferences=[_preference("task-a", ["family-a"])],
        recalibrated=[_recalibrated("task-a", evidence_ids=["E1", "E2"])],
    )

    assert rows[0]["gold_evidence_ids"] == [
        "story/a#chunk-1",
        "story/a#chunk-2",
    ]
    assert rows[0]["gold_supported_facts"][0]["gold_evidence_ids"] == [
        "story/a#chunk-1",
        "story/a#chunk-2",
    ]
    assert audit["multi_citation_fact_count"] == 1


def test_duplicate_questions_are_dropped_as_a_group() -> None:
    question = "同一个问题？"
    rows, audit = MODULE.build_manifest_rows(
        tasks=[
            _task("task-a", question, doc_id="doc-a"),
            _task("task-b", question, doc_id="doc-b"),
        ],
        preferences=[
            _preference("task-a", ["family-a"]),
            _preference("task-b", ["family-a"]),
        ],
        recalibrated=[
            _recalibrated("task-a"),
            _recalibrated("task-b"),
        ],
    )

    assert rows == []
    assert audit["exclusions"]["duplicate_question"] == 2
