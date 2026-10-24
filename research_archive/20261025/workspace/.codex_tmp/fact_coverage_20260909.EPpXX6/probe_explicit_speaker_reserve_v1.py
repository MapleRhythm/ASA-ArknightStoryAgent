"""CPU diagnostic on all58 sparse pools and one current-model scored120.

No model invocation, gold data, training, new generated answer or default edit.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from build_fact_coverage_pairs_v1 import HERE, ROOT, SOURCE, load, sha, digest
from verify_empty_quote_candidate_v1 import counts, save, snapshot
from explicit_speaker_reserve_v1 import explicit_speakers, reserve_ranked

PREFLIGHT = HERE / "explicit_speaker_reserve_preflight_v1.json"
DESTINATION = HERE / "explicit_speaker_reserve_v1.json"
INDEX = ROOT / "indexes/arknights_story/documents.jsonl"
BUDGETS = (0, 2, 4, 6)


def main():
    assert not PREFLIGHT.exists() and not DESTINATION.exists()
    tests = counts(HERE / "explicit_speaker_reserve_tests_v1.xml")
    assert tests == {"tests": 16, "failures": 0, "errors": 0, "skipped": 0}
    frozen = {str(p): sha(p) for p in [
        INDEX, Path(__file__), HERE / "explicit_speaker_reserve_v1.py",
        HERE / "test_explicit_speaker_reserve_v1.py", HERE / "explicit_speaker_reserve_tests_v1.xml",
        HERE / "canonical_sparse58_probe_v1.json", HERE / "susie_seed_retention_v1.json",
        HERE / "susie_seed_retention_preflight_v1.json",
    ]}
    source = snapshot(SOURCE)
    assert source == load(HERE / "checkpoint_v20_state.json")["runtime_source_sha256"]
    save(PREFLIGHT, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "cpu_only": True,
        "artifacts_sha256": frozen, "runtime_source_sha256": source,
        "top_k": 12, "development_sweep_reserves": list(BUDGETS),
        "protocol": [
            "Vocabulary comes only from unchanged canonical document.segments[].speaker.",
            "Use literal question matches, longest overlapping name; no guessed aliases or hypothesis/gold names.",
            "Reserve applies only if at least two known speakers appear in the question.",
            "Keep all candidates, all original text and within-selected-set original order.",
            "Evaluate all58 existing sparse pools; reuse real scores for the single diagnosed question only.",
            "Budgets0/2/4/6 form an explicitly development-tuned sweep after seeing a failure, NOT heldout.",
            "Name coverage is not event/answer support. Compare lost top12 IDs, not only gained target.",
            "No new full RAG result, recall, generation accuracy, default change or training.",
        ],
    })
    with INDEX.open() as stream:
        documents = [json.loads(line) for line in stream if line.strip()]
    by_id = {d["id"]: d for d in documents}
    assert len(by_id) == len(documents) == 54252
    vocabulary = {
        segment.get("speaker", "").strip()
        for doc in documents for segment in doc.get("segments", [])
        if isinstance(segment, dict) and isinstance(segment.get("speaker", ""), str)
    }
    vocabulary = {v for v in vocabulary if 2 <= len(v) <= 32}
    sparse = load(HERE / "canonical_sparse58_probe_v1.json")
    assert sha(INDEX) == sparse["source_artifacts_sha256"][str(INDEX)]
    rows = []

    def summarize_order(old, new, anchors, id_of, text_of, reserve):
        original_ids = [id_of(r) for r in old[:12]]
        new_ids = [id_of(r) for r in new[:12]]
        assert len(new) == len(old) and {id_of(r) for r in old} == {id_of(r) for r in new}
        return {
            "reserve": reserve, "top12": new_ids,
            "entered_top12": [i for i in new_ids if i not in original_ids],
            "left_top12": [i for i in original_ids if i not in new_ids],
            "joint_name_top12": sum(bool(anchors) and all(a in text_of(r) for a in anchors) for r in new[:12]),
        }

    for row in sparse["results"]:
        anchors = explicit_speakers(row["question"], vocabulary)
        hits = row["hits"]
        assert len(hits) == 120
        id_of = lambda h: h["document_id"]
        text_of = lambda h: by_id[h["document_id"]].get("clean_text", "")
        for hit in hits:
            assert documents[hit["doc_index"]]["id"] == hit["document_id"]
        arms = []
        for reserve in BUDGETS:
            new = reserve_ranked(hits, anchors, top_k=12, reserve=reserve, id_of=id_of, text_of=text_of)
            arms.append(summarize_order(hits, new, anchors, id_of, text_of, reserve))
        rows.append({
            "index": row["index"], "id": row["id"], "question": row["question"], "explicit_speakers": anchors,
            "joint_name_pool120": sum(bool(anchors) and all(a in text_of(h) for a in anchors) for h in hits),
            "conditions": arms,
        })

    seed = load(HERE / "susie_seed_retention_v1.json")
    target_question = load(HERE / "susie_seed_retention_preflight_v1.json")["query"]
    anchors = explicit_speakers(target_question, vocabulary)
    target = "activities/act10mini/level_act10mini_st05#chunk-0025"
    measured = []
    for arm in seed["conditions"]:
        old = arm["ranked"]
        id_of = lambda r: r["document_id"]
        text_of = lambda r: r["document"]["clean_text"]
        for item in old:
            assert item["document"] == by_id[id_of(item)]
        for reserve in BUDGETS:
            new = reserve_ranked(old, anchors, top_k=12, reserve=reserve, id_of=id_of, text_of=text_of)
            stat = summarize_order(old, new, anchors, id_of, text_of, reserve)
            stat.update(
                input_condition=arm["condition"],
                selected_rescue_segment_rank=next(i for i, r in enumerate(new, 1) if id_of(r) == target),
                ranked_ids=[id_of(r) for r in new],
            )
            measured.append(stat)
    assert all(sha(Path(p)) == h for p, h in frozen.items()) and snapshot(SOURCE) == source
    summary = {
        "states": len(rows), "vocabulary_size": len(vocabulary),
        "speaker_count_distribution": dict(Counter(len(r["explicit_speakers"]) for r in rows)),
        "multi_speaker_questions": sum(len(r["explicit_speakers"]) >= 2 for r in rows),
        "changed_sparse_top12_states": {
            str(b): sum(bool(r["conditions"][i]["entered_top12"]) for r in rows)
            for i, b in enumerate(BUDGETS)
        },
    }
    save(DESTINATION, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_only": True, "training_ready": False, "preflight_sha256": sha(PREFLIGHT),
        "summary": summary, "results": rows, "single_question": target_question,
        "single_question_explicit_speakers": anchors, "single_question_conditions": measured,
        "vocabulary_sha256": digest(json.dumps(sorted(vocabulary), ensure_ascii=False)),
        "source_and_inputs_unchanged": True, "new_model_calls": 0,
        "new_full_rag_recall": None, "new_certified_answer_accuracy": None,
    })
    print(json.dumps({"summary": summary, "selected_target": [
        {k: r[k] for k in ("input_condition", "reserve", "selected_rescue_segment_rank")}
        for r in measured]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
