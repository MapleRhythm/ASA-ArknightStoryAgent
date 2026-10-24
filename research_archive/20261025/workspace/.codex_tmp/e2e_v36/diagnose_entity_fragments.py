"""Read-only CPU reproduction; do not change tokenizer, entities or labels."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
SRC = ROOT / "release/ASA-ArknightStoryAgent/src"
PRE = HERE.parent / "reader_alias_groups_20260910.v35/preflight.json"
sys.path.insert(0, str(SRC))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from asa_arknight_story_agent.inference.common.patterns import QUESTION_TOKEN_RE
    from asa_arknight_story_agent.inference.planning.query_tokens import extract_content_tokens, is_entity_candidate
    from asa_arknight_story_agent.inference.planning.entity_extraction import extract_entities
    from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload
    from asa_arknight_story_agent.config import OPERATOR_ALIAS_MAP_PATH
    assert not (HERE / "diagnosis_v1.json").exists()
    relative = [
        "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/common/patterns.py",
        "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/planning/query_tokens.py",
        "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/planning/entity_extraction.py",
        "release/ASA-ArknightStoryAgent/src/asa_arknight_story_agent/inference/payload/hypothesis_payload.py",
    ]
    files = [ROOT / name for name in relative]
    before = {str(p): sha(p) for p in files}
    history = {}
    for name, path in zip(relative, files, strict=True):
        head = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
        current = path.read_bytes()
        history[name] = {
            "head_sha256": hashlib.sha256(head).hexdigest(),
            "current_sha256": sha(path), "equals_HEAD": current == head,
        }
        if name.endswith("hypothesis_payload.py"):
            merge = b"entities = dedupe_keep_order(entities + heuristic_entities)[:12]"
            assert merge in head and merge in current
            history[name]["heuristic_merge_already_in_HEAD"] = True
    assert all(history[name]["equals_HEAD"] for name in relative[:3])
    pre = json.loads(PRE.read_text())
    rows = pre["rows"][:8]
    observed = []
    for row in rows:
        question = row["question"]
        tokens = extract_content_tokens(question)
        entities = extract_entities(question)
        observed.append({
            "id": row["id"], "question": question,
            "raw_regex_windows": [
                {"text": m.group(), "start": m.start(), "end": m.end()}
                for m in QUESTION_TOKEN_RE.finditer(question)
            ],
            "content_tokens": tokens, "heuristic_entities": entities,
            "saved_hypothesis_entities": row["hypothesis"]["entities"],
            "candidate_flags": {token: is_entity_candidate(token) for token in tokens},
        })
    source = next(r for r in rows if r["id"] == "fixed58-023")
    sample = next(r for r in observed if r["id"] == source["id"])
    fragments = ["出煌", "挂面做法细", "节不足后", "仍然决定继续教煌"]
    assert set(fragments) <= set(sample["heuristic_entities"])
    assert "煌" not in sample["heuristic_entities"]
    # Explicitly synthetic input to isolate the normalization layer. It is NOT
    # reconstructed or claimed to be the original model's pre-normalization output.
    payload = deepcopy(source["hypothesis"])
    payload["entities"] = ["小大厨", "煌", "挂面"]
    payload["keywords"] = ["教学", "坚持", "原因"]
    original_payload = deepcopy(payload)
    normalized = normalize_hypothesis_payload(
        payload, question=source["question"], dialogue_context="",
    )
    assert payload == original_payload
    assert set(fragments) <= set(normalized.entities)
    assert not set(fragments) & set(payload["entities"])
    after = {str(p): sha(p) for p in files}
    assert before == after
    report = {
        "version": "entity_fragment_read_only_diagnosis_v36",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False, "production_changed": False, "training_ready": False,
        "basis_preflight": str(PRE), "basis_preflight_sha256": sha(PRE),
        "source_sha256": before, "script_sha256": sha(Path(__file__)),
        "operator_alias_map_path": str(OPERATOR_ALIAS_MAP_PATH),
        "git_HEAD_comparison": history, "observations": observed,
        "isolated_synthetic_normalization": {
            "original_model_payload_not_claimed": True,
            "payload": payload, "normalized": asdict(normalized),
            "demonstrated_added_fragments": fragments,
        },
        "findings": [
            "Continuous Chinese text is first chunked by a 2-to-8-character regex, before function-word splitting.",
            "The broad entity-candidate predicate admits several broken clause fragments.",
            "Even a synthetic clean entity list gets those heuristic fragments appended by normalization.",
            "The fixed-window tokenizer and heuristic merge are not new changes from the recent term-cleaning patch.",
            "The same content-token extractor is imported by grounding; globally changing it would also change validation.",
        ],
        "limitations": [
            "Eight previously inspected questions; this is structural reproduction, not a retrieval/answer accuracy measurement.",
            "Synthetic clean payload is not the original planner model output and is not a training label.",
            "No heuristic terms were removed, no retriever rerun, no source/label/config modified.",
            "Do not blindly roll back existing diagnostic-term cleaning or globally replace grounding tokenization.",
        ],
    }
    with (HERE / "diagnosis_v1.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "questions": len(observed), "added_fragments": fragments,
        "merge_already_in_HEAD": True, "production_changed": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
