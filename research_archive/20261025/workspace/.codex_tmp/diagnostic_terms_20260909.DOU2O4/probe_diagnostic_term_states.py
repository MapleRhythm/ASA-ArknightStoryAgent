"""Read-only query/state replay over archived inputs, never actual retrieval."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
sys.path.insert(0, str(args.source_root))
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument
from asa_arknight_story_agent.inference.planning.follow_up_hypothesis import (
    build_heuristic_follow_up_hypothesis, merge_hypotheses,
)
from asa_arknight_story_agent.inference.planning.follow_up_query_generation import build_follow_up_hypothesis_queries
from asa_arknight_story_agent.inference.retrieval.query_rendering import build_retrieval_query
from asa_arknight_story_agent.inference.payload.hypothesis_payload import normalize_hypothesis_payload


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


sources, cases = {}, []
old_probe_path = RUN / "fact_scope_patch_v1.ydKEtY/hypothesis_diagnostic_probe_v1.json"
sources[str(old_probe_path)] = sha(old_probe_path)
for result in json.loads(old_probe_path.read_text())["results"]:
    cases.append({
        "id": "probe:" + result["name"], "source_kind": "probe",
        "question": result["input"]["question"], "hypothesis": result["input"],
        "missing_slots": result["missing_slots"],
    })
matrices = [
    "fixed79.current_quote.matrix.json", "after.matrix.json",
    "same_evidence_retry_paired68_v1.matrix.json",
    "question_focus_paired18_v1.matrix.json", "quote80_sft_adapter_probe9_v1.matrix.json",
]
for name in matrices:
    path = RUN / name
    sources[str(path)] = sha(path)
    for row in json.loads(path.read_text())["rows"]:
        cases.append({
            "id": f"matrix:{name}:{row['id']}", "source_kind": "development_matrix",
            "question": row["question"], "hypothesis": row["hypothesis"], "missing_slots": [],
        })
data_root = ROOT / "data/processed/llama_factory/student_rollout_quote80_sft_v1_200_api_verifier_v1_merged_cutoff5632_filtered"
hypothesis_re = re.compile(r"(?ms)^hypothesis:\s*(\{.*?\})\s*(?:^round:|\Z)")
question_re = re.compile(r"(?m)^question:[ \t]*(.+?)\s*$")
seen = set()
training_stats = {"rows": 0, "without_hypothesis": 0, "duplicate_hypothesis_states": 0, "distinct_states": 0}
for split in ("train", "val"):
    path = data_root / f"{split}.json"
    sources[str(path)] = sha(path)
    for row in json.loads(path.read_text()):
        training_stats["rows"] += 1
        prompt = row["conversations"][0]["value"]
        match = hypothesis_re.search(prompt)
        if not match:
            training_stats["without_hypothesis"] += 1
            continue
        value = json.loads(match.group(1))
        question = question_re.search(prompt).group(1).strip()
        hypothesis = {
            "question": str(value.get("question") or question),
            "intent": str(value.get("intent") or "plot_fact"),
            "query_type": str(value.get("query_type") or "reasoning"),
            "entities": value.get("entities", []), "keywords": value.get("keywords", []),
            "expected_answer_type": str(value.get("expected_answer_type") or "short_text"),
            "dialogue_context": str(value.get("dialogue_context") or ""),
        }
        signature = hashlib.sha256(json.dumps(
            {"question": question, "hypothesis": hypothesis}, ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        if signature in seen:
            training_stats["duplicate_hypothesis_states"] += 1
            continue
        seen.add(signature)
        training_stats["distinct_states"] += 1
        cases.append({
            "id": f"historical:{signature}", "source_kind": "historical_training_input",
            "source_record_id": row["id"], "question": question,
            "hypothesis": hypothesis, "missing_slots": [],
        })
results = []
for case in cases:
    hyp = HypothesisDocument(**case["hypothesis"])
    follow = build_heuristic_follow_up_hypothesis(case["question"], hyp, case["missing_slots"])
    merged = merge_hypotheses(hyp, follow)
    result = {
        **case, "follow_up": asdict(follow), "merged": asdict(merged),
        "direct_query": build_retrieval_query(hyp),
        "follow_up_queries": build_follow_up_hypothesis_queries(case["question"], hyp),
        "merged_query": build_retrieval_query(merged),
    }
    try:
        result["normalized"] = asdict(normalize_hypothesis_payload(
            asdict(hyp), question=case["question"], dialogue_context=hyp.dialogue_context,
        ))
    except Exception as exc:
        result["normalization_error"] = f"{type(exc).__name__}: {exc}"
    results.append(result)
assert len({r["id"] for r in results}) == len(results)
for path, digest in sources.items():
    assert sha(Path(path)) == digest
report = {
    "evaluation_only": True, "training_ready": False, "actual_retrieval": False,
    "source_root": str(args.source_root), "source_files": sources,
    "training_input_stats": training_stats, "results": results,
}
with args.output.open("x", encoding="utf-8") as output:
    json.dump(report, output, ensure_ascii=False, indent=2)
print(json.dumps({"states": len(results), "training_input_stats": training_stats}))
