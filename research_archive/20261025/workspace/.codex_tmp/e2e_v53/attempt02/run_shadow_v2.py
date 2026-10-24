"""Reuse frozen v53 experiment runner with explicitly versioned span protocol."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V53 = HERE.parent / "span_semantic_shadow_20260910.v53"
sys.path.insert(0, str(V53))
import run_shadow as base
import span_protocol_v2 as protocol

base.HERE, base.protocol = HERE, protocol
shared = base.shared


def prepare():
    assert not (HERE / "preflight.json").exists()
    old = shared.load(V53 / "preflight.json")
    base.verify(old)
    rows = shared.load(V53 / "inputs.json")
    for row in rows:
        row["packet"] = protocol.packet_dict(protocol.make_packet(row["canonical_kwargs"]["evidence_prompt_text"]))
    tasks = base.make_tasks(rows)
    pre = {k: deepcopy(old[k]) for k in (
        "base_model", "adapter", "tokenizer", "runtime", "generator", "query_config", "pipeline_config")}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre["tokenizer"], trust_remote_code=True)
    for item in tasks:
        ids = tokenizer.encode(item["prompt"], add_special_tokens=True)
        item.update(prompt_tokens=len(ids), token_ids_sha256=shared.digest(json.dumps(ids)),
                    context_fits=len(ids) + 2048 <= pre["runtime"]["max_model_len"])
    assert len(tasks) == 21
    # Preserve overflow evidence before refusing; no silent source reduction.
    overflow = [{"task_id": t["task_id"], "prompt_tokens": t["prompt_tokens"]}
                for t in tasks if not t["context_fits"]]
    assert not overflow, overflow
    shared.save(HERE / "inputs.json", rows)
    paths = [
        Path(__file__), HERE / "span_protocol_v2.py", HERE / "test_span_v2.py", HERE / "README.md",
        HERE / "check_cpu_v2.py", HERE / "inputs.json", V53 / "preflight.json", V53 / "inputs.json",
        V53 / "run_shadow.py", V53 / "span_protocol.py", V53 / "test_span_protocol.py",
        V53 / "heldout_review_template.json", V53 / "check_cpu_integration.py",
        V53 / "cpu_integration_v1.log", V53 / "run_v1.log",
    ]
    pre.update(
        version="exact_span80_semantic_shadow_v53_attempt02",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        sha256={str(p): shared.sha(p) for p in paths},
        tasks=tasks, verified_nvidia_mapping=base.reader.retrieval.numeric_mapping(), params=base.PARAMS,
        max_actual_calls=33, production_changed=False, training_ready=False,
        human_gold=False, model_independence=False, independent_holdout_available=False,
        source_full_coverage=True, old_citations_rebound=False,
        max_fixed_prompt_tokens=max(t["prompt_tokens"] for t in tasks),
        previous_attempt="Preserved attempt01 CPU quote-limit incompatibility and pre-GPU idle rejection.",
        changes="Only NEW protocol partition<=80, draft refs<=2, compact full-source display, versioned T IDs.",
    )
    shared.save(HERE / "preflight.json", pre)
    print({"prepared_tasks": len(tasks), "max_calls": 33,
           "longest_fixed_prompt": pre["max_fixed_prompt_tokens"],
           "span_counts": {r["family"]: len(r["packet"]["spans"]) for r in rows},
           "preflight_sha256": shared.sha(HERE / "preflight.json")}, flush=True)


if __name__ == "__main__":
    operations = {"prepare": prepare, "run": base.run, "check": base.check}
    if len(sys.argv) != 2 or sys.argv[1] not in operations:
        raise SystemExit("usage: run_shadow_v2.py prepare|run|check")
    operations[sys.argv[1]]()
