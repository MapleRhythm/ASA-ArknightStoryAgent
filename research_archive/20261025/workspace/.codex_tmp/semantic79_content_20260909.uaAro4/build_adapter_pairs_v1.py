"""Six development questions, identical prompts, adapter-on versus adapter-off."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    destination = HERE / "adapter_pairs12_v1.matrix.json"
    assert not destination.exists()
    original_path = RUN / "fixed79.current_quote.matrix.json"
    source = {r["id"]: r for r in load(original_path)["rows"]}
    previous = load(RUN / "semantic79_content_v2.fPkkhM/slot_feedback_paired8_v1.matrix.json")
    chosen = {r["id"] for r in previous["selection"]}
    for prefix in ("relabel-4835313d2c74e78706d2b4e3::", "relabel-c637f5c136cd58d46096f7ce::"):
        matches = [rid for rid in source if rid.startswith(prefix)]
        assert len(matches) == 1
        chosen.add(matches[0])
    assert len(chosen) == 6
    rows, selection = [], []
    for index, rid in enumerate(sorted(chosen)):
        row = source[rid]
        seed = int(hashlib.sha256(("adapter-weight-ablation-v1:" + rid).encode()).hexdigest()[:8], 16) % (2**31)
        variants = [False, True] if index % 2 == 0 else [True, False]
        for active in variants:
            variant = "adapter_on_v1" if active else "adapter_off_same_tokenizer_v1"
            candidate = deepcopy(row)
            candidate.update(
                id=rid + "::" + variant, parent_record_id=rid, renderer=variant,
                sampling_seed=seed, use_adapter=active,
                evaluation_only=True, training_ready=False,
            )
            assert candidate["runtime_prompt"] == row["runtime_prompt"]
            assert candidate["evidence_brief"] == row["evidence_brief"]
            rows.append(candidate)
        selection.append({
            "id": rid, "question": row["question"], "sampling_seed": seed,
            "selection_reason": "Known development failure or positive control; not holdout.",
        })
    assert len(rows) == 12 and len({r["question"] for r in rows}) == 6
    frozen_config = RUN / "after/configs/runtime_gpu_reranker_qwen35_4b.json"
    canonical_config = ROOT / "release/ASA-ArknightStoryAgent/configs/runtime_gpu_reranker_qwen35_4b.json"
    effective = deepcopy(load(frozen_config)["generator"])
    effective["ctx_size"] = effective["vllm"]["max_model_len"] = 12288
    assert effective == load(canonical_config)["generator"]
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "version": "adapter_pairs12_v1",
        "evaluation_only": True, "training_ready": False,
        "source_matrix_sha256": sha(original_path), "builder_sha256": sha(Path(__file__)),
        "frozen_config_sha256": sha(frozen_config), "canonical_config_sha256": sha(canonical_config),
        "max_context": 12288, "max_new_tokens": 2048,
        "selection": selection, "rows": rows,
        "predeclared_protocol": [
            "Same unmodified original runtime prompt and complete evidence within each pair.",
            "Same configured tokenizer, engine, explicit seed, temperature/top-p and context budget.",
            "Only actual lora_request switches between the configured adapter and None.",
            "No previous answer, source-review correction, desired answer or slot feedback is appended.",
            "Three adapter-off-first and three adapter-on-first pairs, in deterministic balanced order.",
            "Track question coverage, cited-source attribution, causal support and formatting separately.",
            "Failures remain in the denominator; no training or production promotion on this selected development set.",
            "Adapter-off in one LoRA-enabled engine isolates weights; it is not an independently configured base-only deployment.",
            "Do not combine with prior Transformers strict9 results or claim production accuracy.",
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"matrix": str(destination), "rows": 12, "selection": selection}, ensure_ascii=False))


if __name__ == "__main__":
    main()
