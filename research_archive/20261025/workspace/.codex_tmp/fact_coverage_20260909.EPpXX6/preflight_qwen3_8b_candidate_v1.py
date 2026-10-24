"""CPU-only asset/native-template preflight for a later base-model comparison."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from build_fact_coverage_pairs_v1 import HERE, RUN, SOURCE, digest, load, sha
from verify_empty_quote_candidate_v1 import save, snapshot

MODEL = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/model/qwen3-8b")


def main():
    destination = HERE / "qwen3_8b_native_preflight_v1.json"
    matrix_path = HERE / "qwen3_8b_native58_v1.matrix.json"
    assert not destination.exists() and not matrix_path.exists()
    config = load(MODEL / "config.json")
    assert config["architectures"] == ["Qwen3ForCausalLM"]
    assert config["torch_dtype"] == "bfloat16" and not config.get("auto_map")
    index = load(MODEL / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 5
    assert all((MODEL / shard).is_file() and (MODEL / shard).stat().st_size > 0 for shard in shards)
    sys.path.insert(0, str(RUN))
    from generate_exx_predictions_transformers import model_asset_manifest
    assets = model_asset_manifest(MODEL)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False,
    )
    original = HERE / "fact_coverage_pairs116_v1.matrix.json"
    source = load(original)
    selected = [row for row in source["rows"] if row["renderer"] == "original_prompt_v1"]
    assert len(selected) == 58
    rows, lengths = [], []
    for row in selected:
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["user"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        assert "<think>\n\n</think>" in rendered
        assert row["evidence_brief"] in rendered
        assert row["user"] in rendered and row["system"] in rendered
        ids = tokenizer.encode(rendered, add_special_tokens=True)
        assert ids == tokenizer.encode(rendered, add_special_tokens=False)
        candidate = deepcopy(row)
        candidate.update(
            id=row["parent_record_id"] + "::qwen3_8b_native_nothink_v1",
            renderer="qwen3_8b_native_nothink_v1", runtime_prompt=rendered,
            runtime_prompt_sha256=digest(rendered), native_messages=messages,
            native_template_enable_thinking=False,
            assistant_prefix="",
        )
        rows.append(candidate)
        lengths.append({
            "id": candidate["id"], "input_tokens": len(ids), "reserved_output": 2048,
            "fits_context12288_without_truncation": len(ids) + 2048 <= 12288,
            "prompt_token_ids_sha256": digest(json.dumps(ids)),
        })
    value = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "qwen3_8b_native58_v1", "evaluation_only": True, "training_ready": False,
        "rows": rows, "source_matrix_sha256": sha(original),
        "builder_sha256": sha(Path(__file__)),
        "model_path": str(MODEL), "max_context": 12288, "max_new_tokens": 2048,
        "predeclared_protocol": [
            "CPU-only preparation; no GPU model load, inference, training or default-model change has occurred.",
            "Same58 frozen questions/evidence states, original system/user messages, and paired4B explicit seeds.",
            "Use the base model's own local native chat template with enable_thinking=False, no forced opening brace.",
            "Changing model and native tokenizer/template together is a deployment-candidate comparison, not a weight-only ablation.",
            "No desired answer, old response, prior judge, source-review correction or new evidence is supplied.",
            "Reuse must check assets, canonical runtime, input/output context fit, and a fresh sustained-idle GPU window.",
            "No presumption that8B is better, compatible with the live server, or ready for deployment.",
        ],
    }
    save(matrix_path, value)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True, "model_loaded_on_gpu": False, "new_generations": 0,
        "model_config": config, "model_asset_manifest": assets,
        "weight_shards": shards, "weight_bytes": sum((MODEL / shard).stat().st_size for shard in shards),
        "native_chat_template_sha256": digest(tokenizer.chat_template),
        "all58_inputs_fit": all(r["fits_context12288_without_truncation"] for r in lengths),
        "rows": lengths, "matrix_sha256": sha(matrix_path), "source_hashes_at_preflight": snapshot(SOURCE),
        "builder_sha256": sha(Path(__file__)), "training_ready": False,
        "limitations": value["predeclared_protocol"],
    }
    save(destination, manifest)
    print(json.dumps({
        "cpu_only": True, "shards_hashed": len(shards), "rows": 58,
        "max_input_tokens": max(r["input_tokens"] for r in lengths),
        "all_inputs_fit": manifest["all58_inputs_fit"], "new_generations": 0,
    }), flush=True)


if __name__ == "__main__":
    main()
