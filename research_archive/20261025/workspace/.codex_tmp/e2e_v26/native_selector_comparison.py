"""Change only the base selector's chat suffix to its saved native no-think template.

The LoRA reader retains its actual training-compatible qwen3_nothink wrapper.
Both reader conditions are regenerated with the same seeds; no gold is supplied.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "reader_evidence_router_20260909.v27"
sys.path.insert(0, str(PREVIOUS))
import run_reader_router as shared
from selector_protocol import SELECTOR_RULES


def prepare():
    destination = HERE / "preflight.json"
    if destination.exists():
        raise RuntimeError("preflight already exists")
    from transformers import AutoTokenizer

    old = shared.load(PREVIOUS / "preflight.json")
    assert shared.load(PREVIOUS / "postflight_v1.json")["rows"] == 8
    tokenizer = AutoTokenizer.from_pretrained(old["tokenizer"], trust_remote_code=True)
    new = deepcopy(old)
    new.update(
        version="native_base_selector_wrapper8_v28",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_preflight_sha256=shared.sha(PREVIOUS / "preflight.json"),
        parent_generations_sha256=shared.sha(PREVIOUS / "generations.json"),
        comparison="Only selector native no-think assistant suffix differs from v27. The reader and all evidence remain unchanged.",
    )
    for row in new["rows"]:
        messages = [
            {"role": "system", "content": SELECTOR_RULES},
            {"role": "user", "content": json.dumps({
                "question": row["question"], "visible_evidence": row["evidence_brief"],
            }, ensure_ascii=False)},
        ]
        native = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) + "{"
        legacy = row["selector_prompt"]
        assert legacy.endswith("<|im_start|>assistant\n{")
        expected = legacy[:-1] + "<think>\n\n</think>\n\n{"
        assert native == expected, "Native template differs beyond predeclared suffix"
        row["selector_prompt"] = native
        row["selector_prompt_sha256"] = shared.digest(native)
        ids = tokenizer.encode(native, add_special_tokens=True)
        assert len(ids) + 2048 <= old["runtime"]["max_model_len"]
        row["selector_prompt_tokens"] = len(ids)
        row["selector_prompt_token_ids_sha256"] = shared.digest(json.dumps(ids))
    new["sha256"][str(Path(__file__))] = shared.sha(Path(__file__))
    new["sha256"][str(PREVIOUS / "postflight_v1.json")] = shared.sha(PREVIOUS / "postflight_v1.json")
    shared.save(destination, new)
    print(json.dumps({"prepared": 8, "selector_suffix_only": True,
                      "sha256": shared.sha(destination)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "idle_then_run"):
        raise SystemExit("usage: native_selector_comparison.py prepare|idle_then_run")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        shared.HERE = HERE
        shared.SOURCE = PREVIOUS / "src"
        shared.idle_then_run()
