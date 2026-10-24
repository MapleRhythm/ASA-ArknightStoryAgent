"""Exercise the actual 9B tokenizer against all saved baseline request envelopes."""
import argparse
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from run_qwen35_9b import canonical_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--calls", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    assert not args.out.exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    events = [json.loads(l) for l in args.calls.read_text().splitlines()]
    checks = []
    for event in events:
        if event["event"] != "request":
            continue
        original = event["prompt"]
        native = canonical_prompt(tokenizer, original)
        evidence = "\nevidence:\n" in original
        if evidence:
            before = original.split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            after = native.split("\nevidence:\n", 1)[1].rsplit("\noutput_schema:", 1)[0]
            assert before == after
            assert re.findall(r"(?m)^\[(E\d+)\]", after) == [f"E{i}" for i in range(1, 21)]
        budget = event["options"]["max_tokens"]
        tokens = len(tokenizer.encode(native, add_special_tokens=True))
        assert tokens + budget <= 10000
        checks.append({"question_index": event["question_index"], "has_evidence": evidence,
                       "tokens": tokens, "output_budget": budget, "evidence_unchanged": True})
    assert len(checks) == 109
    try:
        canonical_prompt(tokenizer, "unrecognized prompt")
    except ValueError:
        invalid_rejected = True
    else:
        raise AssertionError("Unknown chat envelope was accepted")
    result = {"passed": True, "tested_requests": len(checks), "unknown_envelope_rejected": invalid_rejected,
              "max_tokens": max(c["tokens"] for c in checks),
              "full_top20_prompts": sum(c["has_evidence"] for c in checks), "checks": checks}
    with args.out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)


if __name__ == "__main__":
    main()
