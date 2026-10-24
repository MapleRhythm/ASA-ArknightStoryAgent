"""Unfiltered generated outputs for manual comparison, NOT QA accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from backend import PromptEncoder, load_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-smoke", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    tokenizer = Tokenizer.from_file(str(root / "official/tokenizer.json"))
    model, manifest = load_backend("artifact", dtype=torch.bfloat16)
    model.eval()
    stages = ["frozen_no_prompt", "initial_prompt", "trained_prompt"]
    if args.cache_smoke:
        stages = ["frozen_no_prompt"]
        rows = [{"id": "backend_cache_smoke", "question": "谁救出了博士？",
                 "prompt": "<|im_start|>user\n证据：阿米娅救出了博士。谁救出了博士？"
                           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                 "documents": []}]
    else:
        run = json.loads((Path(args.run) / "run.json").read_text())
        assert run["training_complete"] and run["backbone_bytewise_unchanged"]
        rows = [json.loads(l) for l in (Path(args.data) / "val.jsonl").read_text().splitlines()][:args.samples]
    eos = [i for i in (tokenizer.token_to_id("<|im_end|>"),
                       tokenizer.token_to_id("<|endoftext|>")) if i is not None]
    assert eos
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        for stage in stages:
            encoder = None
            if stage != "frozen_no_prompt":
                name = "initial_adapter" if stage == "initial_prompt" else "final_adapter"
                encoder = PromptEncoder.load(Path(args.run) / name)
                encoder.eval()
            for row in rows:
                if not args.cache_smoke:
                    assert len(row["documents"]) == 20 and not row["truncated"]
                    assert all(d["text"] in row["prompt"] for d in row["documents"])
                ids = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
                max_new = 8 if args.cache_smoke else 1024
                assert len(ids) + (encoder.length if encoder else 0) + max_new <= 32768
                started = time.monotonic()
                with torch.no_grad():
                    embedded = model.model.embed_tokens(torch.tensor([ids], device="cuda"))
                    if encoder is not None:
                        embedded = encoder.inject(embedded)
                    generated = model.generate(
                        inputs_embeds=embedded,
                        attention_mask=torch.ones(embedded.shape[:2], device="cuda", dtype=torch.long),
                        max_new_tokens=max_new, do_sample=False, use_cache=True,
                        eos_token_id=eos, pad_token_id=eos[0])
                tokens = generated[0].tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                content = tokenizer.decode(tokens, skip_special_tokens=True)
                try:
                    parsed = json.loads(content)
                except ValueError:
                    parsed = None
                receipt = {
                    "kind": "unfiltered_generated_answer_not_adjudicated_QA_accuracy",
                    "stage": stage, "id": row["id"], "question": row["question"],
                    "source_sha256": manifest["source_sha256"],
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "prompt_tokens": len(ids), "evidence_count": len(row["documents"]),
                    "input_truncated": False, "raw_output": raw, "generated_tokens": tokens,
                    "parsed_output_for_offline_inspection_only": parsed,
                    "content_or_citation_validation_enabled": False,
                    "natural_stop": bool(tokens and tokens[-1] in eos),
                    "output_reached_length_limit": bool(len(tokens) >= max_new and tokens[-1] not in eos),
                    "elapsed_seconds": time.monotonic() - started,
                    "gguf_server_injection_validated": False,
                }
                stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
            del encoder


if __name__ == "__main__":
    main()
