#!/usr/bin/env python3
import argparse, hashlib, json, os, re
from pathlib import Path

def question(row):
    for turn in row.get("conversations", []):
        if turn.get("from") == "human":
            m = re.search(r"(?m)^question:\s*(.+)$", turn.get("value", ""))
            if m:
                return m.group(1).strip()
    return row.get("id", "")

def bucket(q):
    h = int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < 15 else ("val" if h < 30 else "train")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-input-tokens", type=int, default=0)
    args = ap.parse_args()
    rows = json.load(open(args.input, encoding="utf-8"))
    if args.max_input_tokens:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("${REMOTE_HOME}/ASA-ArknightStoryAgent/model/qwen3.5-4b", trust_remote_code=True)
            kept = []
            dropped = []
            for r in rows:
                txt = "\n".join(t.get("value", "") for t in r.get("conversations", []))
                n = len(tok(txt, add_special_tokens=False)["input_ids"])
                (kept if n <= args.max_input_tokens else dropped).append(r)
            rows = kept
        except Exception as exc:
            raise SystemExit(f"tokenizer filtering failed: {exc}")
    else:
        dropped = []
    groups = {}
    for r in rows:
        groups.setdefault(question(r), []).append(r)
    out = {k: [] for k in ("train", "val", "test")}
    for q, rs in sorted(groups.items()):
        out[bucket(q)].extend(rs)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for split, rs in out.items():
        with open(root / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(rs, f, ensure_ascii=False, indent=2)
    manifest = {
        "input": args.input,
        "total": len(rows),
        "dropped_long": len(dropped),
        "max_input_tokens": args.max_input_tokens,
        "question_groups": len(groups),
        "split_counts": {k: len(v) for k, v in out.items()},
        "question_counts": {k: len({question(r) for r in v}) for k, v in out.items()},
        "protocol": "sha1(question)%100: test<15, val<30, train>=30",
    }
    json.dump(manifest, open(root / "split_manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    main()
