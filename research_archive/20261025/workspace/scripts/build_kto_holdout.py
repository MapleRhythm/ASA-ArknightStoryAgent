#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

def question(row):
    for t in row.get("conversations", []):
        if t.get("from") == "human":
            m = re.search(r"(?m)^question:\s*(.+)$", t.get("value", ""))
            if m:
                return m.group(1).strip()
    return row.get("id", "")

def split(q):
    h = int(hashlib.sha1(q.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 15 else ("val" if h < 30 else "train")

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()
rows = json.load(open(args.input, encoding="utf-8"))
groups = {}
for r in rows:
    groups.setdefault(question(r), []).append(r)
out = {"train": [], "val": [], "test": []}
for q, rs in groups.items():
    out[split(q)].extend(rs)
root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
for k, rs in out.items():
    json.dump(rs, open(root/f"{k}.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump({"rows": {k:len(v) for k,v in out.items()},
           "question_groups": {k:len({question(r) for r in v}) for k,v in out.items()}},
          open(root/"split_manifest.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps({"rows": {k:len(v) for k,v in out.items()}}, ensure_ascii=False))
