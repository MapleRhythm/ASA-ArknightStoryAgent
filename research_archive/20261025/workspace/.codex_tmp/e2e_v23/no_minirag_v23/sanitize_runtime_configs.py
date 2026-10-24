#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CANONICAL_VALUES = {
    "enable_minirag": False,
    "minirag_index_path": None,
    "minirag_top_k": 0,
    "minirag_weight": 0.0,
    "minirag_mode_weights": {},
    "minirag_chapter_isolation": False,
    "minirag_auto_second_retrieval": False,
    "minirag_scope_seed_top_k": 0,
    "minirag_expansion_query_top_k": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    changed: list[str] = []
    for root in args.roots:
        for path in sorted(root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            retrieval = payload.get("retrieval")
            if not isinstance(retrieval, dict):
                continue
            before = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            retrieval.update(CANONICAL_VALUES)
            after = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if before == after:
                continue
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            changed.append(str(path))
    print(json.dumps({"changed": changed, "count": len(changed)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
