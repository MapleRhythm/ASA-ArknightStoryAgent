#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import DOCUMENTS_PATH, MINIRAG_GRAPH_PATH, OPERATOR_ALIAS_MAP_PATH  # noqa: E402
from goldenglow.retrieval.minirag import build_and_save_minirag_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the lightweight MiniRAG heterogeneous graph index.")
    parser.add_argument("--documents", type=Path, default=DOCUMENTS_PATH)
    parser.add_argument("--operator-aliases", type=Path, default=OPERATOR_ALIAS_MAP_PATH)
    parser.add_argument("--output", type=Path, default=MINIRAG_GRAPH_PATH)
    parser.add_argument(
        "--teacher-annotations",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Path(s) to annotations.cleaned.jsonl (or any JSONL containing per-source "
            "entity_relations). When provided, teacher-extracted triples will be "
            "merged into the graph."
        ),
    )
    parser.add_argument("--progress", action="store_true", help="Print build progress to stderr.")
    parser.add_argument("--progress-interval", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph = build_and_save_minirag_graph(
        documents_path=args.documents,
        alias_map_path=args.operator_aliases,
        output_path=args.output,
        teacher_annotations_path=args.teacher_annotations,
        progress=args.progress,
        progress_interval=args.progress_interval,
    )
    print(
        "MiniRAG graph built: "
        f"documents={graph['document_count']} entities={graph['entity_count']} "
        f"teacher_relations={len(graph.get('teacher_relations', []))} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
