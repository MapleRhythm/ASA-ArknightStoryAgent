#!/usr/bin/env python3
"""Freeze quote renderer pairs; no reference answer enters the generation input."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from audit_rendered_evidence_visibility import (  # noqa: E402
    hypothesis_from_snapshot, read_rows, rerank_candidates, select_items,
    sha256_file, text_sha,
)
from asa_arknight_story_agent.inference.evidence.texts import prefer_direct_prompt_text  # noqa: E402
from asa_arknight_story_agent.inference.generation.minimal_conclusion_prompt import (  # noqa: E402
    build_minimal_conclusion_prompt,
)


def split_runtime_prompt(prompt: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
        r"<\|im_start\|>assistant\n\{", prompt, flags=re.DOTALL,
    )
    if not match:
        raise ValueError("unexpected runtime quote prompt template")
    return match.group(1), match.group(2)


def build_rows(
    snapshots: list[dict[str, Any]], audit: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles = [item for item in audit["profiles"] if item["mode"] == "quote"]
    if len(profiles) != 1:
        raise ValueError("expected one quote profile")
    profile = profiles[0]
    prefix = f'runtime_gpu_reranker_qwen35_4b/production/{profile.get("runtime_packing", "break")}'
    rows = []
    for index, snapshot in enumerate(snapshots):
        candidates = rerank_candidates(snapshot)
        selected = select_items(
            snapshot, candidates, mode="quote", strategy="production",
            top_k=profile["top_k"],
        )
        pair = []
        # Alternate pair order; both arms use the same model/device/decoding.
        renderers = ("runtime", "direct_only") if index % 2 == 0 else ("direct_only", "runtime")
        for renderer in renderers:
            prompt, brief = build_minimal_conclusion_prompt(
                question=snapshot["question"],
                current_hypothesis=hypothesis_from_snapshot(snapshot),
                evidence=candidates, prompt_evidence=(
                    selected if renderer == "runtime" else
                    [prefer_direct_prompt_text(item) for item in selected]
                ),
                current_round=1, max_retrieval_rounds=2,
                prompt_evidence_top_k=profile["top_k"],
                evidence_max_chars_per_doc=profile["per_doc"],
                evidence_max_total_chars=profile["total"], grounding_mode="quote",
            )
            key = prefix + ("/direct_only" if renderer == "direct_only" else "")
            matching = [item for item in audit["variants"][key]["rows"] if item["id"] == snapshot["id"]]
            if len(matching) != 1 or matching[0]["prompt_sha256"] != text_sha(prompt):
                raise ValueError("runtime prompt drifted from frozen visibility audit")
            if matching[0]["evidence_brief"] != brief:
                raise ValueError("evidence drifted from visibility audit")
            system, user = split_runtime_prompt(prompt)
            pair.append((system, user.replace(brief, "<FROZEN_EVIDENCE>", 1)))
            for template in ("runtime_raw", "canonical_non_thinking"):
                rows.append({
                    "id": f'{snapshot["id"]}::{renderer}::{template}',
                    "source_id": snapshot["id"],
                    "question": snapshot["question"],
                    "question_family": snapshot.get("question_family"),
                    "renderer": renderer, "template": template,
                    "system": system, "user": user,
                    "runtime_prompt": prompt, "runtime_prompt_sha256": text_sha(prompt),
                    "evidence_brief": brief, "evidence_sha256": text_sha(brief),
                    "evidence_id_to_corpus_id": {
                        item["evidence_id"]: item["source_id"] for item in matching[0]["blocks"]
                    },
                    "evaluation_only": True, "training_ready": False,
                })
        if pair[0] != pair[1]:
            raise ValueError("renderer comparison changed non-evidence instructions")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate input IDs")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite matrix")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if sha256_file(args.snapshots) != audit["snapshots"]["sha256"]:
        parser.error("snapshot hash mismatch")
    rows = build_rows(read_rows(args.snapshots), audit)
    result = {
        "version": "asa_quote_render_ablation_v1",
        "evaluation_only": True, "training_ready": False,
        "source_audit_sha256": sha256_file(args.audit),
        "source_snapshot_sha256": sha256_file(args.snapshots),
        "builder_sha256": sha256_file(Path(__file__)),
        "question_count": len({row["source_id"] for row in rows}),
        "notes": [
            "Two renderer arms x two template strata; compare renderers within a stratum.",
            "runtime_raw preserves the exact runtime assistant '{' prefill.",
            "canonical_non_thinking uses checkpoint tokenizer, without JSON prefill.",
            "Frozen development smoke, not production E2E or a new blind test.",
            "No reference answers, gold IDs or reviewer notes enter model messages.",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
