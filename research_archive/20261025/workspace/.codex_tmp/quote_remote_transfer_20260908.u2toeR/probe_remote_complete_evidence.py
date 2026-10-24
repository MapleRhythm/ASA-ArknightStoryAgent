"""Legacy-compatible, synthetic rendering check; not answer accuracy."""
import argparse
import hashlib
import json
from pathlib import Path

from asa_arknight_story_agent.inference.evidence import rendering
from asa_arknight_story_agent.inference.generation.conclusion_prompt_rendering import build_conclusion_prompt
from asa_arknight_story_agent.inference.generation.direct_answer_prompt import build_answer_prompt
from asa_arknight_story_agent.inference.pipeline.types import HypothesisDocument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite probe")
    config = json.loads(args.config.read_text())["inference"]
    question = "阿米娅是否同意这个申请？"
    hypothesis = HypothesisDocument(
        question=question, intent="plot_fact", query_type="fact",
        entities=["阿米娅"], keywords=["同意"], expected_answer_type="事实",
    )
    tail = "阿米娅：我没有同意这个申请。"
    chain = "杜宾：这里是一段背景介绍。" * 110 + tail
    evidence = [{"document": {"id": "synthetic-tail", "clean_text": tail},
                 "evidence_chain_text": chain}]
    rows = []
    for mode in ("quote", "strict", "grounded"):
        for path in ("minimal", "full", "direct"):
            kwargs = dict(
                prompt_evidence_top_k=config["prompt_evidence_top_k"],
                prompt_evidence=evidence,
                evidence_max_chars_per_doc=config["prompt_evidence_max_chars_per_doc"],
                evidence_max_total_chars=config["prompt_conclusion_evidence_max_total_chars"],
                grounding_mode=mode,
            )
            if path == "direct":
                prompt, brief = build_answer_prompt(question, hypothesis, evidence, **kwargs)
            else:
                prompt, brief = build_conclusion_prompt(
                    question, hypothesis, evidence, [], 1, 2, prompt_mode=path, **kwargs,
                )
            rows.append({
                "mode": mode, "path": path, "body_chars": len(chain),
                "rendered_chars": len(brief), "tail_visible": tail in brief,
                "complete_chain_visible": chain in brief,
                "evidence_embedded_in_prompt": bool(brief) and brief in prompt,
                "within_evidence_char_budget": len(brief) <= kwargs["evidence_max_total_chars"],
            })
    report = {
        "synthetic_probe_only": True, "answer_accuracy_evaluated": False,
        "renderer_path": rendering.__file__,
        "renderer_sha256": hashlib.sha256(Path(rendering.__file__).read_bytes()).hexdigest(),
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "renderer_path": report["renderer_path"],
        "synthetic_cases": len(rows),
        "complete_chain_visible": sum(row["complete_chain_visible"] for row in rows),
        "tail_visible": sum(row["tail_visible"] for row in rows),
        "all_within_budget": all(row["within_evidence_char_budget"] for row in rows),
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
