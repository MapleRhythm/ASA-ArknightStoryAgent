#!/usr/bin/env python3
"""Exploratory bundled prompt baseline; no model/production/training changes."""
from __future__ import annotations

from build_inputs import ROOT, load, sha, write

APPENDIX = (
    "\n研究对照补充规则：先判断问题问的是谁的说法或看法，还是事件本身。"
    "清楚的转述可以支持所报告的观点归属，但不自动证明被转述内容为客观事实。"
    "明确的否定可以回答是否类问题，不能把否定当作缺证。"
    "只改变轮次不会改变证据本身是否充分；末轮若证据充分仍应回答。"
    "如果必要信息确实缺失，必须用当前轮次对应的retrieve_more或abstain，"
    "不要把‘无法确定’或问题句放入supported_facts充当答案。"
)


def main():
    selected = [r for r in load(ROOT / "input/eval_matrix.json")
                if r["diagnostic"]["suite"] in {"scope_controlled", "natural_controls"}]
    for row in selected:
        diag = row["diagnostic"]
        diag["original_prompt_messages_sha256"] = diag["prompt_messages_sha256"]
        row["conversations"][0]["value"] += APPENDIX
        diag["prompt_messages_sha256"] = sha([
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["conversations"][0]["value"]},
        ])
        diag["exploratory_prompt_control"] = {
            "only_change": "bundled_generic_rule_appendix",
            "designed_after_reading_initial_scope_outputs": True,
            "not_blind_validation": True,
            "goal": "cheap prompt baseline before any new training",
        }
    write(ROOT / "input/prompt_control_matrix.json", selected)
    write(ROOT / "input/prompt_control_manifest.json", {
        "rows": len(selected), "appendix": APPENDIX, "appendix_sha256": sha(APPENDIX),
        "status": "exploratory, not preregistered blind evaluation",
        "scope": "Same nonthinking Exx interface, additional generic rules; no diagnostic labels in model input",
        "matched_compute_caveat": "Input tokens increase; compare output/latency and do not attribute changes to one sentence.",
    })
    print(len(selected))


if __name__ == "__main__":
    main()
