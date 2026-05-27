#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/llama_factory/action_target_hard_negative_kto_v1"
DEFAULT_PAIRWISE_PATHS = (
    PROJECT_ROOT / "data/processed/evidence_chain_reranker/batch_v1/reranker_pairwise.jsonl",
    PROJECT_ROOT / "data/processed/evidence_chain_reranker/batch_v2_answerability_promptfix1000/reranker_pairwise.jsonl",
    PROJECT_ROOT / "data/processed/evidence_chain_reranker/batch_v2_dpo_filtered/reranker_pairwise.jsonl",
)

SYSTEM_PROMPT = "你是《明日方舟》剧情问答系统的结构化输出模块。只输出指定 JSON。"
CONCLUSION_TASK = "conclusion_generation"
ACTION_WORDS = ("启动", "开启", "启用", "动用", "使用", "发动", "打开")
ACTION_MARKERS = (
    "因为",
    "为了",
    "目的",
    "目标",
    "作用",
    "代价",
    "性命",
    "生命",
    "牺牲",
    "解决",
    "危机",
    "阻止",
    "避免",
    "创造",
    "驱离",
    "阻挡",
    "撤出",
    "保护",
    "故名",
)
QUESTION_MARKERS = ("为什么", "为何", "原因", "目的", "目标", "作用", "用什么方法", "如何", "怎么")
BAD_TARGETS = {
    "之时",
    "时候",
    "方法",
    "办法",
    "这里",
    "这个",
    "这些",
    "那种",
    "正常",
    "更多",
    "证据",
    "问题",
    "计划",
}
BAD_TARGET_SUBSTRINGS = (
    "你",
    "我",
    "他",
    "她",
    "它",
    "您",
    "咱",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "这个",
    "那个",
    "这些",
    "那些",
    "这里",
    "那里",
    "什么",
    "哪里",
    "谁",
    "的",
    "了",
    "才能",
    "必须",
    "如果",
    "一旦",
    "是否",
    "错误",
    "只有",
    "因为",
    "为了",
    "为何",
    "为什么",
    "怎么",
    "如何",
    "目的",
    "原因",
    "作用",
    "意义",
)
TARGET_SUFFIX_RE = re.compile(
    r"(?:的)?(?:直接|真正|主要|最终)?(?:原因|目的|作用|意义|代价|后果|方法|机制|计划|影响|目标|是什么|为何|为什么|怎么|如何).*$"
)
SYSTEM_NOISE_RE = re.compile(r"\[(?:CHAIN_LEN|CAUSAL_ORDER|EVIDENCE_TYPES)=[^\]]*\]\s*|\[E\d+\]\s*")
SENTENCE_SPLIT_RE = re.compile(r"[\n\r。！？；]+")
QUOTED_ACTION_RE = re.compile(r"(?:启动|开启|启用|动用|使用|发动|打开)[“\"'「『]?([^”\"'」』，。！？；\s]{2,20})[”\"'」』]?")
PRE_ACTION_TARGET_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9_.·ⅣⅤⅠⅡⅢⅥⅦⅧⅨⅩ-]{2,24})(?:被|将|已|能|可|可以|会|的)?(?:启动|开启|启用|动用|使用|发动|打开)")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_meta(text: str) -> str:
    return normalize_space(SYSTEM_NOISE_RE.sub("", text or ""))


def truncate(text: str, max_chars: int) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def split_strips(text: str) -> list[str]:
    return [strip_meta(item) for item in SENTENCE_SPLIT_RE.split(strip_meta(text)) if strip_meta(item)]


def clean_target(target: str) -> str:
    target = re.sub(r"\s+", "", target or "").strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")
    target = re.sub(r"^(?:要|将|会|能|可|可以|全力|亲自|重新|正常)", "", target)
    target = TARGET_SUFFIX_RE.sub("", target)
    target = re.sub(r"(?:的时候|时|之时)$", "", target)
    return target.strip("“”\"'「」『』《》：:，,。！？?；;、（）()[]【】")


def is_valid_target(target: str) -> bool:
    if not target or target in BAD_TARGETS or len(target) < 2 or len(target) > 18:
        return False
    if any(token in target for token in BAD_TARGET_SUBSTRINGS):
        return False
    if re.search(r"[，。！？；：、\s]", target):
        return False
    if re.fullmatch(r"[0-9A-Za-z_.-]+", target):
        return False
    return True


def extract_action_targets(text: str, *, conservative: bool = False) -> list[str]:
    targets: list[str] = []
    for match in QUOTED_ACTION_RE.finditer(text or ""):
        targets.append(clean_target(match.group(1)))
    if not conservative:
        for match in PRE_ACTION_TARGET_RE.finditer(text or ""):
            raw = clean_target(match.group(1))
            # Keep the rightmost noun-like suffix when the regex over-captures a clause.
            raw = re.split(r"(?:，|。|；|：|但|若|如|和|与|以及|我们|他们|她们|它们)", raw)[-1]
            targets.append(clean_target(raw))
    output: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if not is_valid_target(target) or target in seen:
            continue
        seen.add(target)
        output.append(target)
    return output


def action_score(text: str, targets: list[str]) -> int:
    compact = re.sub(r"\s+", "", strip_meta(text))
    if not compact or not targets:
        return 0
    target_hit = any(target in compact for target in targets)
    if not target_hit:
        return 0
    action_hit = any(action in compact for action in ACTION_WORDS)
    marker_hits = sum(1 for marker in ACTION_MARKERS if marker in compact)
    return int(target_hit) + int(action_hit) + min(marker_hits, 2)


def choose_direct_strips(text: str, targets: list[str], *, max_strips: int, max_chars: int) -> list[str]:
    scored: list[tuple[int, str]] = []
    for strip in split_strips(text):
        score = action_score(strip, targets)
        if score >= 2:
            scored.append((score, truncate(strip, max_chars)))
    selected: list[str] = []
    for _, strip in sorted(scored, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return selected


def choose_background_strips(text: str, targets: list[str], *, max_strips: int, max_chars: int) -> list[str]:
    scored: list[tuple[int, str]] = []
    for strip in split_strips(text):
        compact = re.sub(r"\s+", "", strip)
        if any(target in compact for target in targets) and any(action in compact for action in ACTION_WORDS):
            continue
        marker_hits = sum(1 for marker in ACTION_MARKERS for _ in [0] if marker in compact)
        if marker_hits or any(token in compact for token in ("证明", "未来", "根基", "奉承", "碌碌无为", "不愿", "本应该")):
            scored.append((marker_hits, truncate(strip, max_chars)))
    selected: list[str] = []
    for _, strip in sorted(scored, key=lambda item: (item[0], len(item[1])), reverse=True):
        if strip and strip not in selected:
            selected.append(strip)
        if len(selected) >= max_strips:
            break
    return selected


def infer_query_type(question: str) -> str:
    if any(token in question for token in ("关系", "关联")):
        return "relation"
    if any(token in question for token in ("为什么", "为何", "原因", "目的")):
        return "causality"
    if any(token in question for token in ("怎么", "如何", "作用", "目标")):
        return "reasoning"
    return "fact"


def build_hypothesis(question: str, target: str) -> dict[str, Any]:
    entities = [target]
    for token in ("真龙", "不反", "航道计划", "第Ⅳ级武器", "源石", "岁陵"):
        if token in question and token not in entities:
            entities.insert(0, token)
    keywords = []
    for token in [*entities, *ACTION_WORDS, "直接原因", "目的", "代价", "危机", "直接证据", "背景动机"]:
        if token in question or token in entities or token in {"直接原因", "目的", "代价", "危机", "直接证据", "背景动机"}:
            keywords.append(token)
    return {
        "question": question,
        "intent": "plot_reasoning" if any(token in question for token in QUESTION_MARKERS) else "plot_fact",
        "query_type": infer_query_type(question),
        "entities": list(dict.fromkeys(entities))[:8],
        "keywords": list(dict.fromkeys(keywords))[:16],
        "expected_answer_type": "原因/目的/作用，要求区分直接证据与背景动机",
        "dialogue_context": "",
    }


def build_prompt(
    *,
    question: str,
    hypothesis: dict[str, Any],
    direct_strips: list[str],
    background_strips: list[str],
    max_rounds: int = 1,
) -> str:
    evidence = []
    for item in direct_strips:
        evidence.append(("direct", item))
    for item in background_strips:
        evidence.append(("background", item))
    lines = [
        f"task: {CONCLUSION_TASK}",
        f"question: {question}",
        f"hypothesis: {compact_json(hypothesis)}",
        f"round: 1/{max_rounds}",
        "evidence_brief:",
    ]
    for index, (kind, text) in enumerate(evidence, start=1):
        lines.append(f"{index}. [{kind}] {text}")
    lines.extend(
        [
            "decision_rule: 对“为什么/为何/目的/作用 + 启动/开启/启用/动用/使用 X”类问题，必须优先使用包含 X 和动作词的直接证据；背景动机只能作为补充，不能替代直接原因。",
            "output_schema: conclusion_v2",
        ]
    )
    return "\n".join(lines)


def build_chosen_answer(target: str, direct_strips: list[str], question: str = "") -> str:
    joined = " ".join(direct_strips)
    compact = re.sub(r"\s+", "", joined)
    question_compact = re.sub(r"\s+", "", question or "")
    reasons: list[str] = []
    if "证明" in question_compact:
        reasons.append("不是单纯为了证明自己的能力，直接证据首先指向解决当下岁陵里的危机")
    if any(token in question_compact for token in ("由真龙", "真龙启动", "性命", "代价")):
        reasons.append("因为全力启用源石要以真龙本人的性命为代价")
    if "关系" in question_compact and "岁陵" in question_compact:
        reasons.append("二者是直接因果关系：启动“不反”就是为了解决岁陵里的那场危机")
    if "区别" in question_compact:
        reasons.append("直接原因是处理岁陵危机，真龙对大炎和源石权柄的看法只能作为背景动机")
    if "解决" in compact and "危机" in compact:
        reasons.append("为了解决当前岁陵危机")
    if any(token in compact for token in ("创造", "航道", "澄澈水域", "联通")):
        reasons.append("为了达成证据中直接描述的目标或作用")
    if any(token in compact for token in ("驱离", "阻挡", "撤出", "保护", "阻止", "避免")):
        reasons.append("为了阻止或处理证据中直接描述的威胁")
    if any(token in compact for token in ("性命", "生命", "代价", "牺牲", "捐躯")):
        reasons.append("同时需要承担证据中明确写出的代价")
    if any(token in compact for token in ("源石", "威能", "权柄")):
        reasons.append("其能力来源或作用与证据中的源石/权柄设定直接相关")
    if not reasons:
        reasons.append("应以包含目标对象和动作词的直接证据为准")
    answer = f"现有证据显示，关于“{target}”的直接回答是：" + "；".join(dict.fromkeys(reasons)) + "。"
    answer += " 直接依据：" + "；".join(direct_strips[:3])
    return answer


def build_rejected_answer(target: str, background_strips: list[str]) -> str:
    background = "；".join(background_strips[:2]) if background_strips else "背景材料中的态度、历史或宏观动机"
    return (
        f"现有证据显示，使用或启动“{target}”的直接原因主要是背景动机：{background}。"
        "因此应把这些背景态度当作主要原因，而不必优先区分包含目标对象和动作词的直接证据。"
    )


def conclusion_payload(question: str, answer: str) -> dict[str, Any]:
    return {
        "question": question,
        "next_action": "answer_directly",
        "answer": answer,
        "missing_slots": [],
        "clarification_question": "",
        "follow_up_hypothesis": None,
    }


def make_record(
    *,
    record_id: str,
    prompt: str,
    payload: dict[str, Any],
    kto_tag: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": record_id,
        "task_type": CONCLUSION_TASK,
        "bucket": "action_target_hard_negative",
        "system": SYSTEM_PROMPT,
        "tools": "[]",
        "kto_tag": kto_tag,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": compact_json(payload)},
        ],
        "meta": meta,
    }


def manual_cases(max_chars: int) -> list[dict[str, Any]]:
    direct = [
        "太尉：陛下要以万金之躯启动“不反”，解决当下岁陵里的那场危机？",
        "莫佚：凡是想要全力启用源石的，都得以真龙本人的性命为代价......故名曰“不反”。",
        "真龙：谁言真龙执掌天下的权柄，竟要依托于这一枚，小小的结晶。",
    ]
    background = [
        "真龙：有这等神物，在炎国传了千年，也不曾有一位真龙开启过“不反”。如果在我手中，大炎沦至需要使用“不反”的地步......那我便是大炎第一号无能之君。",
        "真龙：掌握了这件奇物，大炎本应该做到更多。",
        "真龙最终还是没有践行岁的想法；他本想用又一场百氏之乱来撼动大炎的根基。",
    ]
    questions = [
        "真龙为什么要启动不反？",
        "真龙为什么要启动“不反”？",
        "真龙启动“不反”的直接目的是什么？",
        "真龙动用不反是为了解决什么？",
        "真龙开启不反的原因是什么？",
        "不反为什么需要由真龙启动？",
        "真龙启动不反是为了证明自己能力吗？",
        "真龙启动不反和岁陵危机有什么关系？",
        "真龙为什么要以自己的性命为代价启用不反？",
        "启动不反的直接原因和真龙对大炎的不满有什么区别？",
    ]
    return [
        {
            "source": "manual_act49side_bufan",
            "question": question,
            "target": "不反",
            "direct_strips": [truncate(item, max_chars) for item in direct],
            "background_strips": [truncate(item, max_chars) for item in background],
        }
        for question in questions
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_cases_from_pairwise(paths: list[Path], *, max_auto: int, max_strips: int, max_chars: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    skipped: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            skipped["missing_pairwise_path"] += 1
            continue
        for row in read_jsonl(path):
            query = str(row.get("query") or "").strip()
            if not query or not any(marker in query for marker in QUESTION_MARKERS):
                skipped["not_reason_question"] += 1
                continue
            if not any(action in query for action in ACTION_WORDS):
                skipped["question_without_action_word"] += 1
                continue
            positive = str(row.get("positive") or "")
            negative = str(row.get("negative") or "")
            # Auto samples are intentionally conservative: the target object must be explicit
            # in the user question, otherwise the extractor tends to learn clause fragments.
            targets = extract_action_targets(query, conservative=True)
            if not targets:
                skipped["no_clean_question_target"] += 1
                continue
            for target in targets:
                direct = choose_direct_strips(positive, [target], max_strips=max_strips, max_chars=max_chars)
                if not direct:
                    skipped["no_direct_strip"] += 1
                    continue
                background = choose_background_strips(negative + "\n" + positive, [target], max_strips=max_strips, max_chars=max_chars)
                if not background:
                    background = [truncate(strip_meta(negative), max_chars)] if negative.strip() else []
                key = (query, target)
                if key in seen:
                    skipped["duplicate_query_target"] += 1
                    continue
                seen.add(key)
                cases.append(
                    {
                        "source": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
                        "question": query,
                        "target": target,
                        "direct_strips": direct,
                        "background_strips": background[:max_strips],
                        "source_name": row.get("source_name"),
                        "query_type": row.get("query_type"),
                        "negative_type": row.get("negative_type"),
                    }
                )
                if len(cases) >= max_auto:
                    return cases, skipped
    return cases, skipped


def build_records(cases: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        question = str(case["question"])
        target = str(case["target"])
        direct_strips = list(case["direct_strips"])
        background_strips = list(case.get("background_strips") or [])
        hypothesis = build_hypothesis(question, target)
        prompt = build_prompt(
            question=question,
            hypothesis=hypothesis,
            direct_strips=direct_strips,
            background_strips=background_strips,
        )
        base_meta = {
            "source": case.get("source"),
            "source_name": case.get("source_name"),
            "query_type": case.get("query_type"),
            "target": target,
            "preference_focus": "direct_action_target_evidence_over_background_drift",
            "negative_type": case.get("negative_type") or "background_drift",
        }
        chosen_answer = build_chosen_answer(target, direct_strips, question)
        rejected_answer = build_rejected_answer(target, background_strips)
        records.append(
            make_record(
                record_id=f"action-target-{index:05d}-chosen",
                prompt=prompt,
                payload=conclusion_payload(question, chosen_answer),
                kto_tag=True,
                meta={**base_meta, "preference": "chosen"},
            )
        )
        records.append(
            make_record(
                record_id=f"action-target-{index:05d}-rejected",
                prompt=prompt,
                payload=conclusion_payload(question, rejected_answer),
                kto_tag=False,
                meta={**base_meta, "preference": "rejected"},
            )
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def split_records(records: list[dict[str, Any]], *, seed: int, val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_id = str(record["id"]).rsplit("-", 1)[0]
        by_case.setdefault(case_id, []).append(record)
    case_ids = list(by_case)
    random.Random(seed).shuffle(case_ids)
    val_case_count = max(1, int(round(len(case_ids) * val_ratio))) if len(case_ids) > 5 else 0
    val_ids = set(case_ids[:val_case_count])
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for case_id in case_ids:
        (val if case_id in val_ids else train).extend(by_case[case_id])
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(val)
    return train, val


def dataset_info(dataset_name: str) -> dict[str, Any]:
    tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "observation_tag": "observation",
        "function_tag": "function_call",
    }

    def entry(file_name: str) -> dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "kto_tag": "kto_tag",
            },
            "tags": tags,
        }

    return {
        f"{dataset_name}_train": entry("train.json"),
        f"{dataset_name}_val": entry("val.json"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KTO hard negatives for action-target conclusion drift.")
    parser.add_argument("--pairwise", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="action_target_hard_negative_kto_v1")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--max-auto", type=int, default=120)
    parser.add_argument("--max-strips", type=int, default=3)
    parser.add_argument("--max-evidence-chars", type=int, default=260)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--no-manual", action="store_true")
    parser.add_argument("--no-auto", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    pairwise_paths = [resolve_path(path) for path in (args.pairwise or list(DEFAULT_PAIRWISE_PATHS))]

    cases: list[dict[str, Any]] = []
    if not args.no_manual:
        cases.extend(manual_cases(args.max_evidence_chars))
    if args.no_auto or args.max_auto <= 0:
        auto_cases: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
    else:
        auto_cases, skipped = build_cases_from_pairwise(
            pairwise_paths,
            max_auto=args.max_auto,
            max_strips=args.max_strips,
            max_chars=args.max_evidence_chars,
        )
    cases.extend(auto_cases)
    records = build_records(cases, seed=args.seed)
    train, val = split_records(records, seed=args.seed, val_ratio=args.val_ratio)

    write_json(output_dir / "train.json", train)
    write_json(output_dir / "val.json", val)
    write_json(output_dir / "dataset_info.json", dataset_info(args.dataset_name))
    stats = {
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "cases": len(cases),
        "manual_cases": 0 if args.no_manual else len(manual_cases(args.max_evidence_chars)),
        "auto_cases": len(auto_cases),
        "records": len(records),
        "train": len(train),
        "val": len(val),
        "kto_tags": dict(Counter(str(record.get("kto_tag")) for record in records)),
        "targets": dict(Counter(str(case.get("target")) for case in cases).most_common(30)),
        "sources": dict(Counter(str(case.get("source")) for case in cases).most_common(20)),
        "auto_skipped": dict(skipped.most_common()),
    }
    write_json(output_dir / "build_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
