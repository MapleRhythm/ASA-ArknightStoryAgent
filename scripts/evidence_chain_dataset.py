#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from goldenglow.config import EXCEL_ROOT, STORY_ROOT
from goldenglow.data.story_parser import (
    _build_story_speaker_lookup,
    parse_story_file,
    render_segment,
)


QUERY_TYPES = {
    "fact",
    "relation",
    "causality",
    "reasoning",
    "reveal",
    "mystery",
    "answerability",
}
CANDIDATE_TYPES = {
    "gold",
    "shuffled_order",
    "irrelevant_mixed",
    "incomplete",
    "partially_relevant",
    "unrelated",
    "background_only",
    "answer_adjacent",
    "same_entity_distractor",
    "partial_answer",
    "misleading_chain",
}
DEFAULT_NEGATIVE_SCORES = {
    "shuffled_order": 0.6,
    "irrelevant_mixed": 0.3,
    "incomplete": 0.4,
    "background_only": 0.25,
    "answer_adjacent": 0.45,
    "same_entity_distractor": 0.35,
    "partial_answer": 0.55,
    "misleading_chain": 0.25,
}
PROMPT_TEMPLATE = """你是《明日方舟》剧情 answerability evidence-chain reranker 训练数据标注器。
你必须只输出合法 JSON，不输出任何解释、开场白、markdown 或注释。

输入是一组同一剧情单元内的完整原文，可能包含多个关卡文件。原文包含对话、旁白、场景描述、系统文本。

目标：生成用于训练 evidence-chain reranker 的数据。数据必须帮助模型判断“哪条证据链最能充分、直接、完整回答用户问题”，而不是只判断文本相似度或是否同角色同章节。

核心定义：
- answer-bearing evidence：能直接支撑答案结论的原文片段，例如某人亲口说明原因、计划、身份、真相、关系，或旁白明确揭示结果。
- setup evidence：理解 answer-bearing evidence 必需的上下文，例如地点、身份、行动前提、被问对象。
- background-only evidence：包含同一角色、同一地点、同一事件背景，但没有答案揭示点，不能单独回答问题。
- gold evidence chain：必须包含至少 1 条 answer-bearing evidence，并按原文时序补足必要 setup evidence。
- 训练目标是 answer sufficiency：证据链是否足以回答问题，而不是“看起来相关”。

执行规则：
1. 先把全文拆分为细粒度证据片段，编号为 E1、E2、E3……
2. 每个证据片段必须来自原文，禁止改写、总结、补写。
3. 每个证据片段应语义独立，长度建议 30-180 字；过长对话必须拆分。
4. 每个证据片段需要尽量保留说话人，例如“凯尔希：……”。
5. all_evidence 不能只抽少量摘要片段：
   - 原文少于 20000 字时，all_evidence 建议至少 18 条。
   - 原文 20000-70000 字时，all_evidence 建议至少 30 条。
   - 原文超过 70000 字时，all_evidence 建议至少 40 条，并覆盖开端、中段、结尾的关键揭示场景。
   - all_evidence 必须包含每条 gold chain、background_only、answer_adjacent、partial_answer 等候选链引用到的所有 evidence id。
6. 生成 {min_questions}-{max_questions} 条高质量用户问题，覆盖：
   - fact：事实细节
   - relation：人物关系
   - causality：事件前因后果
   - reasoning：剧情发展推理
   - reveal / mystery：真相、阴谋、幕后计划、身份揭示、某人识破了什么
   其中 reveal/mystery/causality/reasoning 合计应不少于总数的 60%。优先生成“为什么 / 真相是什么 / 阴谋是什么 / 谁主使 / 某人发现了什么 / 某人为何这么做”类问题。
7. 每条问题必须能由 gold 证据链完整回答，不能依赖常识或原文外设定。
8. gold 证据链必须按剧情原生时序排列，长度按问题类型控制：
   - fact / relation：2-4 个证据片段。
   - causality / reasoning / reveal / mystery / answerability：3-8 个证据片段。
   - 若完整回答确实需要更长上下文，最多 10 个证据片段。
   - 对 causality / reasoning / reveal / mystery，如果答案看似只需 1-2 个片段，也必须补入必要 setup 或 outcome，使 gold chain 达到至少 3 个证据片段。
   - 不要为了凑长度加入无关背景；新增片段必须在 answer_focus 中说明作用。
   若某个问题完全无法补出 2 个以上必要片段，不要生成该问题。
9. 每条问题必须写出 answer、answer_evidence 和 answer_focus：
   - answer：用 1-2 句概括 gold chain 支持的答案，不要加入原文外信息。
   - answer_evidence：gold chain 中真正承载答案结论的 evidence id，至少 1 个。
   - answer_focus：一句话说明 gold 为什么能回答问题，例如“E4 直接说明动机，E2 提供对象背景”。
   - answer 必须忠于 evidence：如果证据只支持“疑似/暗示/有人认为”，answer 中也必须保留不确定性，禁止把推测写成确定事实。
10. 每条问题必须生成 6 条 candidate chain：
   - gold：完整、时序正确、能直接回答问题，score 固定为 1.0。
   - background_only：同角色/同地点/同事件，词面高度相关，但没有 answer_evidence，无法回答核心问题；score 取 0.05-0.30。
     background_only 应允许较长链，长度 4-8 个证据片段，用来训练模型识别“很长但没答案”的背景链。
   - answer_adjacent：紧邻答案前后、看起来像关键情节，但缺少真正 answer-bearing evidence；score 取 0.25-0.50。
   - incomplete 或 partial_answer：包含部分 setup 或部分答案，但缺少关键结论；score 取 0.35-0.60。
   - same_entity_distractor：包含同一角色或同一专名，主题相近但回答的是别的问题；score 取 0.15-0.40。
   - shuffled_order 或 misleading_chain：使用相关片段但顺序破坏因果，或混入会误导答案的片段；score 取 0.20-0.60。
11. 负样本必须是 hard negatives：
   - 至少 3 条负样本要来自同一 SOURCE_FILE、同一 story/stage 或相邻剧情文件。
   - 至少 3 条负样本必须包含 query 中的核心角色、地点、组织或事件词。
   - background_only / answer_adjacent 负样本优先选择“前情铺垫、人物档案、行动背景、旁枝对话”，用来训练模型不要把背景误判为答案。
   - 任何负样本都不能包含所有 answer_evidence；如果负样本包含了 answer_evidence，必须缺失必要 setup，且 score 不能超过 0.60。
   - gold 和负样本都要避免把真正 answer-bearing evidence 放到过长链的末尾；answer_evidence 应尽量出现在证据链前 70% 的文本位置，减少训练和推理截断损失。
12. candidate_chain_list 中的负样本不能与 gold 完全相同。
13. score 表示该证据链回答 query 的充分性，而不是文本相似度。
14. 如果一个问题的 gold 只是背景归纳、没有明确 answer-bearing evidence，不要生成该问题。
15. 如果原文不足以支持 {min_questions} 条高质量问题，则生成能被原文严格支持的最大数量，不要编造。
16. JSON 合法性硬要求：
   - 所有 chain / answer_evidence 中的 id 必须已经在 all_evidence 中定义。
   - 禁止输出 trailing comma，例如最后一个字段或最后一个数组元素后不能有逗号。
   - 禁止输出注释、markdown、额外解释文本。
17. 额外输出 `entity_relations` 三元组数组，作为后续 MiniRAG 异构图索引的原料：
   - 从 all_evidence 中抽取关键 (head, relation, tail) 三元组，每个三元组必须能在某条 evidence 中找到字面支撑。
   - relation 优先使用以下中文短语之一：所属、隶属、师从、师徒、亲属、上下级、敌对、合作、暗杀、保护、决定、动机、原因、结果、出现于、发生在、提及、关联。
   - head / tail 必须是人物名、组织名、地点名、事件名、关键道具名；禁止使用代词、问句残片、宽泛词（如"某人"、"某地"、"事情"）。
   - 每个三元组必须标注 `evidence_id`（来自 all_evidence），用于回溯证据。
   - 每段原文至少抽出 6 个三元组；如果原文极短可降到 3 个。
   - 三元组允许覆盖背景信息（不一定与 rerank query 直接相关），目的是为图谱积累节点与边。
18. 每个候选链（含 gold 与 negative）额外标注 `chain_role_tags` 数组：从 ["motive","action","outcome","context","bridge"] 中挑出该链覆盖的角色标签，用于 reranker 学习链结构。

输出格式必须严格为：
{{
  "all_evidence": [
    {{
      "id": "E1",
      "text": "证据片段内容"
    }}
  ],
  "entity_relations": [
    {{
      "head": "凯尔希",
      "relation": "所属",
      "tail": "罗德岛",
      "evidence_id": "E1"
    }}
  ],
  "rerank_dataset": [
    {{
      "query": "用户问题",
      "query_type": "fact | relation | causality | reasoning | reveal | mystery | answerability",
      "answer": "由 gold chain 支持的简短答案",
      "answer_evidence": ["E2"],
      "answer_focus": "E2 直接揭示答案，E1 提供必要上下文。",
      "candidate_chain_list": [
        {{
          "label": "positive",
          "type": "gold",
          "chain": ["E1", "E2"],
          "chain_role_tags": ["motive","outcome"],
          "score": 1.0,
          "score_reason": "证据链按原文时序完整回答问题。"
        }},
        {{
          "label": "negative",
          "type": "shuffled_order",
          "chain": ["E2", "E1"],
          "chain_role_tags": ["motive","outcome"],
          "score": 0.6,
          "score_reason": "证据相关但顺序破坏因果或时间线。"
        }},
        {{
          "label": "negative",
          "type": "background_only",
          "chain": ["E1", "E7", "E9", "E11"],
          "chain_role_tags": ["context"],
          "score": 0.2,
          "score_reason": "词面相关且背景很长，但没有答案揭示点。"
        }},
        {{
          "label": "negative",
          "type": "answer_adjacent",
          "chain": ["E3", "E5"],
          "chain_role_tags": ["bridge"],
          "score": 0.4,
          "score_reason": "临近关键剧情但缺少真正承载答案的片段。"
        }},
        {{
          "label": "negative",
          "type": "partial_answer",
          "chain": ["E1"],
          "chain_role_tags": ["motive"],
          "score": 0.4,
          "score_reason": "证据相关但缺少关键环节。"
        }},
        {{
          "label": "negative",
          "type": "same_entity_distractor",
          "chain": ["E8", "E10"],
          "chain_role_tags": ["context"],
          "score": 0.25,
          "score_reason": "包含同一角色但回答的是另一件事。"
        }}
      ]
    }}
  ]
}}

质量检查：
- 所有 evidence id 必须存在于 all_evidence。
- gold 证据链必须包含 answer_evidence，并能完整回答 query。
- gold 证据链必须按原文顺序排列。
- 每条 query 必须有且只有一个 gold candidate。
- gold chain 长度必须符合题型：fact/relation 为 2-4；causality/reasoning/reveal/mystery/answerability 为 3-8，复杂问题最多 10；不要生成单证据问题。
- 每条 query 必须至少有 5 个 negative candidate。
- 每条 query 必须至少包含 1 个 background_only 或 answer_adjacent 负样本。
- background_only 负样本建议 4-8 个证据片段，不能总是 2-3 段短背景。
- negative chain 不能与 gold chain 完全相同。
- 负样本必须是较难负例，不能全是完全无关随机片段；要特别构造“同角色同剧情但不能回答”的背景负例。
- score_reason 必须说明“为什么不能完整回答”，不能只写“无关”。
- 所有被引用的 evidence id 都必须存在于 all_evidence，尤其不能出现 E43 这类未定义 id。
- 输出 JSON 不能有 trailing comma。
- 输出必须是单个 JSON 对象。
- `entity_relations` 必须为非空数组，每条三元组的 `evidence_id` 必须出现在 all_evidence。
- 每个 candidate 必须带有 `chain_role_tags`，且只能从 ["motive","action","outcome","context","bridge"] 中取值。

下面是完整剧情文本：
{source_text}
"""


@dataclass(slots=True)
class ValidationIssue:
    level: str
    location: str
    message: str


def natural_key(path: Path) -> list[Any]:
    parts: list[Any] = []
    for part in re.split(r"(\d+)", path.as_posix()):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part)
    return parts


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def load_json_payload(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8").strip()
    return parse_json_text(raw)


def parse_json_text(raw: str) -> Any:
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = repair_json_text(raw)
        if repaired != raw:
            return json.loads(repaired)
        raise


def repair_json_text(raw: str) -> str:
    repaired = raw.strip()
    start = repaired.find("{")
    end = repaired.rfind("}")
    if start >= 0 and end > start:
        repaired = repaired[start : end + 1]

    # Common JSON-ish artifact: trailing commas before object/array endings.
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    # Common model artifact: an object field accidentally gets an extra
    # closing brace before the real object terminator on the next line.
    repaired = re.sub(
        r'("score_reason"\s*:\s*"(?:\\.|[^"\\])*")\s*}\s*(\n\s*},)',
        r"\1\2",
        repaired,
    )
    return repaired


def join_api_url(api_base: str, endpoint_path: str) -> str:
    return f"{api_base.rstrip('/')}/{endpoint_path.lstrip('/')}"


def resolve_story_files(args: argparse.Namespace) -> list[Path]:
    files: list[Path] = []
    for item in args.story_files or []:
        files.append(Path(item))
    if args.story_dir:
        story_dir = Path(args.story_dir)
        files.extend(sorted(story_dir.glob(args.glob), key=natural_key))
    resolved: list[Path] = []
    for path in files:
        candidate = path if path.is_absolute() else PROJECT_ROOT / path
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if candidate.is_dir():
            raise IsADirectoryError(candidate)
        resolved.append(candidate)
    return list(dict.fromkeys(resolved))


def render_story_source(files: list[Path]) -> str:
    speaker_lookup = _build_story_speaker_lookup(EXCEL_ROOT)
    sections: list[str] = []
    for path in files:
        try:
            relative_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_path = path
        segments = parse_story_file(path, speaker_lookup=speaker_lookup)
        if not segments:
            continue
        lines = [f"### SOURCE_FILE: {relative_path.as_posix()}"]
        for index, segment in enumerate(segments, start=1):
            lines.append(f"[{index}] {render_segment(segment)}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections).strip()


def command_make_prompts(args: argparse.Namespace) -> int:
    story_files = resolve_story_files(args)
    if not story_files:
        raise RuntimeError("No story files matched the input.")

    source_text = render_story_source(story_files)
    if not source_text:
        raise RuntimeError("No parseable story text was produced.")
    if args.max_source_chars and len(source_text) > args.max_source_chars:
        raise RuntimeError(
            f"Rendered source text has {len(source_text)} chars, exceeding --max-source-chars={args.max_source_chars}."
        )

    prompt_id = args.prompt_id or story_files[0].stem
    prompt = PROMPT_TEMPLATE.format(
        min_questions=args.min_questions,
        max_questions=args.max_questions,
        source_text=source_text,
    )
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.as_posix()
        for path in story_files
    ]
    record = {
        "prompt_id": prompt_id,
        "source_files": source_files,
        "source_chars": len(source_text),
        "prompt": prompt,
    }
    (output_dir / f"{prompt_id}.prompt.txt").write_text(prompt, encoding="utf-8")
    write_jsonl(output_dir / "prompts.jsonl", [record])
    manifest = {
        "prompt_id": prompt_id,
        "source_files": source_files,
        "source_chars": len(source_text),
        "prompt_file": f"{prompt_id}.prompt.txt",
        "prompt_jsonl": "prompts.jsonl",
        "min_questions": args.min_questions,
        "max_questions": args.max_questions,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def extract_chat_response_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            text_parts.append(item["text"])
                    if text_parts:
                        return "\n".join(text_parts)
            text = first_choice.get("text")
            if isinstance(text, str):
                return text

    for key in ("output_text", "text", "content"):
        value = response_payload.get(key)
        if isinstance(value, str):
            return value
    raise ValueError("Could not extract text content from API response.")


def command_call_api(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Set it with: export {args.api_key_env}=...")

    model = args.model or os.environ.get(args.model_env, "").strip()
    if not model:
        raise RuntimeError(f"Missing model name. Pass --model or set {args.model_env}.")

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = PROJECT_ROOT / prompt_path
    prompt = prompt_path.read_text(encoding="utf-8")
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.response_format_json:
        request_payload["response_format"] = {"type": "json_object"}

    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    api_url = join_api_url(args.api_base, args.endpoint_path)
    if not args.quiet:
        print(
            f"[call-api] prompt={prompt_path} chars={len(prompt)} bytes={len(prompt.encode('utf-8'))}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[call-api] endpoint={api_url} model={model} max_tokens={args.max_tokens} timeout={args.timeout}s",
            file=sys.stderr,
            flush=True,
        )
        print("[call-api] sending request...", file=sys.stderr, flush=True)
    http_request = request.Request(
        api_url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    started_at = time.monotonic()
    try:
        with request.urlopen(http_request, timeout=args.timeout) as response:
            if not args.quiet:
                elapsed = time.monotonic() - started_at
                print(
                    f"[call-api] response headers received status={response.status} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
            raw_response = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {error_body}") from exc
    if not args.quiet:
        elapsed = time.monotonic() - started_at
        print(
            f"[call-api] response body read chars={len(raw_response)} elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    raw_output_path = output_path.with_suffix(output_path.suffix + ".raw.json")
    response_payload = json.loads(raw_response)
    raw_output_path.write_text(
        json.dumps(response_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    content = extract_chat_response_content(response_payload)
    try:
        parsed_content = parse_json_text(content.strip())
    except json.JSONDecodeError:
        text_output_path = output_path.with_suffix(output_path.suffix + ".response.txt")
        text_output_path.write_text(content, encoding="utf-8")
        raise RuntimeError(
            f"Model response was not valid JSON. Raw text saved to {text_output_path}; raw API response saved to {raw_output_path}."
        )

    output_path.write_text(
        json.dumps(parsed_content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"[call-api] parsed JSON saved to {output_path}", file=sys.stderr, flush=True)
        print(f"[call-api] raw API response saved to {raw_output_path}", file=sys.stderr, flush=True)
    summary = {
        "prompt_file": str(prompt_path),
        "api_base": args.api_base,
        "endpoint_path": args.endpoint_path,
        "model": model,
        "output_json": str(output_path),
        "raw_response_json": str(raw_output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def normalize_chain(raw_chain: Any) -> list[str]:
    if not isinstance(raw_chain, list):
        return []
    return [normalize_evidence_id(item) for item in raw_chain if normalize_evidence_id(item)]


def normalize_evidence_id(raw_id: Any) -> str:
    evidence_id = str(raw_id).strip()
    if re.fullmatch(r"\d+", evidence_id):
        return f"E{evidence_id}"
    return evidence_id


def normalize_evidence_ids(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    return [normalize_evidence_id(item) for item in raw_ids if normalize_evidence_id(item)]


def chain_text(chain: list[str], evidence_by_id: dict[str, str]) -> str:
    return "\n".join(f"[{evidence_id}] {evidence_by_id[evidence_id]}" for evidence_id in chain if evidence_id in evidence_by_id)


def evidence_order(chain: list[str]) -> list[int]:
    order: list[int] = []
    for evidence_id in chain:
        match = re.fullmatch(r"E(\d+)", evidence_id)
        if match:
            order.append(int(match.group(1)))
    return order


def evidence_sort_value(evidence_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"E(\d+)", evidence_id)
    if match:
        return int(match.group(1)), evidence_id
    return 10**9, evidence_id


# ----------------------------------------------------------------------
# Chain structure metadata helpers
# ----------------------------------------------------------------------
MOTIVE_MARKERS = frozenset({"因为", "动机", "目的", "为了", "原因在于", "出于", "所以才", "为何"})
OUTCOME_MARKERS = frozenset({"于是", "于是乎", "结果", "后来", "最后", "造成了", "导致", "因此", "最终", "果然"})
ACTION_MARKERS = frozenset({"决定", "试图", "命令", "同意", "拒绝", "计划", "策划", "企图", "着手", "开始", "动手", "执行"})


def _classify_evidence_type(text: str) -> str:
    """Classify a single evidence piece into one of: outcome, motive, action, context."""
    text = str(text)
    if any(m in text for m in OUTCOME_MARKERS):
        return "outcome"
    if any(m in text for m in MOTIVE_MARKERS):
        return "motive"
    if any(m in text for m in ACTION_MARKERS):
        return "action"
    return "context"


def _extract_chain_structure(
    chain: list[str],
    evidence_by_id: dict[str, str],
) -> dict[str, Any]:
    """Analyze a chain of evidence IDs and produce structure metadata."""
    types = [_classify_evidence_type(evidence_by_id.get(eid, "")) for eid in chain]
    unique_types = list(dict.fromkeys(t for t in types if t != "context"))
    has_action = any(t in ("motive", "action") for t in types)
    has_outcome = "outcome" in types
    has_context = "context" in types

    if has_action and has_outcome:
        motive_pos = types.index("motive") if "motive" in types else types.index("action")
        outcome_pos = types.index("outcome")
        causal_order = "motive_before_outcome" if motive_pos < outcome_pos else "outcome_before_motive"
    elif has_action:
        causal_order = "action_only"
    elif has_outcome:
        causal_order = "outcome_only"
    elif has_context:
        causal_order = "context_only"
    else:
        causal_order = "unknown"

    return {
        "chain_length": len(chain),
        "evidence_types": types,
        "unique_types": unique_types,
        "has_action": has_action,
        "has_outcome": has_outcome,
        "has_context": has_context,
        "causal_order": causal_order,
    }


def _build_chain_text_with_metadata(
    chain: list[str],
    evidence_by_id: dict[str, str],
    chain_structure: dict[str, Any],
) -> str:
    """Build reranker input text with chain structure metadata prefix."""
    meta_parts = [
        f"[CHAIN_LEN={chain_structure['chain_length']}]",
        f"[CAUSAL_ORDER={chain_structure['causal_order']}]",
        f"[EVIDENCE_TYPES=({'|'.join(chain_structure['evidence_types'])})]",
    ]
    meta_header = " ".join(meta_parts)
    body = "\n".join(
        f"[{eid}] {evidence_by_id.get(eid, '')}" for eid in chain if eid in evidence_by_id
    )
    return f"{meta_header}\n{body}"


def _build_candidate_record(
    candidate: dict[str, Any],
    evidence_by_id: dict[str, str],
) -> dict[str, Any]:
    """Build a candidate record with chain structure metadata for reranker training."""
    chain = candidate.get("chain", [])
    structure = _extract_chain_structure(chain, evidence_by_id)
    return {
        "text": candidate["chain_text"],
        "text_with_metadata": _build_chain_text_with_metadata(chain, evidence_by_id, structure),
        "score": candidate["score"],
        "label": candidate["label"],
        "type": candidate["type"],
        "chain": chain,
        "chain_role_tags": candidate.get("chain_role_tags", []),
        "chain_structure": structure,
    }


ALLOWED_CHAIN_ROLE_TAGS = frozenset({"motive", "action", "outcome", "context", "bridge"})


def _normalize_chain_role_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip().lower()
        if token in ALLOWED_CHAIN_ROLE_TAGS and token not in seen:
            cleaned.append(token)
            seen.add(token)
    return cleaned


def normalize_candidate_list(sample: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = sample.get("candidate_chain_list")
    if isinstance(candidates, list):
        normalized: list[dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_type = str(candidate.get("type") or "").strip()
            label = str(candidate.get("label") or "").strip()
            if candidate_type == "gold":
                label = "positive"
            elif not label:
                label = "negative"
            normalized.append(
                {
                    "label": label,
                    "type": candidate_type or ("gold" if label == "positive" else "unrelated"),
                    "chain": normalize_chain(candidate.get("chain")),
                    "chain_role_tags": _normalize_chain_role_tags(candidate.get("chain_role_tags")),
                    "score": candidate.get("score"),
                    "score_reason": str(candidate.get("score_reason") or "").strip(),
                }
            )
        return normalized

    normalized = []
    positive_chain = normalize_chain(sample.get("positive_chain"))
    if positive_chain:
        normalized.append(
            {
                "label": "positive",
                "type": "gold",
                "chain": positive_chain,
                "chain_role_tags": _normalize_chain_role_tags(sample.get("positive_chain_role_tags")),
                "score": 1.0,
                "score_reason": "Legacy positive_chain converted to gold candidate.",
            }
        )

    negative_chains = sample.get("negative_chain_list") or []
    if isinstance(negative_chains, list):
        default_types = ["shuffled_order", "irrelevant_mixed", "incomplete"]
        for index, raw_negative in enumerate(negative_chains):
            if isinstance(raw_negative, dict):
                negative_type = str(raw_negative.get("type") or default_types[min(index, 2)]).strip()
                chain = normalize_chain(raw_negative.get("chain"))
                score = raw_negative.get("score", DEFAULT_NEGATIVE_SCORES.get(negative_type, 0.25))
                reason = str(raw_negative.get("score_reason") or "").strip()
                role_tags = _normalize_chain_role_tags(raw_negative.get("chain_role_tags"))
            else:
                negative_type = default_types[min(index, 2)]
                chain = normalize_chain(raw_negative)
                score = DEFAULT_NEGATIVE_SCORES.get(negative_type, 0.25)
                reason = "Legacy negative_chain_list converted to scored negative candidate."
                role_tags = []
            normalized.append(
                {
                    "label": "negative",
                    "type": negative_type,
                    "chain": chain,
                    "chain_role_tags": role_tags,
                    "score": score,
                    "score_reason": reason,
                }
            )
    return normalized


def validate_and_normalize_payload(payload: Any, *, source_name: str) -> tuple[dict[str, Any], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    if not isinstance(payload, dict):
        raise ValueError(f"{source_name}: top-level payload must be an object.")

    evidence_items = payload.get("all_evidence")
    if not isinstance(evidence_items, list):
        raise ValueError(f"{source_name}: all_evidence must be a list.")

    evidence_by_id: dict[str, str] = {}
    normalized_evidence: list[dict[str, str]] = []
    for index, item in enumerate(evidence_items):
        location = f"{source_name}.all_evidence[{index}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue("error", location, "evidence item must be an object"))
            continue
        evidence_id = str(item.get("id") or "").strip()
        evidence_id = normalize_evidence_id(evidence_id)
        text = str(item.get("text") or "").strip()
        if not re.fullmatch(r"E\d+", evidence_id):
            issues.append(ValidationIssue("error", location, f"invalid evidence id: {evidence_id!r}"))
            continue
        if evidence_id in evidence_by_id:
            issues.append(ValidationIssue("error", location, f"duplicate evidence id: {evidence_id}"))
            continue
        if not text:
            issues.append(ValidationIssue("error", location, "evidence text is empty"))
            continue
        evidence_by_id[evidence_id] = text
        normalized_evidence.append({"id": evidence_id, "text": text})

    dataset = payload.get("rerank_dataset")
    if not isinstance(dataset, list):
        raise ValueError(f"{source_name}: rerank_dataset must be a list.")

    normalized_dataset: list[dict[str, Any]] = []
    for sample_index, raw_sample in enumerate(dataset):
        sample_location = f"{source_name}.rerank_dataset[{sample_index}]"
        if not isinstance(raw_sample, dict):
            issues.append(ValidationIssue("error", sample_location, "sample must be an object"))
            continue
        query = str(raw_sample.get("query") or "").strip()
        if not query:
            issues.append(ValidationIssue("error", sample_location, "query is empty"))
            continue
        query_type = str(raw_sample.get("query_type") or "reasoning").strip()
        if query_type not in QUERY_TYPES:
            issues.append(ValidationIssue("warning", sample_location, f"unknown query_type={query_type!r}; coerced to reasoning"))
            query_type = "reasoning"
        answer = str(raw_sample.get("answer") or "").strip()
        answer_evidence = normalize_evidence_ids(raw_sample.get("answer_evidence"))
        answer_focus = str(raw_sample.get("answer_focus") or "").strip()
        if not answer:
            issues.append(ValidationIssue("warning", sample_location, "answer is missing; answerability supervision will be weaker"))
        if not answer_evidence:
            issues.append(ValidationIssue("warning", sample_location, "answer_evidence is missing; cannot verify answer-bearing evidence"))
        else:
            missing_answer_ids = [evidence_id for evidence_id in answer_evidence if evidence_id not in evidence_by_id]
            if missing_answer_ids:
                issues.append(ValidationIssue("error", sample_location, f"answer_evidence contains unknown ids: {missing_answer_ids}"))
                continue

        candidates = normalize_candidate_list(raw_sample)
        if not candidates:
            issues.append(ValidationIssue("error", sample_location, "no candidate_chain_list candidates"))
            continue

        normalized_candidates: list[dict[str, Any]] = []
        gold_count = 0
        negative_count = 0
        has_background_negative = False
        seen_chains: set[tuple[str, ...]] = set()
        positive_chain_tuple: tuple[str, ...] | None = None
        invalid_gold = False
        for candidate_index, candidate in enumerate(candidates):
            candidate_location = f"{sample_location}.candidate_chain_list[{candidate_index}]"
            label = str(candidate.get("label") or "").strip()
            candidate_type = str(candidate.get("type") or "").strip()
            chain = normalize_chain(candidate.get("chain"))
            if label not in {"positive", "negative"}:
                issues.append(ValidationIssue("error", candidate_location, f"invalid label={label!r}"))
                continue
            if candidate_type not in CANDIDATE_TYPES:
                issues.append(ValidationIssue("warning", candidate_location, f"unknown type={candidate_type!r}; coerced to unrelated"))
                candidate_type = "unrelated"
            if not chain:
                issues.append(ValidationIssue("error", candidate_location, "chain is empty"))
                continue
            missing_ids = [evidence_id for evidence_id in chain if evidence_id not in evidence_by_id]
            if missing_ids:
                issues.append(ValidationIssue("error", candidate_location, f"unknown evidence ids: {missing_ids}"))
                continue
            try:
                score = float(candidate.get("score"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", candidate_location, f"invalid score={candidate.get('score')!r}"))
                continue
            if not 0.0 <= score <= 1.0:
                issues.append(ValidationIssue("error", candidate_location, f"score out of range: {score}"))
                continue
            chain_role_tags = _normalize_chain_role_tags(candidate.get("chain_role_tags"))
            if not chain_role_tags:
                issues.append(ValidationIssue("warning", candidate_location, "missing chain_role_tags"))
            is_gold_candidate = label == "positive" or candidate_type == "gold"
            if is_gold_candidate:
                sorted_chain = sorted(chain, key=evidence_sort_value)
                if sorted_chain != chain:
                    issues.append(ValidationIssue("warning", candidate_location, "gold chain was reordered into ascending evidence order"))
                    chain = sorted_chain
            chain_tuple = tuple(chain)
            if chain_tuple in seen_chains:
                issues.append(ValidationIssue("warning", candidate_location, "duplicate chain within sample"))
            seen_chains.add(chain_tuple)
            if is_gold_candidate:
                gold_count += 1
                positive_chain_tuple = chain_tuple
                if candidate_type != "gold":
                    issues.append(ValidationIssue("warning", candidate_location, "positive candidate type should be gold"))
                if score < 0.95:
                    issues.append(ValidationIssue("warning", candidate_location, "gold candidate score should be close to 1.0"))
                min_gold_len = 3 if query_type in {"causality", "reasoning", "reveal", "mystery", "answerability"} else 2
                max_gold_len = 10 if query_type in {"causality", "reasoning", "reveal", "mystery", "answerability"} else 6
                if len(chain) < min_gold_len:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            f"gold chain has fewer than {min_gold_len} evidence items for query_type={query_type}; kept because answer_evidence is present",
                        )
                    )
                if len(chain) > max_gold_len:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            f"gold chain has more than {max_gold_len} evidence items for query_type={query_type}",
                        )
                    )
                if answer_evidence and not set(answer_evidence).issubset(set(chain)):
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            "gold chain does not include all answer_evidence ids; sample will be skipped",
                        )
                    )
                    invalid_gold = True
                order = evidence_order(chain)
                if order and order != sorted(order):
                    issues.append(ValidationIssue("warning", candidate_location, "gold chain is not in ascending evidence order"))
            elif positive_chain_tuple and chain_tuple == positive_chain_tuple:
                issues.append(ValidationIssue("warning", candidate_location, "negative chain equals gold chain; candidate skipped"))
                continue
            else:
                if positive_chain_tuple and set(chain_tuple) == set(positive_chain_tuple):
                    score = 0.95
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            "negative uses the same evidence set as gold with different order; score raised to 0.95",
                        )
                    )
                negative_count += 1
                if candidate_type in {"background_only", "answer_adjacent"}:
                    has_background_negative = True
                if candidate_type == "background_only" and len(chain) < 4:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            "background_only negative is short; prefer 4-8 evidence items for long background hard negatives",
                        )
                    )
                if answer_evidence and set(answer_evidence).issubset(set(chain)) and score > 0.6:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            candidate_location,
                            "negative contains all answer_evidence ids with score > 0.6",
                        )
                    )

            normalized_candidates.append(
                {
                    "label": "positive" if candidate_type == "gold" else label,
                    "type": candidate_type,
                    "chain": chain,
                    "chain_role_tags": chain_role_tags,
                    "chain_text": chain_text(chain, evidence_by_id),
                    "score": round(score, 4),
                    "score_reason": str(candidate.get("score_reason") or "").strip(),
                }
            )
        if gold_count != 1:
            issues.append(ValidationIssue("error", sample_location, f"expected exactly one gold candidate, got {gold_count}"))
            continue
        if invalid_gold:
            continue
        if negative_count < 3:
            issues.append(ValidationIssue("warning", sample_location, f"sample has only {negative_count} negative candidates"))
        if not has_background_negative:
            issues.append(ValidationIssue("warning", sample_location, "missing background_only/answer_adjacent hard negative"))
        if len(normalized_candidates) < 2:
            issues.append(ValidationIssue("error", sample_location, "sample has fewer than 2 valid candidates"))
            continue
        normalized_dataset.append(
            {
                "query": query,
                "query_type": query_type,
                "answer": answer,
                "answer_evidence": answer_evidence,
                "answer_focus": answer_focus,
                "candidate_chain_list": normalized_candidates,
            }
        )

    raw_relations = payload.get("entity_relations")
    normalized_relations: list[dict[str, str]] = []
    if isinstance(raw_relations, list):
        for index, item in enumerate(raw_relations):
            location = f"{source_name}.entity_relations[{index}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue("warning", location, "entity_relation 不是对象"))
                continue
            head = str(item.get("head") or "").strip()
            relation = str(item.get("relation") or "").strip()
            tail = str(item.get("tail") or "").strip()
            evidence_id = normalize_evidence_id(item.get("evidence_id"))
            if not head or not relation or not tail:
                issues.append(ValidationIssue("warning", location, "三元组缺少 head/relation/tail"))
                continue
            if evidence_id and evidence_id not in evidence_by_id:
                issues.append(ValidationIssue("warning", location, f"未知 evidence_id: {evidence_id}"))
                evidence_id = ""
            normalized_relations.append(
                {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "evidence_id": evidence_id,
                }
            )

    normalized_payload = {
        "source_name": source_name,
        "all_evidence": normalized_evidence,
        "entity_relations": normalized_relations,
        "rerank_dataset": normalized_dataset,
    }
    return normalized_payload, issues


def convert_to_listwise_records(
    payloads: list[dict[str, Any]],
    *,
    evidence_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        # Build evidence lookup lazily (only when needed)
        eb_id = evidence_by_id
        if eb_id is None:
            eb_id = {e["id"]: e["text"] for e in payload.get("all_evidence", [])}

        for sample in payload["rerank_dataset"]:
            records.append(
                {
                    "query": sample["query"],
                    "query_type": sample["query_type"],
                    "answer": sample.get("answer", ""),
                    "answer_evidence": sample.get("answer_evidence", []),
                    "answer_focus": sample.get("answer_focus", ""),
                    "source_name": payload["source_name"],
                    "candidates": [
                        _build_candidate_record(candidate, eb_id)
                        for candidate in sample["candidate_chain_list"]
                    ],
                }
            )
    return records


def convert_to_pairwise_records(
    payloads: list[dict[str, Any]],
    *,
    evidence_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        eb_id = evidence_by_id
        if eb_id is None:
            eb_id = {e["id"]: e["text"] for e in payload.get("all_evidence", [])}

        for sample in payload["rerank_dataset"]:
            positives = [
                candidate
                for candidate in sample["candidate_chain_list"]
                if candidate["label"] == "positive" or candidate["type"] == "gold"
            ]
            negatives = [
                candidate
                for candidate in sample["candidate_chain_list"]
                if candidate["label"] == "negative" and candidate["type"] != "gold"
            ]
            for positive in positives:
                for negative in negatives:
                    pos_struct = _extract_chain_structure(positive["chain"], eb_id)
                    neg_struct = _extract_chain_structure(negative["chain"], eb_id)
                    pos_text_with_meta = _build_chain_text_with_metadata(positive["chain"], eb_id, pos_struct)
                    neg_text_with_meta = _build_chain_text_with_metadata(negative["chain"], eb_id, neg_struct)
                    records.append(
                        {
                            "query": sample["query"],
                            "query_type": sample["query_type"],
                            "source_name": payload["source_name"],
                            "positive": pos_text_with_meta,
                            "negative": neg_text_with_meta,
                            "positive_chain_structure": pos_struct,
                            "negative_chain_structure": neg_struct,
                            "positive_score": positive["score"],
                            "negative_score": negative["score"],
                            "negative_type": negative["type"],
                            "answer": sample.get("answer", ""),
                            "answer_evidence": sample.get("answer_evidence", []),
                            "answer_focus": sample.get("answer_focus", ""),
                            "positive_chain": positive["chain"],
                            "negative_chain": negative["chain"],
                            "positive_chain_role_tags": positive.get("chain_role_tags", []),
                            "negative_chain_role_tags": negative.get("chain_role_tags", []),
                        }
                    )
    return records


def convert_to_flag_embedding_records(
    payloads: list[dict[str, Any]],
    *,
    evidence_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for payload in payloads:
        eb_id = evidence_by_id
        if eb_id is None:
            eb_id = {e["id"]: e["text"] for e in payload.get("all_evidence", [])}

        for sample in payload["rerank_dataset"]:
            positives: list[str] = []
            negatives: list[str] = []
            for candidate in sample["candidate_chain_list"]:
                chain = candidate.get("chain", [])
                structure = _extract_chain_structure(chain, eb_id)
                text_with_meta = _build_chain_text_with_metadata(chain, eb_id, structure)
                is_positive = candidate["label"] == "positive" or candidate["type"] == "gold"
                if is_positive:
                    positives.append(text_with_meta)
                elif candidate["label"] == "negative" and candidate["type"] != "gold":
                    negatives.append(text_with_meta)
            if positives and negatives:
                records.append(
                    {
                        "query": sample["query"],
                        "answer": sample.get("answer", ""),
                        "pos": positives,
                        "neg": negatives,
                    }
                )
    return records


def command_export(args: argparse.Namespace) -> int:
    input_paths = [Path(path) for path in args.inputs]
    normalized_payloads: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    for path in input_paths:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            payload = load_json_payload(resolved)
            normalized, payload_issues = validate_and_normalize_payload(payload, source_name=resolved.name)
        except Exception as exc:
            issues.append(ValidationIssue("error", resolved.name, str(exc)))
            continue
        normalized_payloads.append(normalized)
        issues.extend(payload_issues)

    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]
    if not normalized_payloads:
        for issue in issues:
            print(f"{issue.level.upper()} {issue.location}: {issue.message}", file=sys.stderr)
        return 2
    if errors and not args.allow_errors:
        for issue in issues:
            print(f"{issue.level.upper()} {issue.location}: {issue.message}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a global evidence_by_id from all payloads for chain structure analysis
    global_evidence_by_id: dict[str, str] = {}
    for payload in normalized_payloads:
        for e in payload.get("all_evidence", []):
            if e["id"] not in global_evidence_by_id:
                global_evidence_by_id[e["id"]] = e["text"]

    listwise_records = convert_to_listwise_records(normalized_payloads, evidence_by_id=global_evidence_by_id)
    pairwise_records = convert_to_pairwise_records(normalized_payloads, evidence_by_id=global_evidence_by_id)
    flag_records = convert_to_flag_embedding_records(normalized_payloads, evidence_by_id=global_evidence_by_id)

    cleaned_records = normalized_payloads
    write_jsonl(output_dir / "annotations.cleaned.jsonl", cleaned_records)
    write_jsonl(output_dir / "reranker_listwise.jsonl", listwise_records)
    write_jsonl(output_dir / "reranker_pairwise.jsonl", pairwise_records)
    write_jsonl(output_dir / "flag_embedding_reranker.jsonl", flag_records)
    issue_records = [
        {"level": issue.level, "location": issue.location, "message": issue.message}
        for issue in issues
    ]
    write_jsonl(output_dir / "validation_issues.jsonl", issue_records)

    summary = {
        "inputs": [str(path) for path in input_paths],
        "payloads": len(normalized_payloads),
        "samples": sum(len(payload["rerank_dataset"]) for payload in normalized_payloads),
        "listwise_records": len(listwise_records),
        "pairwise_records": len(pairwise_records),
        "flag_embedding_records": len(flag_records),
        "errors": len(errors),
        "warnings": len(warnings),
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if (not errors or args.allow_errors) else 2


def command_recover_json(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    payload = parse_json_text(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "input": str(input_path),
        "output_json": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate Arknights evidence-chain reranker datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_prompts = subparsers.add_parser("make-prompts", help="Render story files into Minimax annotation prompts.")
    make_prompts.add_argument("--story-files", nargs="*", default=[])
    make_prompts.add_argument("--story-dir", type=Path)
    make_prompts.add_argument("--glob", default="*.txt")
    make_prompts.add_argument("--prompt-id")
    make_prompts.add_argument("--output-dir", type=Path, default=Path("outputs/evidence_chain_prompts"))
    make_prompts.add_argument("--min-questions", type=int, default=10)
    make_prompts.add_argument("--max-questions", type=int, default=15)
    make_prompts.add_argument("--max-source-chars", type=int, default=90000)
    make_prompts.set_defaults(func=command_make_prompts)

    call_api = subparsers.add_parser("call-api", help="Call an OpenAI-compatible chat completions API for one prompt file.")
    call_api.add_argument("prompt_file", type=Path)
    call_api.add_argument("--output-json", type=Path, required=True)
    call_api.add_argument("--api-base", default=os.environ.get("MINIMAX_API_BASE", "https://api.svips.org"))
    call_api.add_argument("--endpoint-path", default="/v1/chat/completions")
    call_api.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    call_api.add_argument("--model", default="")
    call_api.add_argument("--model-env", default="MINIMAX_MODEL")
    call_api.add_argument("--temperature", type=float, default=0.2)
    call_api.add_argument("--max-tokens", type=int, default=12000)
    call_api.add_argument("--timeout", type=float, default=300.0)
    call_api.add_argument("--response-format-json", action="store_true")
    call_api.add_argument("--quiet", action="store_true")
    call_api.set_defaults(func=command_call_api)

    export = subparsers.add_parser("export", help="Validate Minimax JSON annotations and export reranker training files.")
    export.add_argument("inputs", nargs="+")
    export.add_argument("--output-dir", type=Path, default=Path("data/processed/evidence_chain_reranker"))
    export.add_argument("--allow-errors", action="store_true")
    export.set_defaults(func=command_export)

    recover = subparsers.add_parser("recover-json", help="Recover model text output into a parsed JSON file.")
    recover.add_argument("input", type=Path)
    recover.add_argument("--output-json", type=Path, required=True)
    recover.set_defaults(func=command_recover_json)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
