"""Write an evidence-bound checkpoint report on canonical 241; never promote."""
import datetime
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
manifest = read(run / "manifest.json")
test_suites = ET.parse(run / "diagnostic_tests_v1.xml").getroot().iter("testsuite")
test_suites = list(test_suites)
test_count = sum(int(s.get("tests", 0)) for s in test_suites)
if test_count != 26 or any(int(s.get("failures", 0)) or int(s.get("errors", 0)) for s in test_suites):
    raise ValueError("diagnostic tests are not verified 26/26")
checked = []
for relative, info in manifest["source_files"].items():
    if not relative.startswith("after/"):
        continue
    path = Path(info["origin"])
    current = sha(path) if path.is_file() else None
    checked.append({"path": str(path), "frozen_sha256": info["sha256"],
                    "current_sha256": current, "unchanged": current == info["sha256"]})
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
diff_check = subprocess.run(["git", "-C", str(repo), "diff", "--check"], capture_output=True, text=True)
state = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "run_id": run.name, "canonical_host": "${REMOTE_HOST}",
    "frozen_head": manifest["git_head"], "current_head": head,
    "source_files_checked": len(checked), "changed_source_files": [r["path"] for r in checked if not r["unchanged"]],
    "diff_check_returncode": diff_check.returncode, "diff_check_output": diff_check.stdout + diff_check.stderr,
    "files": checked,
    "gpu_compute_processes": subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"
    ], text=True),
}
with (run / "canonical_state_checkpoint_v1.json").open("x") as f:
    f.write(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
specs = [
    ("Transformers / 当前LoRA / 旧截断", "configured_lora_ctx12288/semantic_summary18.json", "before"),
    ("Transformers / 当前LoRA / 完整正文", "configured_lora_ctx12288/semantic_summary18.json", "after"),
    ("Transformers / base / 同tokenizer", "strict9_base_same_tokenizer_ctx12288/semantic_summary9.json", "after"),
    ("Transformers / 当前LoRA / 加阅读规则", "strict9_lora_source_fidelity_v1_ctx12288/semantic_summary9.json", "source_fidelity_v1"),
    ("Transformers / 当前LoRA / 恢复原换行", "strict9_lora_original_layout_v1_ctx12288/semantic_summary9.json", "original_dialogue_layout_v1"),
    ("vLLM / 当前LoRA / 旧截断", "strict9_configured_vllm_native_sampler_ctx10000/semantic_summary18_retry1.json", "before"),
    ("vLLM / 当前LoRA / 完整正文", "strict9_configured_vllm_native_sampler_ctx10000/semantic_summary18_retry1.json", "after"),
    ("Transformers / 当前LoRA / fixed79", "fixed79_configured_lora_ctx12288/semantic_summary79_retry1.json", "after"),
]
table = []
artifacts = []
for title, relative, group in specs:
    path = run / relative
    if not path.is_file():
        raise ValueError(f"summary not complete: {relative}")
    summary = read(path)["summary"][group]
    labels = summary["labels"]
    failures = summary["records"] - sum(labels.get(k, 0) for k in (
        "supported_complete", "supported_partial", "unsupported", "no_answer", "uncertain",
    ))
    table.append(
        f'| {title} | {summary["records"]} | {labels.get("supported_complete", 0)} | '
        f'{labels.get("supported_partial", 0)} | {labels.get("unsupported", 0)} | '
        f'{labels.get("no_answer", 0)} | {failures + labels.get("uncertain", 0)} |'
    )
    artifacts.append({"path": relative, "sha256": sha(path), "group": group})
fixed = read(run / "fixed79_configured_lora_ctx12288/semantic_summary79_retry1.json")["summary"]["after"]
report = f"""# ASA 规范服务器评测与修复检查点

实验：`{run.name}`。创建时间（UTC）：`{state["created_at"]}`。
本文记录一次持续实验的检查点，不表示项目目标完成或已经上线。

## 结论与可信度

1. 241是唯一代码、数据与结果基准；248仅作为训练/评测镜像。本轮没有向248写入或启动训练。
2. 79条生成及最终展示答案审核已完成一轮，随后只重试3条失败审核：77条满足审核schema，2条仍失败。
   **自动审核暂判{fixed["fully_supported_complete_records"]}/79（{fixed["complete_yield_all_records"]:.1%}）完整正确。这个数字不是经人工确认的正确率。**
   对自动“完整正确”项继续复核，已发现至少两项明显过宽判分：
   - 乔迪题：原文“你的父母，严格来说，是你的祖父母，来自阿戈尔”，答案忽略纠正而称父母来自海洋，judge仍判完整。
   - 夜刀题：答案只说“与另一个为泰拉所熟知的危机有关联”，未明确是什么危机；原文同时提供矿石病/源石的线索，judge却把泛称视作完整答案。
   原评分保留；不悄悄修成gold，也不把其余自动评分都宣称已经人工确认。
3. 单纯去掉LoRA、追加通用阅读规则、恢复对话换行，都未在本组开发题上改善完整回答产出，不采纳为默认方案。
4. 完整证据展示后仍出现人物错归、因果编造、遗漏条件和复述题干不作答。
   引文能在正文中匹配，不能证明绑定事实的语义正确；整个上下文能支持答案，也不代表每个被引用E块绑定正确。
5. 需要优先区分“证据确实不足”与“证据够但生成或引文复制失败”，不能把所有问题都归因于召回或都送去再次检索。

## 固定分母与实验边界

79记录包含58个规范化问题、20个重复问题组，轮次为1/2、2/2、1/3、2/3和3/3混合。
这是历史**可见证据状态回放**，不是79道独立问题的最终多轮端到端回答。
本轮未重新检索，未执行线上完整orchestrator恢复或末轮direct fallback。
历史提示中已经缺失的正文不会因本次取消截断凭空恢复。
这套quote输出协议和最终答案rubric不同于历史Exx审核，不能把历史25/79、31/79直接当同口径基线。

9题是已反复分析的开发诊断集，不再作为独立未见holdout来宣传提升。
所有生成输入均不含gold答案、历史review或judge结果；实际配置的LoRA在241存在并确实被加载。
base消融固定使用同一LoRA目录tokenizer，避免同时改变分词器。

## 自动语义判分（仅暂定模型审核）

下表“未答”指本证据状态下retrieve_more或abstain；不是完整多轮系统最终必然未答。
部分正确不并入完整正确；不确定和审核失败不从分母删除。

| 实验 | 记录 | 完整 | 部分 | 不受支持 | 未答 | 不确定/审核失败 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

fixed79原生成75条answer_directly，经引用校验后41条仍回答、29条retrieve_more、9条abstain。
77条有效judge中，69条判证据充分、8条不足；31条在证据充分时仍未答。
这些也是模型暂定判断，不可称为RAG真实召回率。
有1条达到2048输出上限，原因是重复生成实体列表；它仍留在79分母中，解析器回退为retrieve_more。

## 已完成的真实验证与未晋级候选

- 上一阶段在241发布源码修复10文件、新增3测试，取消每块1000字符截断、使用完整块预算、
  绑定实际可见E编号、关闭隐藏正文/空证据回退等；90/90测试通过。见既有`REMOTE_CANONICAL_STATUS_20260908.md`。
- 本检查点对比冻结后的规范源文件：{len(checked)}个文件，发现{len(state["changed_source_files"])}个内容变化。
  HEAD从`{manifest["git_head"]}`到`{head}`。详见`canonical_state_checkpoint_v1.json`。
- 新增诊断工具的26项单测通过，其中8项验证候选token预算保护；另有候选盲读工具测试单独记录。
  **候选源码的测试通过不代表正式源码已新增这些行为。**
- 真实vLLM 0.26.0、配置LoRA和10000上下文已完成18条生成。
  每条实际输入token ID与未截断提示逐个一致，全部正常stop；不把Transformers的结果冒充vLLM。
- 初次vLLM启动因FlashInfer采样编译缺`curand_precalc.h`失败。
  仅在实验进程设置`VLLM_USE_FLASHINFER_SAMPLER=0`后完成运行；未改系统CUDA、包环境或服务。
- 配置10000上下文下有2/9 strict9及2/79 fixed79无法同时预留2048输出token。
  实际18条vLLM输出较短，没有因此被截断；这是容量隐患，不是已确认线上截断事故。
- 候选增加了精确token预算前检与finish reason诊断，并拟验证12288上下文。
  候选位于`context_guard_candidate_bytes_v1`，204份文件中仅VllmRunner变化。
  8项单测通过，但GPU验证启动时检测到其他计算进程，安全拒绝运行；
  **尚未修改正式源码或10000配置，也未声称12288的真实vLLM验证通过。**
- 两次普通复制候选时挂载路径拒绝chmod；失败目录保留。
  随后用只复制文件内容并逐项验证SHA的方式建立候选，未修改挂载权限。
- 未提交、未push、未重启服务、未覆盖main worktree、未训练未审定标签。

## 下一步工作

1. 完成34条“原模型拟回答、被quote校验拦截”的全量诊断，区分真实语义错误与可修的引用错误。
   它们不是已展示答案，不能混入当前正确产出。
2. 检查候选盲读：先仅按问题和证据独立列应答要点，再看候选答案，防止裁判跟着错误答案补理由。
   先用两例已发现误判的小样验证；结果不作为无偏正确率或训练gold。
3. 空闲GPU可用后验证token预算候选、12288上下文，以及有依据的同证据重生成方案。
   241三张4090及248的A100在本次候选启动检查时都有现存进程；未占用或终止其他用户任务。
4. 只有源文本和评测审定后，才在241冻结训练快照并送248；不把本地实验或自动judge输出直接晋级。

## 结果位置

规范实验目录：

```text
{run}
```

关键文件：

- `manifest.json`：源代码、配置、strict9、历史79输入来源与哈希。
- `configured_lora_ctx12288/`：Transformers配对18条及原始语义审计。
- `fixed79_configured_lora_ctx12288/`：79条生成、引用校验、初始/重试审核与完整分母汇总。
- `strict9_configured_vllm_native_sampler_ctx10000/`：真实vLLM18条及输入token逐项核对。
- `strict9_base_same_tokenizer_ctx12288/`、`strict9_lora_source_fidelity_v1_ctx12288/`、
  `strict9_lora_original_layout_v1_ctx12288/`：未被采纳的消融，失败结果同样保留。
- `source_review_notes_v1.json`：6项单独源文复核，含2条schema错误及有争议判分。
- `candidate_blind_reference_smoke2*`：候选盲读小样；可能仍在运行。
- `context_guard_candidate_bytes_v1/manifest.json`：候选204文件字节校验与唯一改动清单。
- `diagnostic_tests_v1.xml`：26项工具/候选测试。
"""
with (run / "PHASE_REPORT_v1.md").open("x") as f:
    f.write(report)
doc_path = repo / "docs/CANONICAL_ANSWER_QUALITY_CHECKPOINT_20260908_irPugk.md"
with doc_path.open("x") as f:
    f.write(report)
if sha(doc_path) != sha(run / "PHASE_REPORT_v1.md"):
    raise ValueError("canonical report copy differs")
with (run / "phase_report_artifacts_v1.json").open("x") as f:
    f.write(json.dumps({
        "evaluation_only": True, "training_ready": False, "report_sha256": sha(run / "PHASE_REPORT_v1.md"),
        "summary_artifacts": artifacts, "source_state_sha256": sha(run / "canonical_state_checkpoint_v1.json"),
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_files_checked": len(checked), "changed_source_files": state["changed_source_files"],
                  "report": str(run / "PHASE_REPORT_v1.md"), "canonical_doc": str(doc_path),
                  "report_sha256": sha(run / "PHASE_REPORT_v1.md")}))
