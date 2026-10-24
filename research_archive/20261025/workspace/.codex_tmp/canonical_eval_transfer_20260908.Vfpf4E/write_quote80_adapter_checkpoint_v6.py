"""Persist the completed, negative existing-adapter comparison without promotion."""
import datetime
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


run = Path(__file__).resolve().parent
repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent")
comparison_path = run / "quote80_adapter_vs_seeded_control9_v1.json"
comparison = load(comparison_path)
summary = comparison["summary"]
assert summary["records_per_model"] == 9
assert summary["control_labels"] == {"supported_complete": 3, "no_answer": 4, "supported_partial": 1, "unsupported": 1}
assert summary["candidate_labels"] == {"supported_complete": 2, "no_answer": 6, "unsupported": 1}
assert all(r["input_tokens_and_seed_match_control"] for r in comparison["rows"])
manifest_path = run / "quote80_sft_adapter_probe_manifest_v1.json"
manifest = load(manifest_path)
train, val = [manifest["recorded_dataset_splits"][s] for s in ("train", "val")]
assert train["counts"]["rows"] == 315 and train["counts"]["answer_with_fact_quotes"] == 152
assert val["counts"]["rows"] == 27
assert len(manifest["recorded_train_val_question_overlap"]) == 11
folder = run / "quote80_sft_adapter_probe9_ctx12288_v1"
metadata = load(folder / "metadata.json")
assert metadata["completed_rows"] == 9 and metadata["all_input_token_ids_match_control"]
applied = load(run / "canonical_inferred_integrity_patch_v2/applied_verified.json")
for item in applied["effective_changes"]:
    assert sha(repo / item["path"]) == item["after_sha256"]
text = f"""# ASA quote-SFT现有权重隔离对照检查点 v6

记录时间（UTC）：`{datetime.datetime.now(datetime.timezone.utc).isoformat()}`。
241仍为规范源；本报告接续v5，不表示长期正确率目标完成。

## 结果：不切换到这只quote-SFT权重

对相同strict9，用当前权重的最近显式seed控制结果，对比已存在且训练完成的quote-SFT权重。
完整输入字符串、实际token IDs、逐条seed、基础模型全部资产哈希、vLLM/PyTorch版本、
runtime runner代码哈希和12288上下文容量均已核对一致。
本次只更换隔离评测进程中的adapter，没有改变正式模型指针、配置或服务。

以同一新版实质内容审核口径（自动暂判，非独立人工gold）统计：

| 权重 | 状态数 | 完整 | 部分 | 不支持 | 未提供实质回答 | 审核失败 |
|---|---:|---:|---:|---:|---:|---:|
| 当前SODA LoRA控制 | 9 | 3 | 1 | 1 | 4 | 0 |
| quote-SFT候选 | 9 | 2 | 0 | 1 | 6 | 0 |

候选9条均正常stop、严格JSON合法，没有解析异常或输出上限命中，但引用校验只放行3条。
这进一步说明不能用“JSON合法”或“训练过quote格式”替代回答内容核验。
这次仅有每题一次采样，控制生成时间也早于候选；没有排除所有数值/采样波动。
二者整个训练历史不同，不能把差异因果归结为“quote训练本身有害”。

## 已核验的候选来源

实际候选：

```text
{manifest["selected_adapter"]}
```

模型卡、配置及trainer_state对应SFT完成80/80步、2个epoch；记录基座为原始qwen3.5-4b，
不是把适配merged base的LoRA误装到原始基座上。评测前后权重哈希一致。
词表、merges、added tokens与控制一致，实际输入token还再次逐条核验。

保存的数据：

- train：315记录、242问题；152条answer_directly均有supported_facts和fact quote字段。
- val：27记录、27问题；17条answer_directly均有fact quote字段。
- train/val仍重叠11个问题；train与原79状态重叠15个问题，与本strict9重叠1个问题（莫斯提马）。
- val与原79状态重叠26个问题。这是开发重叠信息，不作独立正确率评价。

这里确认的是记录的训练契约和文件来源，未认证历史训练缓存字节。
有quote字段不等于标签已逐条语义审定，也不表示包括推断引用在内的全部新协议都已对齐。

## 具体问题

- 雷吉斯题能给出有证据的原因回答。
- 带剑女子题虽然引用放行，却增加“形影不离”“唯一能理解她的人”等关系强断言。
  新版审核将该条判为不支持；不能因为身份“渡魂剑所化”部分正确就把整条当完整正确。
- 老天师题仍产生没有引用的inferred_fact，被新完整性检查拦下。
  这是推断引用契约尚不兼容的实际例子，不能为了提升通过率重新丢弃推断引用检查。
- 其他多条仍复制了不在相应E块中的quote，说明训练了引用结构也未解决精确引用与事实理解全部问题。

不自动删错fact或拼接剩余内容宣称“完整回答”；不把弃答增加包装为正确题数提高。
目前不扩大此候选为默认，也不将它作为已验证改善方案。

## 当前正式工作区状态

正式release仍为v4记录的115/115测试通过版本；三组完整性修复累计9个不同文件，
基准HEAD不变，所有这些文件的最后哈希再次核对通过。
本轮没有训练、提交、push、服务重启、正式模型切换或训练数据改写。
v5的训练重叠/冲突标签证据和实质内容评测修复保留。

下一步先审核这只quote-SFT后续KTO权重的数据和训练契约，再决定是否值得在同样控制条件下对照；
尚未测试其后续KTO，不能沿用本轮结果评价它。
长远仍需问题/剧情家族隔离、已审定数据、逐条引用语义核验，以及真正新检索/多轮/fallback的E2E评测。

规范实验目录：

```text
{run}
```

关键凭据：
`quote80_sft_adapter_probe_manifest_v1.json`、
`quote80_sft_adapter_probe9_ctx12288_v1/`、
`quote80_adapter_vs_seeded_control9_v1.json`。
"""
paths = [
    run / "QUOTE80_ADAPTER_CHECKPOINT_v6.md",
    repo / "docs/CANONICAL_QUOTE80_ADAPTER_COMPARISON_20260908_irPugk_v6.md",
]
for path in paths:
    with path.open("x") as output:
        output.write(text)
assert sha(paths[0]) == sha(paths[1])
with (run / "quote80_adapter_checkpoint_v6_artifacts.json").open("x") as output:
    output.write(json.dumps({
        "report_sha256": sha(paths[0]), "comparison_sha256": sha(comparison_path),
        "manifest_sha256": sha(manifest_path), "metadata_sha256": sha(folder / "metadata.json"),
        "promoted": False, "training_ready": False, "goal_complete": False,
    }, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"report": str(paths[1]), "sha256": sha(paths[0]), "promoted": False}))
