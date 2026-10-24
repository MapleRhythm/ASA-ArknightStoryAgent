# 4B：20 条完整证据，禁止静默截断

已应用到远端 `${REMOTE_HOME}/ASA-ArknightStoryAgent`，未覆盖本地另一个版本及其未提交工作。未重启服务、未训练、未调用生成模型。

## 已修改

- `configs/runtime_inference_gpu.json`：前 20 条证据，完整正文开关开启，CRAG 关闭，每篇和总字数上限均为 0（实现中解析为无上限）。
- `src/goldenglow/inference/cpu_pipeline.py`：
  - 完整模式保留检索结果前 K 条原顺序，不再二次去重、句段精炼、重选或用证据链替换正文。
  - 完整开关同时取消每篇、总字数限制，旧调用者即使传了正数字数上限也不生效。
  - conclusion、direct answer 和 follow-up hypothesis 三条输入路径都可保留完整证据正文。
  - 完整模式将三个阶段的 prompt、SHA256、证据 ID 写入成功结果的 `model_runtime.full_prompt_snapshots`；每道新题清空旧快照。
  - vLLM 显式计算完整输入 token + 输出预留空间，用同一组 token IDs 请求生成。超过上下文容量抛出 `PromptContextOverflowError`，不删证据，也不在结论/最终兜底中吞掉错误并伪装成拒答。
- `scripts/run_latest_kto_interactive.sh`：默认前 20 条并强制完整模式，不再用环境变量重新启用证据字数裁剪。
- 新增 `tests/test_full_evidence_contract.py`。

这些设置针对当前 GPU 默认配置/上述启动脚本；未改动其他历史实验配置。已运行进程不会自动热加载新 Python 代码。

这里“完整”指检索结果的 `clean_text` 正文，不代表把其所属整章剧情全文重新取回。保留已有换行/空白规范化和内部标记清理，不删除剧情句段。证据数量仍为用户指定的 20 条。

## 验证

本地暂存版本和远端实际修改版本均通过 8 项离线回归测试，覆盖：

- 第 32 句段之后的内容、每篇超过 1000 字、总长度超过 12000 字仍保留。
- 相似文档不再挤掉已选的 20 条，证据链不替换正文。
- conclusion 的 minimal/full 模式、直接回答、后续检索输入均保留正文。
- 容量超限明确报错，不调用生成，也不转为“证据不足”。
- 发送给 vLLM 的 token IDs 与容量检查用的完全相同。
- 新题清空旧快照；默认配置完整模式开启。

另用先前六题保存的最终检索证据和远端实际 LoRA tokenizer 做了 CPU-only 输入构造验证：

| 题目 | conclusion 输入 tokens | 直接回答输入 tokens | 后续检索输入 tokens |
| --- | ---: | ---: | ---: |
| 古米训练 | 6464 | 6361 | 7776 |
| 凛冬领袖 | 5549 | 5446 | 6844 |
| 真理姓名/学校 | 6335 | 6232 | 7667 |
| 薇卡 | 7212 | 7109 | 8616 |
| 梅与无人机 | 5804 | 5701 | 7181 |
| 泡普卡 | 6048 | 5945 | 7330 |

18 份构造输入逐一核验了 20 条完整正文（忽略已有空白规范化、内部标记清理），全部通过。
conclusion/直接回答预留 1536 输出 tokens，后续检索预留 384；均能放入现有 10000-token 上下文，无需扩大配置。
薇卡的“最好的朋友”关键句已确认存在。

完整构造输入与 token 计数保存在 `six_prompt_capacity.jsonl`。这是新代码的输入验证，不是重新运行 4B 得到的答案，也不是历史逐轮真实 prompt 的恢复。

## 备份与文件指纹

- 修改前备份：本目录 `prechange.tar.gz`；远端 `/tmp/asa-no-truncate.PS4BZO/prechange.tar.gz`。本地 `original/` 也保留了三份修改前文件。
- 可核验补丁：`no_truncate.patch`。
- 修改后远端源码 SHA256：`29dfbba72879cec2fa58dabb9c7f6ecc7a36359a457f2482e008e399db790fdc`
- 修改后配置 SHA256：`49dfe81d5a2349f213b556abdee218f24a644ae3ab92bb57a8a460740e854527`
- 修改后启动脚本 SHA256：`f5321b67d6602141d3db3153368a01b20e73134d0fc4f1854d220c4dbff49f39`
- 测试文件 SHA256：`086e4e0c7735f3adef3f859006ca039a3eb45d46e72dc7e299279cab1c25ffde`

保存并保留了远端配置此前已有的未提交内容，只应用此次窄范围补丁；未重置工作区或删除任何结果。
