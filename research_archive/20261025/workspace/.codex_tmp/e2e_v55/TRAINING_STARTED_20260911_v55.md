# v55：248 A100 训练已实际启动

2026-09-11；下列启动时间来自248服务器，时区为 Asia/Shanghai。

状态：训练运行中，未完成、未晋级。启动记录已回传241并验证 SHA。
已保存的交接快照包含第1、2个优化步；不是仅启动了 shell 或加载模型。

## 运行位置

- 服务器：`${REMOTE_USER}@${REMOTE_HOST}`
- PID：`3725104`
- tmux：`asa_sft_v55`
- 工作目录：
  `${REMOTE_STORE}/exx_grounding_v1/runs/asa_historical_sft_20260911_v55`
- 完整训练日志：工作目录下 `train.log`
- 逐步指标：`model/trainer_log.jsonl`
- 独立优化步回执：`steps.jsonl`
- 新模型输出：`model/`
- 成功完成标记：`complete.json`；异常标记：`failure.json`

训练循环于12:01:52进入；第1个优化步于12:02:27完成。
交接快照记录第2/114步，loss=0.6771358848，epoch约0.01758。
首步loss=0.6929897666；这些训练loss不是剧情问答或检索准确率。

## 实际训练范围

- 历史训练910条、验证95条、测试0条，共1005条记录；目录名中的559不是条数。
- 训练任务：783条grounded action、63条初始hypothesis、64条后续hypothesis。
- 验证任务：79条grounded action、8条初始hypothesis、8条后续hypothesis。
- 基座：Qwen3.5-4B；初始化LoRA：
  `exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827`。
- 沿用该历史数据对应SFT配置：rank16、学习率3e-6、1 epoch、
  batch1/梯度累积8、bf16、gradient checkpointing、qwen3_nothink、
  cutoff12000；实际可训练参数32,464,896。
- 新版本显式seed20260911、独立缓存/输出、每步记录loss、保留本轮checkpoint。

这是历史数据的独立重训，不是当前线上SODA LoRA续训，也不是新增标注或扩容实验。
该历史数据的构建审计记录245条经GLM压缩、757条直接复制、3条失败回退复制；
不可将全部1005条说成这次新生成的GLM标签，更不是独立人工金标准。
v54本次GLM鉴权失败产生的标签仍为0条，未进入训练。

## 启动核验与共享资源

- 241冻结原始配置、完整数据、代码和模型身份；248逐项核对后才启动。
- 8项独立安全测试通过，本地/241/248重复执行不合计成24项。
- 1005条都用真实训练模板/分词器进行无限长度和12000上限编码对比，
  输入token及监督label完全相同；截断0条，监督目标均非空。
- 最长训练9830 tokens，最长验证8855 tokens。
- 使用已有248训练环境，未安装、升级或修改软件包。
- 用户明确允许共享A100；启动前只有获准共享的其他用户PID450914，
  占1892MiB。交接检查该进程仍在，未发送任何停止或修改命令。
- 本训练进程PyTorch allocator上限为总显存85%；这是分配器限制，
  不代表控制所有CUDA分配或保证其他用户未来任意扩容。
- 启动磁盘剩余约15.5GB；脚本在磁盘低于3GiB时会使本任务报错停止，
  不删除历史文件，不自动重试或削减训练数据。

## 规范记录与后续

241规范实验目录：
`${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/asa_historical_sft_20260911_v55`

启动回执归档：
`incoming_from_248/startup_receipt.tgz`，并已解包核验。
这是启动阶段快照，日志内容仅截至当时，不能当作训练完成报告。

SHA-256：

- `snapshot.json`：
  `24eace21c559c530826fd8326ef8c6b98046121c1cb339116f5749dd15e7abbf`
- 241训练快照压缩包：
  `702aa3fcddfd78f4f451c1011f7c8eb1654c2336f4ddb40afa3c7a0b8a57b31a`
- 248预检：
  `51f96c7b303b9bd49e37ce42399bf07756424e405996e11c3e9c7530caf83d1c`
- 248启动回执压缩包：
  `55c7500adb6dba17508a4de04ffc61e6520cbfb6078b6526849b45d30939e8d8`

没有改旧数据、默认模型、推理服务、检索索引；没有提交或推送。
不要重复执行本版本启动器，也不要覆盖旧运行；如需续接，先读实际状态。
训练完成后仍需回传新模型、核对哈希，并评测剧情理解和检索。
此前公开benchmark评测因用户要求优先训练而暂停，尚无新benchmark成绩。
