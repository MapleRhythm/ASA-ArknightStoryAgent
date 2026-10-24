# v55 当前训练：第三版已越过先前失败位置

2026-09-11，状态为运行中，未完成、未晋级。
本报告取代第一版“启动成功”记录作为当前状态；旧记录和失败均保留。

## 当前运行

- 训练服务器：`${REMOTE_USER}@${REMOTE_HOST}`，A100 GPU0，共享运行。
- PID：`3735215`；tmux会话：`asa_sft_v55_a03`。
- 工作目录：`${REMOTE_STORE}/exx_grounding_v1/runs/asa_v55_a03_20260911`。
- 日志：工作目录下 `train.log`。
- 逐步指标：`model/trainer_log.jsonl`；优化步回执：`steps.jsonl`。
- 输出模型：`model/`。完成后有 `complete.json`；异常保存在 `failure.json`。
- 训练循环于248时间12:23:15进入，第1步12:23:52完成。
- 已核验快照：第7/114步，loss=0.49566242099；先前失败在完成第5步之后。
- 第7步回执的PyTorch峰值分配约17.4GiB，不代表全程最终峰值。
- 常驻的其他用户PID450914仍占1892MiB，未停止或修改。

910条历史训练、95条验证，1 epoch、LoRA rank16、学习率3e-6、
batch1/累积8、bf16、gradient checkpointing、qwen3_nothink、cutoff12000。
基座Qwen3.5-4B，起点为历史SFT LoRA
`exx_grounding_v1_sft_success559_a100_r16_lr3e6_e1_20260827`，
不是当前线上SODA LoRA。新模型单独保存。

## 两次失败及本次调整

1. 第一版 `asa_historical_sft_20260911_v55`：完成5步后CUDA OOM；
   一个长样本的反向传播请求8.61GiB，超过保留共享余量的85%分配器上限。
   PID3725104已退出，未产生可恢复checkpoint；原始日志和失败回执均保留。
2. 第二版 `asa_historical_sft_20260911_v55_attempt02`：预处理的临时目录过长，
   引发 `AF_UNIX path too long`，主进程表现为EOFError；0个训练步。
   PID3732674已退出，完整失败记录保留。
3. 第三版：仍保留85%上限，使用已有Liger0.8.2的融合线性层/交叉熵计算，
   不替换RMSNorm、SwiGLU或rope；另使用短的mktemp临时目录，
   在启动前实际验证multiprocessing Manager通信。

没有删样本、改标签、缩短原文或目标、调低学习率或改epoch来绕开失败。
三版完整编码摘要一致：训练最长9830、验证最长8855 tokens，截断0条。
8项独立安全测试通过；重复执行不累加测试数。
真实小型Qwen3.5模型的float32/bf16 loss及梯度对比通过（各14个梯度张量）。
这不保证训练轨迹逐位一致；真实训练前5步loss最大绝对差约0.0009774。

## 回传与边界

241规范目录：
`${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/asa_v55_a03_20260911`。
第7步启动阶段快照已回传 `incoming_from_248/startup_receipt.tgz`，并解包核验。
前两版失败归档也已回传各自的241实验目录，未覆盖旧启动报告。

绑定SHA-256：

- 当前快照：`6ec14d35d96b37a59738837f5a72c38da063b7356847acb703c99afa8df3b2ca`
- 当前预检：`00844551658f2d70f0238c291559ea78f315995026f66e7ae3f548996d5af030`
- 第7步回执归档：`918f9c77b60259bd930828c37126a95ebea6ee6a718e0d83b0449c678c5d052c`
- 第一版失败归档：`1e3c478c66a2e48d5d71103867d530e304e9e922da634174e815097221842bb0`
- 第二版失败归档：`f9c5b822d198ecef060f731ff90da46e1ea104f62ead1d310ff53e07e24ffe63`

这是历史数据重训，不是新标注或数据扩容；本轮GLM新标签仍为0。
历史数据中245条经GLM压缩、757条复制、3条失败回退复制，不能称为全部新GLM标签或人工金标准。
训练loss不是剧情理解或检索成绩。公开benchmark与检索评测尚未运行。
没有改线上默认模型、检索索引或服务，没有安装依赖或改已有库文件、提交、推送。
请勿重复启动本版本；续接先查看实际进程、日志及完成/失败标记。
