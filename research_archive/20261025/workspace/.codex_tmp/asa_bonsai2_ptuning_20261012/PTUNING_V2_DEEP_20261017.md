# P-tuning v2：Qwen3.5 / Bonsai 27B 混合架构适配

## 实现边界

原来的 `PromptEncoder` 是输入端16个软提示加残差MLP，不是逐层 P-tuning v2。
本实验方法名为 `p_tuning_v2_hybrid_direct_kv_and_gdn_prefix`，独立保存，不覆盖旧实验。

- 16个全注意力层：每层独立、直接优化的K/V前缀，K在RoPE之前注入。
- 48个Gated DeltaNet层：每层独立的归一化隐藏态前缀，经过冻结的投影、卷积和循环更新驱动状态。不是给GDN伪造普通KV cache。
- 在输入端一次预留16个位置，各层复用这些位置，不累加序列长度。
- 共4,456,448个FP32可训练参数；输入种子是固定buffer。
- 主干冻结，但训练时梯度仍经过冻结的可微运算。不是对GGUF服务器做反向传播。
- 这是P-tuning v2的**混合架构适配版**，不是原论文模型的原封不动移植。

参考：THUDM/P-tuning-v2 官方 `model/prefix_encoder.py` 的独立逐层K/V参数；
当前环境 Transformers 5.6.2 的真实 Qwen3.5 源码。

## 初始化与对照

只使用旧750条数据训练实验的 **initial_adapter**，不用其final_adapter。
先用冻结主干收集该初始浅层提示在每层的激活，分别初始化GDN输入和全注意力K/V；
要求逐层初始化后的logits复现原初始浅层提示（真实27B门检阈值max abs ≤0.01）。
校准不使用答案。门检的合成短样本更新在正式训练前撤销。

正式训练继续使用原750条train、46条val、AdamW、lr=2e-5、1 epoch、
20261012随机种子与相同数据顺序；所以新增的主要实验变量是逐层前缀参数化。
前缀参数量与旧MLP并不完全相同，不能称“参数量严格相同”的对照。

## 安全与正确性

- `ptuning_v2_hybrid.py`：可卸载hooks、严格基模/架构/权重hash绑定的适配器、
  明确区分prefill与cached decode的贪心生成。
- `run_ptuning_v2_hybrid.py`：只读验证旧数据；真实27B门检→训练→46题生成。
- `launch_ptuning_v2_hybrid.py`：单GPU独立后台进程，不停止其他任务。
- 训练时注入上下文覆盖forward和backward，保证非重入checkpoint重算时注入仍有效。
- 仅支持无padding、batch size 1；不支持的输入报错，不静默继续。
- 所有20条证据必须逐字出现在实际prompt；超上下文直接拒绝，不截断。
- 每步将适配器、优化器、CPU/CUDA RNG提交到同一个独立checkpoint事务目录，
  最后原子更新latest指针；中断后不会把不同步的适配器和优化器配对。
- 缓存、临时文件、日志、checkpoint均使用数据盘；启动检查GPU空闲显存、RAM、磁盘。
- 进程显存上限预留至少1.5 GiB，不终止其他作业。
- 不新增API调用、标注、答案规则校验、生产变更。

## 门检

远端CPU真实小型Qwen3.5：6项测试通过：

1. 逐层校准logits一致、各层参数不共享、正文embedding不变。
2. 所有逐层参数有效梯度，checkpoint与普通反传一致，主干无梯度。
3. 改变前缀后，3步缓存解码与全重算一致，拒绝重注入已填充cache。
4. 适配器严格保存/重载和基模绑定，拒绝覆盖已有目录。
5. 异常退出恢复作用域、hooks移除、上下文预算守卫。
6. checkpoint适配器/优化器/RNG事务完整性与相同步数安全重放。

真实27B门检已通过，完整结果以远端 `gate.json` 为准：

- 初始浅层→逐层校准logits、保存重载一致。
- 合成短样本：全部64层收到梯度且全部80个参数块实际更新。
- 缓存decode与全重算对比，记录原浅层数值误差作参考。
- 最长完整20条证据的一次反向传播，CPU activation offload，不缩短输入。
- 主干参数及buffer前后逐字节checksum相同。

首次实测：初始logits max abs **0.0**；全部80个参数块收到有限非零
梯度并更新；3步cached/full-recompute的top1均一致，KL最大约5.82e-5。
最长样本为7473个prompt token、391个target token、完整20条证据；
其反传已通过，峰值16,405,768,704 bytes（约15.3 GiB），主干逐字节校验未变。
正式训练已完成首步且提交适配器/优化器/RNG checkpoint：
loss 0.712148、总体梯度norm 0.010665、最小逐层梯度norm 7.22e-5，
5560个prompt token、144个target token、20条完整证据，耗时约54.9秒。
这只是首步运行健康检查，不代表内容正确率；最新步数查看远端receipt。

## 首次启动

远端根：
`${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012`

实验目录：`ptuning_v2_deep_20261017_attempt01`

首次进程PID：1978211，GPU1。launch、log在实验目录旁；进度、门检、
训练run和checkpoints在实验目录内。PID仅为首次启动记录，判断存活应再查。

```sh
${REMOTE_PYTHON} launch_ptuning_v2_hybrid.py \
  --gpu 1 --out ${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012/ptuning_v2_deep_20261017_attempt01
```

断点恢复（只能对已有完整gate且尚未完成训练的实验使用；不能与存活进程并行运行）：

```sh
CUDA_VISIBLE_DEVICES=1 ${REMOTE_PYTHON} \
  run_ptuning_v2_hybrid.py --resume \
  --out ${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012/ptuning_v2_deep_20261017_attempt01
```

恢复时仍须将进程私有缓存放到数据盘；runner在导入训练依赖前自动设置。
不要更改代码后强行绕过identity hash恢复；保留原版本或开独立新实验。

## 评估口径

训练完成自动生成同一46道历史开发题，每题完整20条固定证据，原生协议，
最多1024输出token，不开答案规则校验；原始文本与token IDs全部保存。
这不是盲测，也不是检索全链路测试。语义正确率需要生成后单独人工复核，
不得用teacher-forced loss或JSON有效率代替正确率。

旧750数据浅层方案的内容复核是25/46=54.3%，4B SFT+KTO是20/46=43.5%。
新逐层方案目前未得出语义正确率；不预先承诺会提高。
GGUF服务端逐层适配器注入尚未实现或验收，不等于可直接部署到原GGUF服务。
