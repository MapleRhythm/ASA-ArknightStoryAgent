# Exx正式默认链路 — exx-20260915-v1

固定项：

- 2026-09-11完成的Exx SFT v55模型，Qwen3.5-4B基座。
- LoRA SHA256: `0adb413f5ba8b40423341cfd0f1e17f2624684c43e9ff8c7035f2ba115426dd1`。
- `grounded_action_exx_v1`：模型生成事实和 `evidence_ids`，不复制quote。
- 程序按E编号回查本次可见正文，使用现有词面校验，并由facts重建答案；
  不代表已经实现独立语义蕴含验证，也不自动生成逐字引文。
- 每次前20条完整检索chunk，不用链文本替换、不CRAG、不pinning、不MMR；
  所有入模证据逐项保护，缺失即报错。超token容量报错，不静默缩减预算。
- 源码/config/tokenizer/基座权重/adapter使用 `RELEASE_LOCK.json` 核验。
- 本地GPU推理，无付费API；固定两轮，结论预算1536，context10000。

## 远端正式入口

在 `${REMOTE_HOME}/ASA-ArknightStoryAgent`：

```bash
${REMOTE_PYTHON} scripts/run_exx_official.py --verify-only
CUDA_VISIBLE_DEVICES=0 bash scripts/run_exx_official.sh "泡普卡为什么来到罗德岛？"
```

无显式历史配置的 `scripts/run_cpu_inference.py` 与旧名称
`scripts/run_latest_kto_interactive.sh` 都转入此正式入口，不再默认旧SODA/KTO。
`configs/runtime_inference.json`、`runtime_inference_gpu.json`、
`runtime_exx_official.json` 是同一锁定配置的别名。
独立持久版本目录为 `releases/exx-20260915-v1`，默认指针为 `current-release`。
需要实验覆盖模型/协议时，请显式调用版本目录中的通用CLI，不能把实验当成正式默认结果。

## 回滚与范围

```bash
${REMOTE_PYTHON} scripts/rollback_exx_20260915.py
```

回滚逐项检查文件未被后续修改，然后恢复发布前文件；新模型、版本目录、评测记录
都保留。备份在 `outputs/release_backups/exx-20260915-v1`。
旧模型/旧quote协议未删除，历史配置仍可显式用于复现。

本次“正式”指用户授权固定的远端默认执行版本，不等于准确率≥60%的认证。
质量结果见20题回归报告；它是历史开发题，不是线上盲测。
未推送GitHub/Hugging Face，未重新训练，也未替换独立API服务进程。
