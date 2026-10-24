# DeepSeek-flash 小样本试测：中止，未产生模型答案

日期：2026-09-12（Asia/Shanghai）。

## 状态

- 目标模型：`deepseek-flash`
- 官方接口：`https://api.deepseek.com/chat/completions`
- 预定：复用GLM/4B已冻结的14题、oracle与BM25 Top-3两种证据，共28次请求。
- 实际发送：仅写入第1条 `request_started` 标记；没有 `request_finished`，
  没有HTTP响应和模型答案。
- 原因：通过SSH伪终端输入凭据时发生回显；为避免继续使用已暴露凭据，
  立即中止进程。远端PID已退出。
- 账单状态：未验证；由于没有记录到HTTP响应，不能声称“未计费”。
- 没有重试、没有切换线上模型、没有改训练数据、检索、服务或配置。

## 安全修正

本次之后不再使用该凭据。下一次若继续，必须使用新凭据，并从非TTY
管道输入；代码在读取前拒绝TTY、普通文件、空值、空白和超长输入。
本地7项安全输入测试通过。

## 封存与回传

248：
`${REMOTE_STORE}/exx_grounding_v1/experiments/asa_v58_deepseek_pilot_20260912`

241：
`${REMOTE_ASA_ROOT}/rank_v8_coverage/experiments/asa_v58_deepseek_pilot_20260912`

中止归档已在两台服务器保存并核验：
`interrupted_receipt.tgz`

归档SHA-256：
`0cd80e5e50c964b42ae747a64edaf1cc47f2c55c2931932c13b821869a5c656a`

没有DeepSeek分数。前一轮GLM的28题成功结果仍然有效，但不能用于推断
DeepSeek表现。
