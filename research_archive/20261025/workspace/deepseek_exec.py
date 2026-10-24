#!/usr/bin/env python3
"""调用网关里的 DeepSeek 模型做执行，返回纯文本（代码 / 命令 / 方案）。

Claude 负责规划与落地执行，本工具让 DeepSeek 产出执行内容。

用法:
    python3 deepseek_exec.py "任务描述..."
    cat 任务.md | python3 deepseek_exec.py --model deepseek-v4-pro
    python3 deepseek_exec.py --system "你是……" --max-tokens 16384 "任务"

环境变量:
    ASA_GATEWAY_BASE    网关地址，默认 https://litellm.dex-gem.ai
    ASA_GATEWAY_TOKEN   网关 key，默认取 ANTHROPIC_AUTH_TOKEN
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("ASA_GATEWAY_BASE", "https://litellm.dex-gem.ai")
TOKEN = os.environ.get("ASA_GATEWAY_TOKEN", os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))

DEFAULT_SYSTEM = (
    "你是执行者：根据任务直接产出可落地的代码、命令或方案。"
    "不要寒暄、不要解释过程，输出内容必须能被直接使用。"
)


def call(model: str, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
    url = f"{BASE}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    # 若模型把结果放在 reasoning_content，且 content 为空，则退回 reasoning_content
    if not content.strip():
        content = msg.get("reasoning_content") or ""
    return content


def main() -> None:
    p = argparse.ArgumentParser(description="调用网关 DeepSeek 模型执行任务")
    p.add_argument("prompt", nargs="*", help="任务描述；省略时从 stdin 读")
    p.add_argument("--model", default="deepseek-v4-pro", help="默认 deepseek-v4-pro")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
    if not prompt.strip():
        p.error("需要提供 prompt 或 stdin 输入")
    if not TOKEN:
        sys.exit("缺少网关 token：请设置 ASA_GATEWAY_TOKEN 或 ANTHROPIC_AUTH_TOKEN")

    try:
        sys.stdout.write(call(args.model, args.system, prompt, args.max_tokens, args.temperature))
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:1000]}")
    except urllib.error.URLError as e:
        sys.exit(f"连接失败: {e.reason}")


if __name__ == "__main__":
    main()
