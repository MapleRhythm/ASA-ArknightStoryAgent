from __future__ import annotations


def render_qwen_chat_prompt(system_prompt: str, user_prompt: str, *, assistant_prefix: str = "{") -> str:
    return (
        "<|im_start|>system\n"
        + system_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user_prompt.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + assistant_prefix
    )
