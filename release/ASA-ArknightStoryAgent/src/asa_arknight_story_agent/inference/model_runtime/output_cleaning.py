from __future__ import annotations

import re


LLAMA_TIMING_LINE_RE = re.compile(r"^\[\s*Prompt:.*\]$", re.MULTILINE)


def sanitize_generation_output(text: str, prompt: str) -> str:
    output = text.strip()
    if output.startswith(prompt):
        output = output[len(prompt):].lstrip()
    output = LLAMA_TIMING_LINE_RE.sub("", output).strip()
    output = re.sub(r"<think>.*?</think>\s*", "", output, flags=re.DOTALL).strip()
    output = re.sub(r"^warning:.*$", "", output, flags=re.MULTILINE).strip()
    output = re.sub(r"^(main|common_|llama_|load_|print_info:|system_info:|sampler ).*$", "", output, flags=re.MULTILINE).strip()
    output = output.replace("[end of text]", "").strip()
    return output
