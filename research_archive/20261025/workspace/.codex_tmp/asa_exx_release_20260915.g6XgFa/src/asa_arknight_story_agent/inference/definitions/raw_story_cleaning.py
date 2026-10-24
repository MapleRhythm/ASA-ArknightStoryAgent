from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta


def raw_line_context(lines: list[str], index: int, *, window: int = 2) -> str:
    start = max(0, index - window)
    end = min(len(lines), index + window + 1)
    return "\n".join(line.strip() for line in lines[start:end] if line.strip())


def clean_raw_story_context(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        sticker_texts = re.findall(r'text="((?:\\.|[^"\\])*)"', line)
        if sticker_texts:
            for sticker_text in sticker_texts:
                sticker_clean = (
                    sticker_text.replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\t", " ")
                    .strip()
                )
                if sticker_clean:
                    cleaned_lines.append(sticker_clean)
            continue
        line = re.sub(r'^\[name="([^"]+)"\](.*)$', r"\1：\2", line)
        line = re.sub(r"\[[^\]]+\]", "", line).strip()
        if line:
            cleaned_lines.append(line)
    return strip_internal_evidence_meta("\n".join(cleaned_lines)).strip()
