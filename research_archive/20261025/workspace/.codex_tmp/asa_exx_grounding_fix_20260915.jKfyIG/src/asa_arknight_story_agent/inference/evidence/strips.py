from __future__ import annotations

import re

from asa_arknight_story_agent.inference.common.patterns import LINE_SPLIT_RE


def split_evidence_strips(text: str, *, max_strips: int) -> list[str]:
    strips = [
        re.sub(r"\s+", " ", item).strip()
        for item in LINE_SPLIT_RE.split(text)
        if re.sub(r"\s+", " ", item).strip()
    ]
    return strips[:max_strips]
