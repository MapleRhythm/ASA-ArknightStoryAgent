"""Fail closed if a full-document request loses any selected evidence."""
import re

from asa_arknight_story_agent.inference.common.text_utils import strip_internal_evidence_meta


def validate_full_document_prompt(pipeline, prompt, evidence):
    if not getattr(pipeline, "prompt_evidence_require_full_documents", False):
        return
    normalized = re.sub(r"\s+", " ", prompt).strip()
    for item in evidence:
        document = item["document"]
        body = re.sub(r"\s+", " ", strip_internal_evidence_meta(document["clean_text"])).strip()
        if not body or body not in normalized:
            raise ValueError(f"Full evidence missing from prompt: {document['id']}")
    labels = re.findall(r"(?m)^\[(E\d+)\]", prompt)
    if labels != [f"E{i + 1}" for i in range(len(evidence))]:
        raise ValueError("Full-evidence prompt has missing, duplicate or reordered evidence labels.")
