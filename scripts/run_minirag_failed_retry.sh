#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TEACHER_API_KEY:-}" ]]; then
  echo "TEACHER_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

RETRY_DOC_IDS="data/processed/minirag_graph_annotations_v3/retry_failed_missing_doc_ids.txt"
OUTPUT="data/processed/minirag_graph_annotations_v3/chunk_graph_compact.jsonl"
FAILED_HISTORY="data/processed/minirag_graph_annotations_v3/failed_compact.jsonl"
TAG="$(date +%Y%m%d_%H%M%S)"
FAILED_RETRY="data/processed/minirag_graph_annotations_v3/failed_compact_retry_${TAG}.jsonl"
RAW_RETRY="data/processed/minirag_graph_annotations_v3/raw_compact_retry_${TAG}"

python - <<'PY'
import json
from pathlib import Path

output = Path("data/processed/minirag_graph_annotations_v3/chunk_graph_compact.jsonl")
failed = Path("data/processed/minirag_graph_annotations_v3/failed_compact.jsonl")
retry = Path("data/processed/minirag_graph_annotations_v3/retry_failed_missing_doc_ids.txt")

covered: set[str] = set()
if output.exists():
    for line in output.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        doc_id = str(payload.get("doc_id") or "").strip()
        if doc_id:
            covered.add(doc_id)
        for item in payload.get("doc_ids") or []:
            doc_id = str(item or "").strip()
            if doc_id:
                covered.add(doc_id)

missing: list[str] = []
seen: set[str] = set()
if failed.exists():
    for line in failed.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in payload.get("doc_ids") or []:
            doc_id = str(item or "").strip()
            if doc_id and doc_id not in covered and doc_id not in seen:
                seen.add(doc_id)
                missing.append(doc_id)

retry.parent.mkdir(parents=True, exist_ok=True)
retry.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
print(json.dumps({"retry_doc_ids": len(missing), "retry_file": str(retry)}, ensure_ascii=False))
PY

if [[ ! -s "$RETRY_DOC_IDS" ]]; then
  echo "No missing failed doc_ids to retry."
  exit 0
fi

PYTHONPATH=src python scripts/generate_minirag_graph_annotations.py \
  --documents indexes/arknights_story/documents.jsonl \
  --doc-ids-file "$RETRY_DOC_IDS" \
  --output "$OUTPUT" \
  --raw-dir "$RAW_RETRY" \
  --failed-output "$FAILED_RETRY" \
  --api-type anthropic_messages \
  --api-base "${TEACHER_API_BASE:-https://api.svips.org/v1}" \
  --api-key-env TEACHER_API_KEY \
  --auth-header x-api-key \
  --model "${TEACHER_MODEL:-MiniMax-M2.7-highspeed}" \
  --batch-size "${MINIRAG_RETRY_BATCH_SIZE:-35}" \
  --max-prompt-chars "${MINIRAG_RETRY_MAX_PROMPT_CHARS:-50000}" \
  --max-chunk-chars "${MINIRAG_RETRY_MAX_CHUNK_CHARS:-900}" \
  --relations-per-100-chunks "${MINIRAG_RETRY_RELATIONS_PER_100:-35}" \
  --relation-chars-per-target "${MINIRAG_RETRY_RELATION_CHARS_PER_TARGET:-220}" \
  --min-relation-yield-ratio 0 \
  --split-retries "${MINIRAG_RETRY_SPLIT_RETRIES:-3}" \
  --min-retry-batch-size "${MINIRAG_RETRY_MIN_BATCH_SIZE:-12}" \
  --max-output-tokens "${MINIRAG_RETRY_MAX_OUTPUT_TOKENS:-18000}" \
  --parallel "${MINIRAG_RETRY_PARALLEL:-2}" \
  --timeout "${MINIRAG_RETRY_TIMEOUT:-900}" \
  --api-retries "${MINIRAG_RETRY_API_RETRIES:-3}" \
  --retry-sleep "${MINIRAG_RETRY_SLEEP:-60}" \
  --request-interval "${MINIRAG_RETRY_REQUEST_INTERVAL:-5}" \
  --compact-schema \
  --resume

echo "Retry failed output: $FAILED_RETRY"
echo "Retry raw dir: $RAW_RETRY"
