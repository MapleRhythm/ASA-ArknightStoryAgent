# ASA reranker teacher pipeline (answer_support_v2)

`gold_evidence`, `gold_evidence_ids`, `coverable_evidence_ids`, candidate
`covered_evidence_ids`, and candidate `relevance` are prohibited teacher inputs.
They may still be used in a held-out evaluation harness, but never to produce
teacher labels or training targets.

## Current v1 flow (retired for new labels)

1. The listwise QA dataset provides `query`, `answer`, and `gold_evidence`.
2. Hybrid retrieval collects up to 120 candidates.
3. Text overlap with `gold_evidence` writes candidate `covered_evidence_ids`
   and `relevance` into the raw candidate file.
4. The teacher receives the answer, all candidates, **and** `gold_evidence`.
   Candidate IDs and `original_rank` also expose retrieval order.
5. The teacher returns one 0–100 score and an E-ID assignment for every
   candidate. Validation checks structure but not textual support.
6. The existing trainer reads the old candidate `relevance`; it does not read
   `teacher.ranking`, so teacher labels do not directly affect its loss.

The 52 existing DeepSeek records are kept as v1 diagnostics and must not be
mixed with v2 outputs.

## New v2 flow

1. The production retrieval stack creates the fixed candidate pool before any
   teacher call: BM25 sparse retrieval + FAISS dense retrieval + MiniRAG graph
   retrieval, with the existing chapter/story scope, fusion, and configured
   neighbor/same-story supplements. The teacher neither generates nor expands
   this pool.
2. Export exactly 120 candidates per query. The current 649-record full120
   export is the frozen input for this run; do not rerun retrieval while a
   teacher job is in progress.
3. Migrate the frozen full120 export with
   `sanitize_reranker_teacher_inputs.py`.
   The output contains only query metadata, reference answer, candidate ID,
   kind, and text. Retrieval ranks/scores and document IDs stay outside the
   teacher input.
4. Reject any input that still contains gold or legacy labels.
5. Phase A asks the teacher to decompose only the reference answer into atomic
   claims (`Q1`, `Q2`, ...). The claim result has its own resumable JSONL cache.
   This is only a label-audit aid; it does not alter retrieval and is not an
   input feature of BGE.
6. Phase B sends all 120 candidate texts plus the frozen claims (not the prose
   reference answer again). Candidate
   order is deterministically shuffled and original `Cxxx` IDs are replaced by
   random-position aliases (`Pxxx`) so ranking is not leaked.
7. The teacher returns:
   - all 120 aliases ranked by each candidate's independent support strength;
   - detailed judgements only for direct/partial/contradiction evidence and up
     to 12 hard-context negatives;
   - a verbatim candidate quote for every direct/partial/contradiction label;
   - uncovered claim IDs.
   It must not demote duplicate support merely to interleave answer claims,
   because BGE scores each `(query, candidate)` independently.
8. Deterministic validation verifies the complete ID set, quote substrings,
   claim IDs, label invariants, and exact uncovered-claim consistency.
   Validation failures are repaired with feedback; HTTP 400/401/402/403 stops
   the job immediately rather than retrying the entire dataset.
9. `audit_answer_support_teacher.py` audits the full output. Gold remains
   available only for independent evaluation, never for label generation.
10. `materialize_answer_support_teacher.py` selects verified positives and hard
   negatives and creates compact listwise groups with `relevance` values. Claim
   information may balance sampling/weights, but it never becomes a retrieval
   input or a candidate-set-aware BGE feature.
11. `train_coverage_reranker.py` trains only on materialized groups and rejects
    raw teacher JSONL. Its default pair length is 1024, matching online
    inference; on the frozen 649 x 120 pool this truncates only 259/77,880
    pairs (0.33%).
12. At inference, BGE scores 120 independent `(query, candidate)` pairs in
    small batches (currently 8). A separate optional post-rerank selector may
    improve diversity/coverage; its behavior must not be distilled as if BGE
    could see the whole candidate set.
13. Evaluate on a held-out story-family split using answer/evidence recall,
    claim coverage, answerability/abstention, and end-to-end grounded-answer
    metrics. Do not choose a checkpoint using the training teacher scores.

## Commands

Run from the rank-v8 work directory.

```bash
# Sanitize the existing frozen production full120 export:
python scripts/sanitize_reranker_teacher_inputs.py \
  --input data/glm_5_3/input_full120/train_full120.jsonl \
  --output data/answer_support_v2/input/train_safe.jsonl

python scripts/label_coverage_candidates_with_deepseek.py \
  --input data/answer_support_v2/input/train_safe.jsonl \
  --output data/answer_support_v2/deepseek_flash/train_teacher.jsonl \
  --claims-output data/answer_support_v2/deepseek_flash/train_claims.jsonl \
  --model deepseek-v4-flash --max-workers 2

python scripts/audit_answer_support_teacher.py \
  --input data/answer_support_v2/deepseek_flash/train_teacher.jsonl \
  --expected-total 533 \
  --report data/answer_support_v2/deepseek_flash/train_audit.json

python scripts/materialize_answer_support_teacher.py \
  --input data/answer_support_v2/deepseek_flash/train_teacher.jsonl \
  --output data/answer_support_v2/train_groups.jsonl
```

API keys must be passed through environment variables and must not be written
to command lines, logs, scripts, or data files.
