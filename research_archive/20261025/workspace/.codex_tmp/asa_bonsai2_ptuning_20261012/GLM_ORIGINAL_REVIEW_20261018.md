# Original-label GLM review, positions 301–486

Launched on 2026-10-18 at 23:07:08 Asia/Shanghai.

- Remote host: `${REMOTE_USER}@${REMOTE_HOST}`.
- Remote experiment root: `${REMOTE_ASA_ROOT}/bonsai2_ptuning_20261012`.
- Run: `glm_original_review_301_486_20261018`.
- Launch PID: `3500600` (launch identity, not a permanent liveness claim).
- Script: `glm_review_original_20261018.py`.
- Original source: `READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl`.
- Original source SHA256: `5fc6e331c9d02a24003c0aa89ea9b402c08f1b6907d3fb624bf31961f73a8ca4`.
- Scope: exactly original positions 301–486 inclusive, 186 examples.
- Provider configuration: existing Coding Plan transport, `glm-5.3`, no standard-endpoint fallback.
- Inputs: question, round, original target, and all 20 complete evidence passages.
- Outputs: semantic judgments/issues, receipts, SQLite recovery ledger; no label overwrite.

The first example must produce a valid API review before the rest run.
A semantic rejection is a valid preflight response, not an API failure.
All issue-bearing reviews are quarantined even if both boolean checks are true.
API failures stay pending and are not mislabeled as semantic failures.
The batch has two workers, two attempts per call, a 400-call / 6M-accounted-token
cap and the inherited failure circuit breaker. One bounded second pass may retry
pending tasks. No automatic training or promotion is implemented.

Inspect `status.json`, `preflight.json`, `reviews.json`, `api_receipts.jsonl`
and `run.log` under the remote run directory. Do not use the base annotation
exporter: this is review-only and deliberately does not generate new labels.
To resume after inspecting a stopped process, invoke this script's `run` command
with the same source and output paths. `launch` refuses an existing output
directory; the exclusive dataset lock prevents duplicate runners.

Local and remote new-wrapper tests: 3 passed before launch. These are offline
tests, not proof of API availability or label correctness.

Confirmed API preflight at 23:09:11 Asia/Shanghai: original position301
returned a valid semantic review and was accepted (1 call,12,867 tokens).
`preflight.json` records this. The remaining185 tasks then enter the worker
pool. This is a timestamped partial result, not a full-batch quality score.
All23 related local regression tests also passed.

This GLM review is distinct from the earlier assistant's offline review and is
not independent human gold. Earlier ordinary-endpoint insufficient-balance
records do not establish whether Coding Plan is currently available.

## Follow-up, 2026-10-19

At01:25 Asia/Shanghai the original process had exited. Final original-run state:
184 accepted,1 quarantined (position330),1 pending (position472),190 calls,
1,504,639 accounted tokens. Stop reason: `pending_tasks_require_inspection`.
Position472 returned `finish_reason=length` twice at8192 output tokens;
these were completed API responses, not network timeouts or semantic judgments.

An isolated recovery launched at01:26:23, PID3639787:
`glm_original_review_472_recovery_20261019`, using
`glm_review_length_recovery_20261019.py`. Only position472 is included.
It keeps the same Coding Plan model, full20 evidence and review prompt,
raising only the output cap to16384, with2 API calls /150,000 accounted-token
limits. Original-run results and settings are untouched.

Downloaded the original run JSON/receipts locally and verified all185
completed reviews against original target hashes, source positions, exact
full20 review prompt cache keys, successful response endings and receipt IDs.
Position330 remains a GLM-flagged case, not yet an assistant-confirmed error:
review flags the unsupported `重建` search keyword and warns against losing
the already-supported first subquestion. Do not report184/186 as model QA
accuracy or claim all486 labels have been reviewed. Training is not released.

At01:28:27 the isolated position472 recovery finished successfully:
`finish_reason=stop`,9,717 completion tokens (9,471 reasoning tokens),
15,180 total tokens, empty issues and both semantic booleans true.
Effective combined result:186 reviewed,185 accepted,1 quarantined,0 pending.
Across both runs:191 calls and1,519,819 accounted tokens.
The original run's pending status is intentionally preserved; consult the
separate combined summary for effective coverage.
