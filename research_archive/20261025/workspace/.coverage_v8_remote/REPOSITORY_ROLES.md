# ASA repository roles

Effective 2026-08-19:

- `${REMOTE_USER}@${REMOTE_HOST}` is the canonical ASA repository and data source.
  Canonical code, fixed retrieval candidates, API teacher labels, checkpoints,
  manifests, evaluation reports, and resumable state are created and retained
  there first.
- `${REMOTE_USER}@${REMOTE_HOST}` is a disposable training mirror. It receives immutable,
  versioned training inputs and code snapshots from 241, runs training or
  evaluation, and returns only explicitly selected artifacts to a new staging
  path on 241 for review.

## Direction rules

1. Normal synchronization is one way: `241 -> 248`.
2. Never use `--delete` when synchronizing to 248.
3. Never edit teacher JSONL, candidate pools, or manifests on 248.
4. Never resume API labelling on 248.
5. Training outputs from 248 must not overwrite canonical paths on 241. Copy
   them to `incoming_from_248/<run_id>/`, verify checksums and metrics, then
   promote explicitly on 241.
6. Every training snapshot must include a manifest containing source paths,
   SHA-256 checksums, record counts, protocol/prompt version, creation time, and
   an immutable `run_id`.
7. Existing historical files on 248 are retained as diagnostics only unless a
   canonical 241 manifest explicitly adopts them.

## Current canonical working root

`${REMOTE_ASA_ROOT}/rank_v8_coverage`

Current teacher protocol: `answer_support_v2.2_pointwise`.
