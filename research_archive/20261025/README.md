# 2026-10-25 public research source archive

This is a **sanitized, historical research-source archive**, not a runtime installation package and not a production integration. It preserves 1,948 approved public research-code, protocol, test, schedule, and report files under `workspace/` using their original relative paths. `PUBLIC_MANIFEST.jsonl` records each archive-relative path, publication purpose, SHA-256, byte size, and any fields sanitized before publication.

The archive retains 114 P-tuning source reports for historical traceability. Those experiments remain historical research artifacts: they do not change the production runtime, release configuration, or deployed model.

## What is deliberately absent

- 1,343 private raw artifacts (about 2.27 GB) remain in the local persistent publication area and have **not** been uploaded to Hugging Face.
- The final 180-item dataset, two adapters, and the 1117-step training runs are still remote-only; this archive does not fabricate local copies.
- Raw prompts, predictions, evaluation/training rows, receipts, logs, credentials, login material, and third-party full-text literature are outside this archive.

See [`NOTICE.md`](NOTICE.md) for rights and packaging boundaries, the repository [handoff document](../../docs/PI_HANDOFF_20261025.md) for current status, and [`PUBLIC_MANIFEST.jsonl`](PUBLIC_MANIFEST.jsonl) for integrity verification.
