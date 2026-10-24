# Exx grounding fix — 2026-09-15

Parent: frozen exx-20260915-v1. Never modify the parent or its scores.

Scope: conservative surface normalization, bounded same-evidence repair before
retrieving again, and explicit partial-answer preservation. Do not disable
grounding, introduce answer keys into prompts, train, or call paid APIs.

V2's8-question trial did not resolve the target false rejections and will not
be promoted. V3 adds narrow discourse bridges: existing source speaker labels,
reported epistemic phrases, question temporal scope, and rule-topic frames
only when the entire factual rule body is literally in the cited evidence.
No character-coverage threshold is loosened. These are heuristic grounding
checks, not independent semantic verification. V2 remains preserved.

Acceptance:

- Unknown IDs, invisible evidence and unsupported relations remain rejected.
- A repair uses exactly the original visible evidence/IDs; no new retrieval.
- Maximum one repair per question. Strict JSON, no silent output recovery.
- Previously accepted facts cannot be silently dropped by the repair.
- Partial answers retain abstain status and are never scored as full success.
- Cross-round partial references must be rebound by exact evidence text and
  revalidated, not reused by transient E number.
- Replays and new inference are reported separately from original full-chain
  scores. Improvements on these development questions do not certify 60%.
- Stage as a new release; switch the default only after tests and small real
  inference validation. Preserve rollback to exx-20260915-v1.
