# In progress — 2026-09-15 Exx grounding repair

User asked to fix false lexical rejection and loss of useful answers.
Parent frozen release and prior scores are untouched.

Local staging: this directory. Remote stage:
`/tmp/asa-exx-grounding-fix.dmM6HD`.
SSH socket: `/tmp/asa-exx-fix-20260915.sock`.
New release: `${REMOTE_HOME}/ASA-ArknightStoryAgent/releases/exx-20260915-v2`.
Default remains v1 until review.

Implemented:

- Narrow surface normalization; no blanket disabling of grounding.
- One repair per question using exactly the same 20 evidence bodies and IDs.
- Original candidates and independently passing facts retained in diagnostics.
- No silent deletion of locked facts or reduction of candidate fact count.
- Explicit partial-answer preservation, still `abstain`, including exact-text
  cross-round reference rebinding and revalidation.
- Remove duplicate Exx final fallback which repeated essentially the same prompt.

Checks: all348 local tests passed (special MiniRAG fixture environment supplied).
40 saved output checks (first/last of20) completed: old/new validators both
accept final Q1/Q6; this is compatibility, not a new accuracy score.
90 focused tests passed including18 new cases.

Remote v2 full lock verification passed. Targeted fresh full-chain8 started:
Q6/8/10/11/13/16/18/19, GPU1, PID1362957.
Output: `${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/exx_grounding_fix_smoke8_20260915_v2`.
Log: `/tmp/asa-exx-grounding-fix.dmM6HD/smoke8.log`.
No paid APIs, new training, public push or service restart.
