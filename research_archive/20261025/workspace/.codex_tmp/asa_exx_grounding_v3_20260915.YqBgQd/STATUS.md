# In progress — V3 grounding fix

User asked to fix Exx false rejection and failed-answer recovery.
No public push/API change/new training/paid API. No subagents.
Original worktrees and v1 release preserved.

V2 local: ../asa_exx_grounding_fix_20260915.jKfyIG
V2 trial finished8/8: onlyQ6 correct, Q10 partial, other6 reject.
V2 not promoted.30 calls,28stop/2length;515.74 seconds. All failures retained.

V3 this directory adds narrow discourse bridges (no entity/answer hardcoding)
on top of V2's bounded repair and reference-safe partial preservation.
359 local tests passed with MiniRAG fixture environment supplied.
Saved40-output replay: original first candidates Q8/13/18 now pass, known
wrong Q11/19 still rejected. This is not a fresh20-question score.

Remote stage: /tmp/asa-exx-grounding-v3.Yp4jqp
Remote release: ${REMOTE_HOME}/ASA-ArknightStoryAgent/releases/exx-20260915-v3
SSH socket: /tmp/asa-exx-fix-20260915.sock
Remote full lock verification passed.
Fresh full-chain8: Q6/8/10/11/13/16/18/19; GPU1; PID1368414.
Out: ${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/exx_grounding_fix_smoke8_20260915_v3
Log: /tmp/asa-exx-grounding-v3.Yp4jqp/smoke8.log.
First3 completed: Q6/Q8 correct, Q10reject. Default still v1.

Next: finish8, download/audit, semantic review against original rubric.
If Q8/13/18 resolved and no new major-error deliveries, promote_fix.py
can CAS-switch default to v3 with exact doc backups. Needs a manually
reviewed review.json matching result SHA and qids.
Do not claim60% or convert8 selected development questions into a20Q score.
Then install rollback_exx_20260915_v3.py without overwriting an existing file,
verify all default entry aliases, copy report to acceptance, check hashes,
confirm own processes ended/GPU freed, close SSH and final Chinese handoff.
