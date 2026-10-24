# Isolated ownership-rule prompt comparison (v51)

Six v41 **control** retrieval/hypothesis inputs, two paired new seeds, baseline
versus one generic system-message addition: 24 first-round calls. The old
v39 routing-recovery candidate is not used. No fresh retrieval or planner.

Both conditions invoke the current canonical first-round reader and unchanged
validators. A narrowly scoped generator bridge changes only the outgoing
system text for the experimental condition. Evidence, user message, schema,
legacy chat/brace prefill, LoRA, model, budgets and sampling parameters remain
fixed. This is not a fresh E2E benchmark or an independent validation set.

No names, expected answers, reviewed source excerpts or historical model outputs
are inserted into prompts. The six questions were already used for development.
No retry, automatic E-ID repair, next round, final direct fallback, source
promotion, training, default-model switch, service restart, commit or push.

Every raw response and original failure remains in append-only receipts.
Action counts are not accuracy. Review every generated answer against the
actual visible source before any promotion decision.

Canonical repository is `${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`.
Use v49's exact installed-source mapping and a fresh >=60-second idle interval
inside the existing GPU2 lock. MiniRAG and subagents remain disabled.
