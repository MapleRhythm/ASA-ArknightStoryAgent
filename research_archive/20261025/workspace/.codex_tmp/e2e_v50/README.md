# Relation-claim sensitivity audit (v50)

CPU-only, post-hoc development audit of three observed displayed errors.
Canonical workspace: `${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`.

Replay the actual canonical reader paths for 007/control/seed1,
017/candidate/seed0 and 049/control/seed0 with their recorded raw responses,
exact prompts, prepared evidence, kwargs, conclusions and answers. Transparently
capture validator inputs; do not change any validation return value.

Then evaluate three pairs of explicitly assistant-authored, single-fact
projections against the same real evidence and references. These projections
are diagnostic fixtures, not model outputs or proposed answers to the original
questions. In particular, a fact about Juanna need not answer a Pramanix question.
Keep separate invalid-E-ID and absent-quote controls.

Use v49's exact installed-source mapping without rewriting older manifests.
Do not change production, aliases, prompts, labels or historical outputs.
No new LLM/retrieval/GPU, training, model switch, restart, commit or push.
No automatic E-ID rebinding, weakened validator, correctness rate or human-gold
claim. Source excerpts and assistant observations are only post-hoc audit data.
