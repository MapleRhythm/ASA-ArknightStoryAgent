# v53 attempt02: compatible fixed-span protocol

Canonical: `outputs/experiments/span_semantic_shadow_20260910.v53_attempt02`.
The complete original v53 remains intact, including its frozen preflight,
failed CPU integration log and GPU-idle rejection before any LLM call.

Attempt01 was NOT tested semantically. Its <=220-ish sentence groups could
violate existing quote<=80 / max2 references per fact. Attempt02 fixes only the
new protocol to satisfy those existing limits. It does not relax validators.
Full original evidence is partitioned losslessly into <=80-codepoint T spans.
T IDs are distinct from attempt01 S IDs. E headings are rendered once per
block for compactness; no source blocks or neighboring context are removed.

Fresh draft facts may select at most2 spans. The verifier may inspect all
source and select at most8 diagnostic spans, but a draft passes the shadow
gate only when all verifier-selected supporting spans were already selected
by the drafter. No citation correction/rebinding or regenerate-until-pass.

Reuse the frozen v53 runner explicitly, with a new preflight, source/model
guards, inputs, seeds, logs and fresh GPU idle check. Same 15 frozen-fact
checker calls, 6 fresh drafts, <=12 draft-checker calls; maximum33 actual calls.
Same 4B SODA LoRA, 12288 context / 2048 output and sampling. No default switch,
MiniRAG, new retrieval, training, service restart, production writes or agents.

Neither development source observations nor synthetic protocol verdicts are
human gold. Source-consistent but question-irrelevant statements remain a
separate category. No independent question/story-heldout dataset is claimed.
The unfilled review template remains in attempt01.
