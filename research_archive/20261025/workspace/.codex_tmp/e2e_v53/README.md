# v53 exact-span + semantic shadow prototype

Canonical target: `outputs/experiments/span_semantic_shadow_20260910.v53`
in `${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`.
Local files are development/transfer copies, not the canonical repository.

No release-source changes, services, training, MiniRAG, new retrieval, default
model switch, subagents, commits, pushes or automatic E-ID rebinding.

Two separate tracks:

1. Shadow verification of the six v50 assistant-authored contrast claims and
   three recorded erroneous model facts. These are deliberately selected
   development cases, not an independent benchmark or human gold.
2. Fresh drafts on the same three questions under a NEW exact-span protocol.
   The model chooses span IDs; code constructs citations BEFORE canonical
   validation. A separate same-model call checks each new claim. The answer
   must pass both unchanged canonical validation and the semantic shadow gate.
   No retry, quote repair, citation rebinding or selective fact rewriting.

The verifier uses the same existing 4B SODA LoRA in a separate call. It is not
an independent model or an authoritative judge. Original answers/references
remain immutable; verification findings never change their live output.

Every visible source character is retained within its original E block;
sentence grouping is mechanical. Speaker labels remain original text; no
heuristic speaker/coreference ownership is asserted. Full neighboring context
is shown, even if it contains counterevidence. Span IDs are scoped by a
source-SHA-derived packet identity; full hashes and offsets are retained.

Pre-register prompts, model/tokenizer/config/source hashes, six projection
labels and three raw facts. Six fixtures × two seeds plus three historical
facts × one seed = 15 checker calls. Three questions × two seeds = 6 fresh
drafts, with at most two checker calls per draft: maximum 33 total LLM calls.
No old correct answers/source-review comments/expected labels enter prompts.
Same 12288 context / 2048 output limits; overflow is recorded, never truncated.
Use GPU2 only after fresh >=60-second idle window inside the existing lock.

Metrics: per-claim predictions versus explicitly assistant-reviewed reference
expectations; preserve supported-but-irrelevant cases separately. Report
unsupported claims passed, source-consistent claims withheld, parser failures,
uncited-context failures and latency. Do not call action counts accuracy.

There is no new human-reviewed, question/story-disjoint test set available in
this experiment. A held-out review template is delivered, but cannot justify
production promotion until independently sourced and reviewed. Do not rename
old fixed58 questions as unseen holdout or model-generated labels as human gold.
