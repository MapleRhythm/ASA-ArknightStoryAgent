# Current LoRA referenced-data contract audit (v52)

Read-only CPU audit of the current model card, matching KTO configuration, and
the currently registered merged train/validation JSON files. Do not run the
training configuration, rebuild data, modify labels, load training_args.bin,
or change the default model. MiniRAG stays disabled.

Distinguish current reference consistency from training-time content proof:
matching names/configuration do not prove the current data bytes were used by
the historical training run. The configuration also names an earlier LoRA;
this audit is not the complete model-training history.

Measure target schema separately for positive and negative KTO examples.
Never treat rejected examples or stale meta.preference_role as positive labels.
Compare question/prompt overlap using actual text/digests, not numeric ID suffixes.
Do not equate validation loss, labels, schema compliance or overlaps with
semantic accuracy, human gold or causation of present model mistakes.

Canonical workspace:
`${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_HOME}/ASA-ArknightStoryAgent`.
Guard the already-installed v49 trace fix and all earlier protected files.
Preserve all historical manifests and data. No GPU/model/retrieval calls.
