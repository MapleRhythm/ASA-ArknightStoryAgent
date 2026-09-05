# Calibrated binding evaluation (2026-09-05)

## Scope

This evaluation re-scores the frozen `clean_sft` and `gap_mix` predictions
against the GLM strict recalibration changes.  It does not overwrite the
original gold, predictions, or legacy metrics.  A second run also applies the
ambiguity sidecar; `ambiguous` facts are excluded from hard binding matching.

The evaluator is:

`scripts/evaluate_exx_calibrated.py`

The remote artifacts are stored under:

`/mnt/store/zhb/exx_grounding_v1/eval/exx_binding_gap_mix_v3c_calibrated_20260905/`

## Results

There are 79 aligned rows and 62 calibrated answer rows.  The strict-only and
ambiguity-aware runs have the same aggregate values here because the 86
contradiction rows are not present in this 79-row frozen validation subset.

| metric | clean-SFT | gap-mix | gap-mix minus clean |
|---|---:|---:|---:|
| action accuracy | 74.68% | 73.42% | -1.27 pp |
| exact calibrated E-ID set | 29.03% | 27.42% | -1.61 pp |
| mean calibrated E-ID Jaccard | 0.513 | 0.510 | -0.002 |
| mean fact-text similarity | 0.388 | 0.375 | -0.013 |
| mean claim-local binding score | 0.616 | 0.595 | -0.021 |

The calibrated reference improves the absolute clean-SFT exact-set score from
27.42% to 29.03%, but does not change the model ordering.  This is evidence
that the legacy gold metric was somewhat pessimistic, not evidence that
gap-mix has a hidden binding gain.

## Interpretation

1. `gap-mix` remains better for schema and duplicate-fact control, but worse
   on action and calibrated binding metrics in this frozen set.
2. The result still cannot establish an SFT ceiling: the set has only 58 unique
   question families, and the calibrated metric still does not score alternate
   valid evidence unless it is present in the recalibrated reference.
3. A semantic verifier or human audit is required for the final claim-support
   decision.  The local verifier should be used as a diagnostic/ranking signal,
   not as a new gold label.

## Runtime correction

Both GPU runtime configs now set `reranker_max_length` to `1536`, matching the
augmented v2 training/evaluation protocol.  The old `1024` setting would
silently truncate part of the pair input and invalidate a direct v1/v2
comparison.

## Recommended next experiment

Before any RLVR run:

1. Run a blind semantic audit on all rows where clean-SFT and gap-mix differ.
2. Score each predicted fact against its own cited evidence, not only against
   the calibrated E-ID set.
3. Build a family-held-out set with no duplicate question keys.
4. Re-evaluate reranker v1/v2 with candidate-pool recall, final-prompt recall,
   and warm p50/p95 latency.

