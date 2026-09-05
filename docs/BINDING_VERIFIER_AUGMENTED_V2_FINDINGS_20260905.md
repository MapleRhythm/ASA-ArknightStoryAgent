# Binding verifier augmented v2 findings (2026-09-05)

## What changed

The verifier-pair builder now has two opt-in controls:

- `--ambiguity <jsonl>` applies the third-pass GLM adjudication.  `supported_by_union`
  and `supported_by_some` retain their adjudicated E-ID set; `ambiguous` and
  `unsupported` are excluded.
- `--include-suspected-missed-positives` adds candidates that the hard-negative
  audit explicitly judged `supported` as additional positive variants.

The default invocation is unchanged, so the old dataset remains reproducible.
The new 2026-09-05 dataset is stored on 248 at:

`/mnt/store/zhb/exx_grounding_v1/data/binding_verifier_pairs_augmented_v2_20260905/`

It contains 766 pairs (691 train / 75 internal eval after the trainer's
question-group split), including 600 gold pair rows and 166 pair rows generated
from 73 GLM-confirmed missed-positive variants.  These variants are paired only
against GLM-confirmed unsupported candidates.

## Training

Candidate model:

`/mnt/store/zhb/exx_grounding_v1/models/binding_verifier_reranker_augmented_v2_20260905/`

Configuration:

- BGE reranker rank-mix-v6 base
- max length 1536 (the old 1024 setting truncated about 20% of the held-out
  pairs)
- weighted pairwise softplus
- learning rate `1.5e-5`, two epochs, batch size 2, accumulation 4
- A100, 174 optimization steps, no OOM

Training-state optimizer files were removed after completion; model weights and
the `checkpoint-174` weights were retained.

## Independent held-out result

The comparison uses the exact 17 question groups selected by the trainer
(75 pair rows).  The hard-negative file was filtered to the same 16 source
records for the missed-positive test.

| model | clean-pair accuracy | clean mean margin | missed-positive accuracy | missed-positive AUC |
|---|---:|---:|---:|---:|
| base rank-mix-v6 | 69.3% | 3.11 | 75.5% | 0.657 |
| old binding verifier v1 | 100.0% | 14.76 | 85.7% | 0.805 |
| augmented verifier v2 | 100.0% | 14.52 | **93.9%** | **0.904** |

The clean-pair set is saturated and therefore not a useful release gate.  The
missed-positive result is the meaningful signal: v2 improves over v1 by
8.2 percentage points and raises AUC by 0.099 on held-out question groups.

On a separately reconstructed listwise pool (37 held-out records, with all
GLM-confirmed positives and negatives present), both v1 and v2 reached 100%
Top-1/Top-3/Top-5.  v2's positive-vs-best-negative margin was 13.02 versus
13.23 for v1.  This small, saturated listwise slice shows no regression, but
does not by itself justify a production switch.

## Release caution

This is still a candidate, not yet the production default.  Before switching
the runtime:

1. run the full listwise retrieval recall evaluation with v1 vs v2;
2. check runtime latency with `reranker_max_length=1536`;
3. run a blind manual/GLM audit on newly promoted top-k evidence;
4. keep v1 as an immediate rollback.

The 1536 setting must be carried into the runtime configuration; leaving the
runtime at 1024 would discard much of the training change.
