# Clean-binding SFT retry and retrieval-reward findings (2026-09-04)

## Clean-binding SFT retry

The retry run used the recalibrated binding set (910 train rows, 95 validation
rows), initialized from `exx_grounding_v1_sft_success559`, with LoRA rank 16,
learning rate `2e-6`, one epoch, and `cutoff_len=8192`.  The original 12k run
OOMed at step 49; this run completed all 114 steps on the 248 A100 without
OOM.

Model output:

`/mnt/store/zhb/exx_grounding_v1/models/exx_binding_clean_sft_v2_a100_r16_lr2e6_e1_retry8192_20260904`

Training metrics:

- train loss: `0.6742`
- final eval loss: `0.6892`
- one intermediate checkpoint: `checkpoint-100`
- final checkpoint: `checkpoint-114`

### Baseline vs clean-SFT (79 grounded validation rows)

| metric | baseline | clean-SFT | delta |
|---|---:|---:|---:|
| strict JSON | 92.4% | 97.5% | +5.1 pp |
| schema valid | 57.0% | 77.2% | +20.3 pp |
| action accuracy | 69.6% | 74.7% | +5.1 pp |
| schema + action | 41.8% | 59.5% | +17.7 pp |
| duplicate fact rate | 34.2% | 17.7% | -16.5 pp |
| premature answer | 16.5% | 12.7% | -3.8 pp |
| exact evidence set (gold answers) | 27.4% | 25.8% | -1.6 pp |
| evidence Jaccard (gold answers) | 0.485 | 0.520 | +0.035 |
| claim-citation alignment (gold answers) | 0.174 | 0.199 | +0.025 |
| over-abstain | 5.1% | 8.9% | +3.8 pp |
| generation truncated | 0.0% | 2.5% | +2.5 pp |

Action accuracy by gold action changed as follows:

- `answer_directly`: 83.9% → 85.5%
- `retrieve_more`: 0.0% → 8.3%
- `abstain`: 60.0% → 100.0%

The net action gain is real (8 improvements versus 4 regressions), but three
answerable questions became abstentions.  The remaining regression is a
degenerate `retrieve_more` completion that repeatedly emits the same entities
and keywords until the 768-token limit.  This is a protocol/data problem to
fix before RLVR, not a reason to discard the clean-SFT adapter.

Evaluation artifacts:

`/mnt/store/zhb/exx_grounding_v1/eval/exx_binding_clean_sft_v2_a100_r16_lr2e6_e1_retry8192_20260904_val`

The watcher now uses the batched vLLM generator.  With `CC=/usr/bin/gcc-11`,
`CXX=/usr/bin/g++-11`, and `CUDAHOSTCXX=/usr/bin/g++-11`, 79 rows took 61.6 s
for baseline and 73.3 s for clean-SFT.  The old Transformers watcher was
stopped after producing no prediction file (only 1/79 rows in several
minutes).

## B1+B2 retrieval reward validation

The validation script evaluated 30 hard-pool queries; 26 had at least one
uncovered gold segment.

| follow-up strategy | mean coverage gain | mean novelty | positive-gain rows |
|---|---:|---:|---:|
| repeat original query | 0.000 | 0.000 | 0/26 |
| query built from one missing segment (oracle) | 0.923 | 0.821 | 16/26 |

Interpretation: a retrieval RL reward must be conditioned on the uncovered
evidence set.  Rewarding generic query rewriting provides no measurable B2
coverage gain; a gap-driven query has a strong signal.  The oracle result is
an upper bound, not a deployable query generator.

Validation log:

`/mnt/store/zhb/exx_grounding_v1/logs/retrieval_reward_validate_20260904.log`

## Recommended next step

Start a small RLVR smoke from the clean-SFT adapter only after adding:

1. a repetition/length penalty for `follow_up_hypothesis.entities` and
   `keywords`;
2. a hard cap on follow-up arrays and a truncation-aware reward;
3. a local 4B binding-verifier reward (GLM remains offline-only);
4. gap-conditioned B1+B2 reward using the uncovered-E-ID set.

Compare the smoke adapter against both the original baseline and clean-SFT on
the same 79-row validation set before launching a full RL run.
