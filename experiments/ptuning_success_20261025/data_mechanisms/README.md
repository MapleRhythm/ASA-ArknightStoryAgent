# Question-slot-complete label diagnosis

The intervention candidate is **stop supervision after all requested slots are answered**, while preserving actor/recipient, event stage, attribution and local uncertainty. A uniform fact cap cannot represent this: a five-fact answer can be necessary, while a fourth corroborating fact can extend an already complete answer.

## What the available evidence supports

This directory contains an offline selector, aggregate inventory and abstract tests. Original evidence and row-level judgments stay in the task's private evidence directory. No existing labels, review state, training data or production code are modified.

| Available dataset | Rows | Answer / retrieve / abstain | Target character median | Scope |
|---|---:|---:|---:|---|
| Historical input-failure snapshot | 171 | 158 / 9 / 4 | 300 | Not used by the two main adapters; not the final repaired 180 |
| Historical reader-error source | 486 | 368 / 94 / 24 | 314 | Includes known failures excluded by curated selection; not the full 1117 |
| Base development labels | 46 | 41 / 1 / 4 | 431.5 | Reused development questions, not an independent test |

In the historical 171, the 28 `minimal_complete` targets have 2–4 facts; 26 have 3–4. All 29 `partial_support` targets use `answer_directly`, consistent with preserving known parts. These counts describe serialization, not semantic error or redundancy rates. None establish a cause of the two adapters' behavior.

Risk-stratified full-evidence reading of six cases found:

- A three-slot stage question was completely covered by facts 1–3. Fact 4 corroborated slot 1 using another scene. That is a **supported scope extension**, not a factual-error judgment. It is a candidate for reviewed stop-supervision, not quarantine. This example is from historical 171, not either main adapter's training or final repaired 180.
- A contrasting five-fact target needed four attributed allegations plus the respondent's denial of anticipating one event. Deleting the later facts would lose required content or erase the allegation/fact distinction. This is a historical 486 example, not proof of full1117 membership.
- A four-fact multi-actor target correctly bound names, addressee and affected person. No systematic erroneous role-label teaching was established in this selected reading.
- A partial-support target retained known information and local unknowns. One ordering phrase deserves review; the observation does not establish a final-candidate defect.
- A historical partial-support example discarded a known answer via `retrieve_more`. It was already marked failed and separately repaired; the curated manifest excludes known failures and uses no revised labels. It cannot support a claim that this error remains in 1117, and this task does not redo its repair.

The teacher rules already requested brevity and coverage before stopping. Repeating “be concise” does not change the target continuation being positively supervised.

## Run the selector

From the workspace root, with Python 3 and no external dependency:

```bash
python3 ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/data_mechanisms/diagnose_labels.py \
  --dataset snapshot171=.codex_tmp/asa_bonsai2_ptuning_20261012/input_failure_review_snapshot_20261022/annotation/train.jsonl \
  --dataset historical486=.codex_tmp/asa_bonsai2_ptuning_20261012/READER_ERROR_V3_ALL486_SOURCE_20261017.jsonl \
  --dataset baseval46=.codex_tmp/asa_bonsai2_ptuning_20261012/input_failure_review_snapshot_20261022/base_val.jsonl \
  --private-output .memory/tasks/2026-10-25-ptuning-success/evidence/data-mechanisms/generated \
  --aggregate-output ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/data_mechanisms/available_sample_aggregate.json
```

Private output includes per-fact serialized character positions and complete selected evidence packs, plus `ready_for_training: false`. Public aggregate output contains only counts and source hashes. `--top-k-per-kind` selects highest fact counts/lengths deterministically, and every non-answer `partial_support` row is also selected for reading. These are review strata, never automatic failures or deletion instructions. Citation changes between document-stage IDs are a scope-review signal, not evidence of a bad cross-stage join.

Optional `--local-tokenizer DIRECTORY` uses a local fast tokenizer through installed Transformers, with `local_files_only=True` and no remote code. Without it the tool reports characters, **not tokens**. Character offsets refer to the exact serialized target; optional token counts exclude EOS and overlapping boundary-token counts are not an additive loss decomposition.

```bash
cd ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/data_mechanisms
python3 -m unittest -v
```

Tests use abstract fixtures to check the functional contract: select risk candidates without changing a five-required-fact target, preserve distinct repeated-fact offsets, distinguish a review flag from a verdict, retain complete evidence privately, and avoid plot text in aggregate output.

## Exact target intervention contract

Apply only after the true 1117 dataset is recovered and a separate review/experiment authorization is granted. Do not substitute historical 171 or import unreviewed final180 rows.

1. On each selected **original train row**, read the question and complete 20 passages. Enumerate requested slots without consulting model answers. A slot can require multiple facts; a fact can cover multiple slots.
2. For every factual clause record its speaker/actor, recipient/patient, event stage, attribution (`X says` versus established event), modality (`plan` versus happened), and evidence IDs. For every absent relation, specify the local unknown after reading all 20 passages.
3. Map each target clause to a slot or to a logically necessary qualification. A supported clause with neither is an optional scope extension. Mark it for separate review; do not call it false. If deleting it would remove the only requested relationship, negative finding, causal bridge or uncertainty boundary, retain or rewrite it.
4. Produce a **new candidate target**, preserving the original action unless a separately adjudicated action defect exists. Keep all requested known parts and explicitly bound unknown parts. Never convert partial answerability to whole-answer abstention, never guess identities, never turn an allegation into fact, and never merge different stages into one clause.
5. Serialize a complete native-schema JSON answer, close its brackets, and append the normal `<|im_end|>` training terminator. Train the valid shorter completion including its stopping token. Merely masking old suffix tokens would remove their penalty but would not teach this new stopping transition; a raw token cutoff could additionally teach broken JSON.
6. Keep candidate, original hash, slot map and second-review judgment separate. This is not approval to merge or change `ready_for_training`.

An abstract scope extension:

```text
Question slots: delivery actor, recipient, object.
Original: actor delivered; recipient received; object identified;
          later observer repeats that delivery happened.
Reviewed candidate: retain the first three requested relations and stop.
```

A necessary longer answer:

```text
Question slots: which four allegations were made, which one was disputed.
Retain four attributed allegations and the separate disputing reply.
Do not enforce a three-fact cap or collapse all allegations into admitted acts.
```

## Why targets can matter under the actual loss

The implementation uses one example per optimizer step, no accumulation, and mean cross-entropy over the entire target plus terminator. The prompt is not supervised. For a target of T supervised tokens, `L = sum(CE_t) / T`. A longer example is **not** automatically weighted more across steps. Within that example, however, each optional continuation token is positively supervised and consumes part of the mean-loss mass instead of supervising the sufficient answer and its stopping transition. This supports testing target scope; it does not prove it caused generation errors.

Implementation evidence: original `train_pilot.py:125–138,162–181`, `run_ptuning_v2_hybrid.py:93–117`, and `train_optimized_v2_20261019.py:205–225`; adapter teammate independently inspected these paths. The diagnostic tool does not compute semantic loss or a model-quality metric.

## Next discriminating experiment

First test the protocol teammate's inference-only slot-complete condition on both recovered adapters, with full evidence and unchanged decoding. This tests whether generation is responsive to the constraint; it does not isolate training labels.

If proceeding to a data intervention, use two matched training arms from the same initialization, adapter architecture, exact original1117 row IDs/order, optimizer, steps and seed:

- **A:** original labels.
- **B:** only separately reviewed scope-extension labels rewritten to slot-complete native JSON. Retain necessary-long controls unchanged. Do not add data, change actions or alter evidence.

Freeze a source-family-disjoint evaluation set before seeing either arm's output, and exclude rewritten train rows and the repeatedly developed 46 from the primary claim. The 46 can remain a diagnostic panel. Independent, arm-blind review must judge **whole-answer correctness** against all visible evidence, with required-slot omissions, wrong bindings/stages and action errors as explanatory secondary outcomes.

Prediction: B reduces unsupported/incorrect additions after the sufficient answer without increasing slot omissions or whole-answer abstention. A reduction in fact count, target NLL, or offline removable clauses without a gain in whole-answer correctness refutes the proposed benefit. If core wrong bindings remain unchanged while tails shrink, treat scope as a limited contribution and prioritize binding interventions. If necessary-long controls lose content, revise the stop policy rather than celebrating brevity.

## Binding targets

Scope-complete targets address tails; they do not address bindings that are
wrong in the first or core fact. `BINDING_TARGET_SPEC.md` specifies the
binding-target intervention (forward-only preference probe first, then reviewed
uncued-binding and contrast-twin items, then a matched-count training
comparison), and `binding_pairs.py` validates candidate pair files against it
(`test_binding_pairs.py`, abstract fixtures). Both are specifications and checks;
no candidate has been built, reviewed or authorized.
