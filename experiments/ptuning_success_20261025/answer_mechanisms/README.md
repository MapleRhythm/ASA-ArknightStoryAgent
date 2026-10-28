# Where v2 answers go wrong: 17-pair audit

The input-layer adapter and the optimized layer-wise v2 adapter were trained on the
same 1117 rows for 1117 steps and answered the same 46 development questions with
all 20 passages. Historical whole-answer content grades: input-layer 24 complete,
v2 15 complete. Thirteen questions were complete under input-layer and not under
v2; four went the other way. This directory locates the first material error in
each of those 17 pairs. It does not grade anything new and measures no model.

## Findings

| v2 lost a complete answer (13) | positions | what happened |
|---|---|---|
| clear clean prefix | 10, 13, 16, 30, 32, 35, 41 | a correct answer to every slot is complete by fact *k*; a later fact misbinds speaker, participant, addressee or phase |
| boundary clean prefix | 8, 33 | same shape, but the prefix's adequacy or the first error is a judgment call |
| first fact wrong | 21, 27, 45 | the first fact already carries the binding/identity error |
| error entangled with core | 22 | the fact that links the core cause to the asked entity is the one with the identity error |

| v2 gained a complete answer (4) | positions | what happened |
|---|---|---|
| late necessary cause | 1, 17 | the direct cause the question asks for first appears in fact 4 (pos 1) and fact 6 (pos 17) |
| binding repaired | 24 | input-layer's identity error absent in v2 |
| question relevance | 31 | v2 answers the asked interpretation instead of restating context |

Error kinds across the 13 losses (`summary.json`, field `mechanism`): speaker
binding 3, participant binding/direction 3, addressee 1, phase 1, scene 1,
relation 1, identity 2, unsupported inference 1. Every one is a failure to keep
*who said/did what to whom, in which scene and stage* attached to the right fact.
Length is not the mechanism: position 35's v2 answer has fewer facts than
input-layer's (5 vs 6) and still lost; position 27's v2 answer has four facts and is wrong in fact 1.

## How to read 7/13

7/13 is a **conditional diagnostic on losses selected because they were lost**.
It says that among v2's lost questions, the most common shape is "sufficient
correct core, then a misbound extra fact". It is not a recoverable fraction of
v2, and **15 + 7 = 22/46 is not a v2 accuracy, counterfactual or otherwise**:

1. The cut point was found by a reader who already knew which fact was wrong.
   No deployable rule reproduces it. A fixed fact cap fails in both directions:
   positions 1 and 17 need facts 4 and 6; position 45 is wrong at fact 1 and only
   reaches its correct "cannot be confirmed" conclusion at fact 3.
2. An instruction or training change that makes the model stop earlier changes
   the whole generation, including the 15 complete and 18 other non-complete v2
   answers. Truncating historical text predicts none of that.
3. Positions 10 and 16 are round variants of one question; the 7 positions are
   6 question groups. The 46 rows are 43 groups of a reused development set.
4. All semantic labels are an assistant's audit, not independent human gold.
   The historical grades themselves are not ground truth: for position 8 the
   audit found that the historical review penalised a supported event (the
   remaining unsupported inference still keeps that answer incomplete).

The ratio does not transfer to the best adapter. Of input-layer's 22
non-complete answers only four (15, 24, 26, 38) have the clean-prefix-then-error
shape; the rest fail at facts 1–2 or by omission (`../input_errors/`). Perfect
stopping on input-layer is therefore worth at most about +4 (24 → 28) even as an
offline upper bound, and reaching 28 for real also needs core binding fixes.

## Protected cases for any intervention

Any scope or stop intervention is judged against these positions in addition to
the whole-answer primary metric (registered in `prompt_arm.json` and
`../protocol_experiments/preregistered_scope_ab.json`):

- **21, 22, 27, 45 — scope control cannot fix them.** The error sits in the first
  or core fact. They only turn correct if binding itself improves. If an
  intervention fixes the seven clean-prefix cases but not these, scope was the
  lever and binding needs a training-side intervention
  (`../data_mechanisms/BINDING_TARGET_SPEC.md`).
- **1, 17 — stopping early would break them.** Their necessary cause is late. An
  intervention that loses either is a regression even if it shortens answers and
  removes tail errors elsewhere.
- **24** is a gain from a repaired identity binding, but the same entity is still
  misidentified in 10, 16 and 22. One fixed position is not evidence of a stable
  identity representation.

## Intervention prepared from this audit

`prompt_arm.json` (arm `question_slots_binding_v1_1`) is the arm-B instruction
of the preregistered A/B: check the question's necessary slots, bind speaker,
addressee, actor, patient and stage per fact, keep evidence's original
designations when no identity link is stated, keep speculation marked, stop
after all slots are covered, never drop a necessary slot to save facts. Its file
hash is frozen in the preregistration; `paired_protocol.py prepare` refuses to
plan if the text drifts. Revision v1 → v1.1 (before any generation) replaced an
unqualified "no fact cap" with the native 1–8 range, because v2 already wrote
≥ 8 facts on 19/46 questions.

## Files and rerun

- `analyze.py` — verifies the private audit against immutable sources (source
  file hashes, every read passage's hash and full range, exact first-error
  character offsets) and writes counts only. Mutating an offset or a source hash
  fails explicitly. It is a provenance check, not a grader.
- `summary.json` — its output: per-position mechanism, scope status, first error
  fact index, adequate prefix end, passages read. No story text.
- `prompt_arm.json` — the arm-B instruction and protected positions.
- `test_analyze.py` — synthetic tamper tests (offset, source hash, partial read).
  Checks raise `ValueError`, so they also hold under `python -O`.

```bash
# from the workspace root; the annotation file is private task evidence
PYTHONDONTWRITEBYTECODE=1 python3 \
  ASA-ArknightStoryAgent/experiments/ptuning_success_20261025/answer_mechanisms/analyze.py \
  --source-root .codex_tmp/asa_bonsai2_ptuning_20261012 \
  --annotations .memory/tasks/2026-10-25-ptuning-success/evidence/answer-mechanisms/case_evidence.json
```
