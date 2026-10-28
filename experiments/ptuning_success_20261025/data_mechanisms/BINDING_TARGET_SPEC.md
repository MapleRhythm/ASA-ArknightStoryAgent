# Binding-target data intervention: specification

Status: specification and validator only. Nothing here has been generated,
trained, merged or authorized. Candidate files are private task evidence and
carry `ready_for_training: false`; promotion requires the user's explicit
authorization naming the candidate file hash.

## Why a binding intervention, and why the existing data is not it

The 17-pair audit (`../answer_mechanisms/README.md`) found every v2 loss to be a
binding failure: a fact keeps the right content but attaches it to the wrong
speaker, addressee, participant, identity, stage or scene. Four of thirteen
(positions 21, 22, 27, 45) fail in the first or core fact, so stopping earlier
cannot fix them.

The 1117-row training set already contains binding-oriented rows: its curated
manifest lists 71 `speaker_direction` and 67 `event_state_scope` targeted
additions. In the locally available source of those additions (486 rows, 89
`speaker_direction`), the question itself names the parties and asks for the
binding ("who handed what to whom", "how did X correct Y"), and 71 of 89 targets
have 2–3 facts. The dev46 failures look different: the misbound fact is one the
question did **not** ask about (seven clean-prefix losses), or the question names
one party and the passages resolve the other only through `你/她`, an unlabelled
speaker line, an alias, or a second scene with the same predicate.

Working hypothesis: training taught **cued binding** (binding is the asked slot)
but not **uncued binding** (binding as a qualifier that must survive while the
model writes about something else). The probe in Stage 0 tests this before any
data is built.

## Stage 0 — forward-only binding preference probe (no training, no generation)

For each pair (correct clause *c*, wrong twin *w* differing in one role), score
both by teacher forcing under the unchanged native prompt with all 20 passages:

- **cued**: question asks the role; the clause is fact 1 of the target.
- **uncued**: question asks something else; the clause is appended as fact *k+1*
  after a correct reviewed prefix of *k* facts.

Metric per adapter and condition: fraction of pairs with
`logp(c) > logp(w)` and mean log-probability margin over the differing span
(sum over the tokens where *c* and *w* differ, aligned on the shared prefix and
suffix). Two panels, never pooled:

- *training-side panel*: pairs built under Stage 1 rules (family-disjoint from
  dev46; reviewed sources).
- *dev46 natural-error panel* (evaluation only, never training): for the 13 lost
  positions, *w* = v2's first misbound clause, *c* = the reviewer's minimal
  correction; the prefix is v2's own preceding facts.

Predictions:

| outcome | reading | next action |
|---|---|---|
| cued preference high, uncued near chance, both adapters | uncued binding was never learned | Stage 1 |
| both conditions near chance | the adapter cannot carry role bindings across the 20-passage prompt at all; more targets of the same kind are unlikely to help | adapter/conditioning question, not data |
| input-layer ≫ v2 on uncued, similar on cued | v2's optimization degraded binding stability in long continuation | optimization comparison, not data |
| dev46 natural pairs: v2 prefers its own error given its own prefix | the error is a committed preference, not a sampling accident; scope stopping only hides it | Stage 1 plus the scope A/B |

Cost: two forward passes per pair at ~5–6k prompt tokens; a few hundred pairs
per adapter. Requires the real adapters and the decoder/runtime receipt from the
decode-parity gate.

## Stage 1 — constructing pairs and training items

### Eligible sources (reviewed data only)

1. Rows of the training set of record (curated manifest train SHA-256
   `659586261f7a…`) whose recorded review is passing
   (`label_supported=true`, `decision_appropriate=true`, `issues=[]`).
2. Final-180 repaired candidates only after their own review completes; the 156
   pending rows are ineligible.
3. Never: dev46 tasks or any passage/source family they use; held-out
   confirmation families; the historical 171 snapshot; model outputs as targets;
   unreviewed GLM labels.

The local 486-row source shares no passage text and no source family with dev46
(checked by passage hash and `document_id` family); the validator rechecks this
per pair against every supplied evaluation task file.

### Anchor: a binding hazard visible in the same 20 passages

| hazard | example shape (abstract) |
|---|---|
| `unlabelled_speaker` | a quoted line without a speaker tag next to a tagged line by another character |
| `pronoun_referent` | `你/她` whose referent is fixed only by the scene's addressee |
| `shared_predicate` | the same act (revenge, succession, retirement) said of two different people in the passages |
| `alias_or_namesake` | one entity under two names, or an object sharing a name with a person |
| `cross_stage_entity` | one entity in two stages whose states must not be spliced |
| `allegation_vs_fact` | A claims B did X; X is not narrated as fact |

The reviewer resolves the role from an explicit cue (speaker tag, vocative,
narration) and records it as `anchors` (passage ID + character range). A role
that cannot be resolved from a cue is not eligible as a binding target; the
target then keeps the passage's own designation.

### Wrong twin (single-role edit)

`speaker_swap`, `addressee_swap`, `participant_substitution`, `identity_merge`:
replace exactly one entity mention with a **confusable entity present in the
same passages**. `actor_patient_reversal`: exchange the two parties.
`phase_splice`, `scene_transfer`, `attribution_flip`: recombine true pieces into
a false order, scene or status. Both reviewers must read all 20 passages and
confirm the twin is unsupported or contradicted; a twin that turns out true
elsewhere is discarded.

### Training items (positive supervision only)

The current objective is mean cross-entropy over the target with one example per
step. Wrong twins are therefore **never** training targets; they serve Stage 0
and any later margin objective (unlikelihood/preference), which is a different
training objective and needs its own authorization and arm.

- `uncued_binding`: a new question whose slots (per `SLOT_RUBRIC.md`) are not the
  binding, but whose correct answer necessarily contains the hazardous clause.
  This is the missing supervision.
- `contrast_twin`: two questions on byte-identical passages that differ only in
  the role asked (A→B vs B→A; stage 1 vs stage 2); targets differ only in the
  binding.
- Targets: native schema, `answer_directly`, 1–8 facts with 1–2 valid `E` IDs,
  stop after the slots are covered (the slot-complete contract in `README.md`),
  normal terminator. No new actions, no evidence changes.

### Review, quotas, status

- Two distinct reviewers per pair, both with `all20_checked`, `correct_supported`,
  `wrong_false`.
- Trainable items ≤ 10% of base rows (≤ 111 for 1117); ≤ 3 pairs per source family;
  cover all six hazards; both rounds.
- `ready_for_training` is always `false` in candidate files; `binding_pairs.py`
  refuses `true`. No automatic merge. The training set of record is never
  edited in place; an authorized set is a new file with its own manifest.

### Validator

```bash
PYTHONDONTWRITEBYTECODE=1 python3 binding_pairs.py \
  --pairs <private candidates.jsonl> \
  --exclude-tasks <dev46 tasks.jsonl> [--exclude-tasks <held-out tasks.jsonl>] \
  --slot-rubric ../protocol_experiments/SLOT_RUBRIC.md \
  --aggregate-out <private or public aggregate.json>
```

It checks every rule above that is mechanically checkable (20 passages and
hashes, evaluation overlap by passage and family, anchor ranges, single-entity
substitution with a confusable present, two reviews, slot rubric hash, target
schema, wrong clause never a target fact, twin on identical passages, quotas,
`ready_for_training=false`). Semantic truth of either clause is the reviewers'
judgment; the validator cannot certify it. Output holds counts and pair IDs only.

## Stage 2 — matched training comparison (only after authorization)

Same initialization, input-layer adapter architecture (the current best), data
order seed, optimizer, LR, schedule and decoder; one example per step:

- **A** original 1117 (existing run, reference);
- **C** 1117 + *N* matched ordinary reviewed items from the same sources and
  review, without hazards;
- **B** 1117 + *N* binding items.

B vs C isolates binding content from "more data/steps"; C vs A isolates the
added data. Primary outcome: whole-answer correctness under `SLOT_RUBRIC.md` on
a frozen, family-disjoint held-out set judged arm-blind; dev46 positions 21, 22,
27, 45 (core binding) and 1, 17 (late necessary cause) as diagnostic panel;
Stage 0 probe rerun on B and C.

Prediction: B raises uncued preference margin and lowers required-scope
`actor_recipient`/`identity`/`phase_modality` errors relative to C without new
omissions. Falsifier: margin rises but whole-answer correctness does not, or C
matches B (the gain is from extra data, not binding content).

Resources: Stage 1 review of ≤ 111 binding + ≤ 111 control items, two passes
each. Stage 2: two new training runs of ~1228 steps each; the historical
input-layer run recorded 41,507 s (11.5 h) of training for 1117 steps, so
expect ~12.7 h per arm before evaluation.
