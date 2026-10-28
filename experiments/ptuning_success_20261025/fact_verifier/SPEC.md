# Fact-level binding verifier: assessment and minimal design

Scope: frozen Bonsai2 27B P-tuning answers on the repeated 46-question dev set
(input-layer 24/46 complete, optimized-v2 15/46). Every semantic label here is
an assistant judgment, not independent human gold. All "complete" counts below
are offline counterfactual upper bounds or expectations. They are not model
accuracy, and the 46 questions are a reused development set. No model was
called to produce anything in this directory.

## 1. Can the existing verifiers serve as fact filters? No.

**What was measured.** Base Qwen3.5-4B, zero-shot, compared with an
independent GLM-5.3 judgment on 500 (claim, chunk) pairs
(`.codex_tmp/analysis/AGREEMENT_FINDINGS.md`, `metrics_full.remote.json`):

| metric | value |
|---|---:|
| agreement | 91.2% |
| Cohen's kappa | 0.781 |
| AUC (first-token support logit margin vs GLM) | 0.937 |
| precision / recall / specificity (support = positive) | 80.1% / 88.6% / 92.1% |
| confusion TP / FP / TN / FN | 117 / 29 / 339 / 15 |

**Why these numbers do not cover our errors.** Half of the pairs are
dense-retrieved *different* chunks. That makes the task "is this the right
passage?", which the 4B handles well (negative-side agreement 96.8%). The P-tuning
failures are a different kind of error: the cited passage is correct and contains every content word,
but a role is attached to the wrong entity (speaker, addressee, participant,
possessor, identity, phase, scene). The closest measured population is the
gold-positive side, where the chunk is the right passage and GLM judged
121/250 claims not fully supported. On that side:

| positive side (right passage, n=250) | value |
|---|---:|
| 4B rejects a GLM-unsupported claim (r) | 99/121 = 81.8% |
| 4B rejects a GLM-supported claim (false deletion, q) | 14/129 = 10.9% |
| agreement, fact-type claims | 76.7% |
| AUC, relation-type claims | 0.813 (n=34) |

The documented disagreement patterns are the binding patterns: an unnamed
utterance attributed to a named character, half-supported compound claims,
and unproven membership. No measurement exists on role-swapped claims. The
inputs were truncated to 600 characters (184/500 pairs). The claims came from SFT
gold, not from P-tuning outputs, and GLM itself agrees with the dataset's own
positives only 75.2% of the time.

The BGE binding rerankers (v1, augmented v2) rank chunks against a claim and
do not model roles. Their v2 held-out numbers were invalidated by the
2,000-character hard-negative truncation and by compound-claim positives
(`docs/HARDNEG_V2_SAFETY_AUDIT_20260907.md`).

**Why a support classifier is structurally blind here.** On the 135 unique
annotated facts (section 4), the fraction of each fact's character bigrams
found in its own cited passages separates correct from wrong facts with AUC
0.615 (binding errors alone: 0.612). The best single-line match gives 0.59.
Wrong facts reuse the same evidence content as correct ones. In the evidence, 6,992 of
13,267 non-empty passage lines have no speaker label, and 919 of 920 passages
contain such lines. An unlabeled line can be narration, a continuation of the
previous speaker, or an interlocutor whose name was not rendered (one dev case
shows the third kind). That convention ambiguity is a direct input-side source
of speaker misbinding.

**Decision arithmetic.** Deleting a truly wrong fact can never turn a complete
answer into an incomplete one, because an answer that contains a material
error is already not complete. All harm therefore comes from false deletions
of necessary correct facts. Expected complete count on the input layer (fact
mode, 2 necessary facts per already-complete answer, low–high = boundary cases
counted as failures/successes):

| q (false deletion) | r (catch) | expected complete /46 |
|---:|---:|---:|
| 0.109 | 0.818 | 22.5 – 24.4 (4B operating point: no gain) |
| 0.05 | 0.9 | 25.8 – 28.2 |
| 0.02 | 0.9 | 27.4 – 30.0 |
| 0.01 | 0.9 | 28.0 – 30.6 |
| 0 | 1 | 29 – 32 (oracle) |

The zero-shot 4B at its measured operating point is break-even or negative
against the 24/46 baseline. Reaching 28/46 requires q ≤ about 0.01 in fact mode.
Clause repair (section 3) relaxes this to q ≈ 0.03–0.04 because it removes
the need to delete core facts that carry an error. The full grid is in
`counterfactual_input.json` and `counterfactual_v2.json`, with k = 1, 2, 3.

## 2. A cheap rule is not enough either

A deterministic speaker heuristic flagged 22 input-layer facts: the last known
character name before an attribution verb, compared with the label of the
best-matching line, or the inherited label on an unlabeled line. 10 of the 22
matched the 21 binding instances found by an independent per-case review
(precision 0.45, recall 0.48). Three of its flags fell on correct facts of
complete answers, one of them that answer's core fact. On v2 it also flagged
the one fact that made a gained answer complete. The hard part is deciding
which entity a Chinese attribution or second-person pronoun refers to. That
extraction belongs to the model. Deterministic code verifies the model's
extracted anchor; it does not replace the model.

## 3. Minimal viable design

### Verifier call (one per unique fact)

Input: the question, one fact, the full text of every passage the fact cites
(never truncated), and the fact's position in the answer.

The prompt makes the verifier extract roles before it gives a verdict:

1. Quote the shortest evidence span that the fact paraphrases.
2. From the evidence, state the role tuple of that span: speaker (explicit
   label, or "unlabeled"), addressee (vocative or second-person referent),
   participants, possessor, scene, and phase.
3. State the same tuple as the fact asserts it.
4. Compare the two tuples. Output a verdict.

Output (one JSONL row; the contract enforced by `apply_verifier.py`):

```json
{"position": 38, "fact_index": 4,
 "verdict": "supported | unsupported | binding_error | uncertain",
 "confidence": 0.0,
 "binding_role": "speaker | addressee | participant | possessor | identity | phase | scene | polarity | relation | other",
 "anchor": {"evidence_id": "E14", "quote": "<verbatim span>",
            "evidence_role_value": "<who, per evidence>",
            "claim_role_value": "<who, per fact>"},
 "error_text": "<verbatim substring of the fact that is wrong>",
 "repair": "excise | substitute",
 "addresses_question": true}
```

### Deterministic gate (implemented, no model)

A deletion is accepted only when all of the following hold:

- the verdict is `unsupported` or `binding_error`, with confidence at or
  above the threshold;
- the quote occurs in the named passage (insensitive to punctuation and
  whitespace);
- for `binding_error`, the two role values differ (alias table optional);
- for `speaker` errors, the evidence-side speaker equals the quote line's
  explicit label. On an unlabeled line the default (`inherit`) requires it to
  equal the nearest label above; `strict` rejects and `permissive` accepts.

Exact duplicate facts are dropped without a verdict. v2 repeats facts up to 3
times in one answer.

### When deleting would remove the core

The three outcomes are ordered from most to least preferred. A pruned answer
whose core was removed is never shipped.

1. **Clause repair** (mode `clause`). If `error_text` is given, it is either
   excised, or (`repair: substitute`) the claimed role value inside it is
   replaced by the anchored evidence-side role value. Free-form rewriting is
   rejected. Two cases that whole-fact deletion cannot fix need exactly this:
   a correct relation fused with an unsupported rank, and a correct identity
   with a wrong name in parentheses. One input-layer case is fixed by a single
   role substitution inside its first fact.
2. **Regenerate.** The status is `regenerate` when every fact is deleted, or
   when every fact the verifier marked `addresses_question` is deleted. The
   regeneration prompt should carry the verifier's corrected role tuple
   ("the span <quote> is said by <evidence_role_value>"). This step needs a
   model and is not implemented here.
3. **Otherwise**, the answer is pruned.

## 4. Offline counterfactuals (assistant oracle)

Annotations: `oracle_annotations_{v2,input}.textfree.json`. Each fact carries
a label, the required points it covers, and `dup_of`. Private notes stay in task
memory.

**Round-1 cases on v2 (13 lost-complete and 4 gained-complete).**

- Whole-fact deletion recovers 8 of the 13: positions 10, 13, 16, 30, 32,
  35, 41, 45.
- 2 more are boundary cases (8, 33).
- 2 lose a necessary point when the whole fact is deleted, but are fixed by
  clause repair (21, 22).
- 1 cannot be fixed by deletion (27): its core point was never generated.
- None of the 4 gained answers is harmed.

v2 totals: 23–25 with deletion, 25–27 with clause repair. This counts only the
annotated cases; v2's 18 other non-complete answers are not annotated.

Position 45 has a first-fact error but is fixed by deletion. The correct
"cannot confirm" fact comes later and survives. Cutting the answer after a
fixed prefix would have discarded it.

**Input layer (the 22 non-complete answers, reviewed by the input-errors
worker; the fixable subset was re-checked here against aligned evidence
lines).**

| subset | deletion | clause repair |
|---|---|---|
| fixable: 15, 18, 24, 26, 38 | complete | complete |
| boundary: 20, 46 | boundary | complete |
| boundary: 39 | boundary | boundary |

The remaining 14 are omissions, false-premise answers that need
abstain/retrieve, or first-fact scene/addressee errors. A verifier cannot fix
these.

Input-layer totals: 24 → 29–32 with deletion, 31–32 with clause repair.

The target of 28/46 is therefore reachable only by a verifier near the oracle
(section 1 table). The four late-error cases (15, 24, 26, 38) are also what a
stop-when-sufficient rule fixes, so the two interventions are not additive on
those cases. The verifier's own contribution is the cases whose error sits in
or before the core: 18 by deletion, and 18, 20 and 46 by clause repair.

## 5. Acceptance gate before any verifier output is applied

Run the candidate verifier on all unique facts of the 46 input-layer answers.
Measure against the private annotations plus the facts of the 24 complete
answers:

- q (validated deletions of correct facts): at most 0.01 for fact mode, at
  most 0.03 for clause mode. The answers contain 193 unique facts. About 120
  of them are correct (91 in complete answers and 32 annotated in the others).
  At that size, zero false deletions only bounds q below about 2.5% (rule of
  three), so fact mode cannot be certified at 0.01 on this set alone.
- r on the 21 annotated binding instances: at least 0.9.
- Report `rejected_by_reason`. A high `unlabeled_speaker_unverified` count
  means the verifier is guessing speakers on unlabeled lines.

Only then score the filtered answers. Report the result as "dev-set filtered
answers", never as a new model accuracy.

## 6. Usage

```bash
python3 apply_verifier.py validate --generated <generated_native46.jsonl> \
  --tasks <tasks.jsonl> --verdicts verdicts.jsonl
python3 apply_verifier.py apply --generated <generated_native46.jsonl> \
  --tasks <tasks.jsonl> --verdicts verdicts.jsonl --mode clause \
  --threshold 0.9 --unlabeled inherit --out filtered.jsonl --report report.json
python3 apply_verifier.py counterfactual \
  --annotations oracle_annotations_input.textfree.json --out counterfactual_input.json
python3 -m pytest -q test_apply_verifier.py   # synthetic data only
```

The `apply` output keeps `raw_text` unchanged for `unchanged` and `regenerate`
rows. Changed rows are re-rendered as canonical JSON containing only the kept
facts. The report contains counts and positions only.
