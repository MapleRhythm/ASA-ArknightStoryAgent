# SLOT_RUBRIC v1.1 — frozen whole-answer content rule

Status: frozen for `preregistered_scope_ab*.json`. v1.1 (before any generation)
replaced the v1 action rule, which wrongly failed a justified last-round abstain
on a question whose only answer is "cannot be confirmed", and added the
presupposition-failure rule. Its SHA-256 is recorded in each
preregistration and checked by `paired_protocol.py prepare`. Editing one byte
invalidates every prepared run; publish a new version instead of patching this one.

The rubric decides one thing per readable completion: **is the whole answer
content-correct for this question given all 20 visible passages?** JSON validity,
fact count, citation binding and length are recorded separately and never
substitute for this decision.

## 1. Fixing necessary slots before any output is seen

A *slot* is one piece of information the question explicitly asks for. Slots are
written per question from the question text alone, then checked against the
20 passages, and frozen (private slot file, one row per task, carrying this
rubric's hash) before generation. Round variants of the same question share
the same slots.

Derivation rules, applied to the question text:

| Question form | Slots |
|---|---|
| Each `？`-terminated clause, each `并`/`而`/`和`-joined interrogative | one slot per interrogative |
| `为什么/为何/原因` | the **direct** cause or motive of the named agent for the named event. When the passages contain both an earlier speculation and a later explicit statement of motive, the explicit statement is required; speculation alone leaves the slot unsatisfied |
| `如何/怎样/怎么` | the method or process sequence that produced the named outcome |
| `谁/被谁` | the identity of the agent, bound to the named action |
| `是什么关系` | the relation between the two named parties, in the direction the passages state |
| one interrogative over several coordinated named targets (`A和B`, `各`, `X并Y`, `是X也是Y`, `X而非Y`) | one slot per named target; one fact may satisfy several slots |
| `A和B……有何不同` | slot A-side, slot B-side; the contrast is satisfied when both sides are stated with correct ownership |
| `分别` over items the question does not name | one slot: the complete set the passages give, each item bound to its owner |
| `说了什么/透露了什么/指的是什么` | the content of the named speaker's statement, bound to that speaker, addressee and scene |
| `反映了什么/意味着什么` | one slot: the interpretation the passages support, including the evidential basis that makes it an answer rather than a restatement of the question |
| `是否……能否确认` | slot verdict, slot evidential status (confirmed / contradicted / not determinable from these 20 passages) |
| `多少/几` | the number, bound to its target and phase |

Slot file row (private JSONL; contains question wording, never in Git):
`{"task_id", "necessary_slots": [str, ...], "slot_rubric_sha256": <this file>, "author", "evidence_checked": bool}`.
`paired_protocol.py prepare` refuses rows written under another rubric hash and,
after generation, refuses judgments whose `necessary_slots` differ from the row.

Scope constraints in the question (`在……时`, `在……中`, `……后`, `近八小时内`,
named scene or round) are not slots. They restrict which events may be used to
fill slots. A fact taken from another scene or stage to fill a slot is an error
(§3, `scene` or `phase_modality`), not a supported extra.

A slot is **satisfied** when the completion states the content the passages most
directly support for it, with the qualifiers listed in §2. Partial answerability
is legal: if the passages genuinely do not determine a slot, the slot is
satisfied by an explicit local unknown, and fabricating content for it is an
error. Reviewers judge satisfaction against all 20 passages, not against a
reference answer and never against historical model outputs or grades.

## 2. Qualifiers that belong to a slot

A slot answer carries the bindings whose change would change its truth:

- speaker and addressee of any attributed statement;
- actor and patient/recipient of any action;
- identity of each entity (names, aliases, titles, objects that share a name);
- event stage/scene and chronological order where the answer relates events;
- modality: plan vs happened, allegation vs established, speculation vs stated motive.

A slot answer that names the right content with a wrong qualifier is **not**
satisfied and also counts as a material error.

## 3. Material errors (any one makes the whole answer incorrect)

Each error is recorded with `scope` (`required` if it sits in a slot answer or its
qualifier, `extra` otherwise), a claim locator and an evidence rationale.

| type | definition |
|---|---|
| `actor_recipient` | speaker, addressee, actor, patient, participant or direction reassigned (A's words given to B; "you" resolved to the wrong person; who-refused-whom inverted) |
| `identity` | two entities merged, an alias/title attached to the wrong entity, an unsupported name/rank/kinship/subordination introduced |
| `phase_modality` | stages from different times combined into a false sequence or cause; plan stated as done; allegation or speculation stated as fact |
| `scene` | content from a different scene or location used to answer a scene-constrained question, or one scene's purpose transferred to another |
| `unsupported` | an assertion that none of the 20 passages supports, including inferences stronger than the text (suspicion, intent, causation) |
| `contradiction` | an assertion that a passage contradicts |
| `other` | any other factual error; rationale must state why it is material |

Materiality is judged identically in both arms and both adapters. Attribution
changes are material even when the paraphrase is close (rewriting someone's
accusation as the target's own confession is `actor_recipient`). Historical
reviews tolerated some such changes; their grades are therefore not comparable
and are never reused for arm A.

Not errors:

- paraphrase preserving every §2 binding;
- a supported fact that maps to no slot and no necessary qualifier: record it in
  `unnecessary_supported` (diagnostic of scope expansion, no penalty);
- an inference explicitly marked as inference whose basis is in the passages;
- an evidence ID that is wrong while the claim itself is supported (citation
  diagnostic, not content).

## 4. Whole-answer decision

`whole_answer_correct = sufficient AND no omitted slot AND no material error AND action appropriate`

- *sufficient*: every slot satisfied (§1).
- *action appropriate*: a slot has **determinable content** when the 20
  passages establish something substantive for it. "Not determinable from these
  passages" is a local unknown, not determinable content, even when the slot
  asks for confirmability.
  - At least one slot has determinable content: only `answer_directly` is
    appropriate; retrieve/abstain is incorrect even if well-formed.
  - No slot has determinable content: round `1/2` → `retrieve_more`; round `2/2`
    → `abstain` whose reason states what is not established, or
    `answer_directly` that states the same local unknowns without filling them.
    Either 2/2 form satisfies the slots; a reason or fact that adds a material
    error is still an error.
  - Presupposition failure: if the passages do not establish the question's
    premise, answering as though it held is `unsupported` (material), whatever
    the action. Stating that the premise is not established is the correct
    content.
- Malformed or truncated JSON is judged on its readable content. Execution
  failures (missing output, timeout, OOM, length limit without EOS) are incorrect.

## 5. Reviewer procedure

1. Read question and the frozen slots; read all 20 passages (`all20_checked`).
2. Mark each slot satisfied/omitted; list omitted slot strings exactly.
3. Walk every fact in order; classify each as slot answer, necessary qualifier,
   `unnecessary_supported`, or material error with type and scope.
4. Record the first material error's fact index (diagnostic for "error after a
   sufficient prefix" vs "error in the core").
5. Decide the action; compute the whole-answer decision; write a rationale with
   the passage IDs relied on.

Reviewers see no arm, adapter, prompt variant, historical grade or historical
output. Disagreements are adjudicated blind. Assistant-only reviews must be
labelled non-independent and cannot be reported as human-gold accuracy.
