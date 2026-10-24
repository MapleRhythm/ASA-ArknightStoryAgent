# ASA v3.5 document reranker experiment and system research plan

## Scope of this run

- Canonical host: 241.
- 248 is only a disposable training/evaluation mirror.
- All generated labels, pair files, logs, weights, reports and manifests live
  under the mounted run directory.
- The supervisor verifies the physical mount with `mountpoint` before its first
  `mkdir` and again before each write-heavy stage.  Framework caches, temporary
  files and tracker state are explicitly redirected under the mounted run.
  A missing mount is a hard failure; it must never fall through to a same-name
  directory on the system disk.
- The frozen 649 x 120 v3.5 pool contains document candidates only.  This run
  is therefore a document-reranker baseline.  It does not train the dynamic
  chain scorer or chain-member attribution used later in the online stack.

## Training gate

1. Stream-audit DeepSeek and GLM shards: protocol, 120-ID completeness,
   labels, ambiguity invariants and train/validation story-family isolation.
2. Export D/P/S > X/I weighted pairwise records independently per provider.
3. Merge by complete query group: DeepSeek primary, GLM fallback.  Never mix
   providers inside one query group.
4. Start from base `bge-reranker-v2-m3`, not v6.
5. One epoch, learning rate 6e-6, max length 1024, weighted softplus,
   effective batch 16.  Keep at most two checkpoints and reload best eval loss.
6. Compare base, online v6 and new model on exactly the same complete
   validation pools.  Required metrics: D/DP/DPS hit and recall at 1/5/10/32,
   nDCG, hard-context rate and 120-candidate latency, both globally and by
   query type.
7. Do not deploy from validation loss.  A new unseen story-family test and
   end-to-end prompt-evidence coverage evaluation are release gates.

## Why this baseline may improve recall but hurt head ranking

The principal exporter creates D/P/S > X/I pairs only for teacher groups
with at most five D+P+S candidates; over-broad groups remain in the immutable
label audit but contribute no training pairs.  Within accepted groups, only a
small number of D > P > S ordinal pairs are retained.  The objective therefore
optimizes answer-helpful evidence recall more strongly than direct-answer
precision.  This is useful for measuring the recall ceiling but may allow S
items to displace D items at R@1/R@5.  If that happens, do not add epochs.
Move to query-group listwise training with tier-aware target mass and base-model
KL/score distillation.

## Online-stack discontinuities that BGE training alone cannot fix

1. The online reranker scores every document and then scores up to 80 dynamic
   multi-document chain strings with the same cross-encoder.
2. A chain score is propagated to each member using heuristic role and overlap
   bonuses; one answerable chain can therefore lift a weak member.
3. Final Top-32 uses `document_score + chain_member_score`, but prompt selection
   calls `evidence_score()`, which returns `evidence_chain_score` first and
   discards the document score.  Prompt Top-10 can therefore reorder Top-32 by
   a different score.
4. Prompt text prefers the complete chain text over the member document text.
   Multiple selected members of the same chain collapse into one rendered
   block, so `prompt_evidence_top_k=10` need not produce ten E-IDs.
5. Non-MMR prompt selection ignores question and hypothesis.  MMR, when
   enabled, only rewards lexical diversity and still has no answer-facet or
   minimal-support-set objective.
6. `has_answerable_evidence()` treats any evidence text of at least 80
   characters as answerable and may trigger an extra direct-answer generation.
7. The 4B model performs initial hypothesis generation, action/answer
   generation, optional follow-up planning and a possible final direct-answer
   retry.  This, not BGE, dominates historical P95 latency.
8. The current evidence-id branch can pass its structured claim-to-ID checks
   and then fall through into the legacy whole-answer token validator.  That
   applies two incompatible grounding policies to one answer and can convert a
   valid compact citation into `retrieve_more`/`abstain`.  Evidence-id success
   must be terminal; a regression test now covers this path locally.

## Required ablations after the document baseline

Run every variant on unseen story families and novel human-written questions:

- document-only BGE, no chain scoring;
- document BGE plus chain score, without propagation to members;
- current chain-to-member propagation;
- final Top-32 order preserved into prompt;
- current prompt re-sort by `evidence_chain_score`;
- direct member text versus preferred full chain text;
- score-only Top-10 versus query-conditioned set selection;
- one retrieval/action round versus two rounds;
- heuristic evidence-length sufficiency versus a trained evidence-state model.

Report candidate recall, prompt complete-support rate, direct-support precision,
redundant member rate, E-ID count, grounded claim pass rate, abstention reasons,
answer correctness and P50/P95 latency.

## Primary paper direction

### Attribution-Aware Dynamic Evidence Chain Reranking

Research question: when a dynamically assembled chain supports an answer,
which members contribute direct, partial, bridging or no support?

Model outputs:

- document direct-support score;
- chain answerability/completeness score;
- per-member marginal contribution;
- optional minimal-support-set mask.

Training data:

- build chains with the deployed chain constructor;
- teacher labels the complete chain and every member independently;
- create leave-one-member-out chain variants;
- label whether removal changes answerability or loses an answer facet;
- add false-member chains by replacing one member with same-story hard context.

Losses:

- tier-aware document listwise loss;
- chain answerability BCE/listwise loss;
- member attribution BCE or ordinal loss;
- leave-one-out consistency loss;
- sparsity penalty for a minimal sufficient set;
- KL/score anchoring to base BGE for fact/reveal retention.

Evaluation:

- chain answerability AUC/nDCG;
- member attribution F1;
- minimal support set recall/precision;
- Top-10 complete support and redundancy;
- final grounded-answer accuracy, false citation and latency.

This direction is specific to ASA's real online mechanism and is stronger than
claiming another generic coverage-aware reranker.

## Secondary paper direction

### Latency-Constrained Evidence-State Policy for Agentic Narrative RAG

Replace the 4B model's repeated action decisions with a small calibrated policy
over `ANSWERABLE`, `RETRIEVABLE`, `UNANSWERABLE`, `AMBIGUOUS` and
`FALSE_PREMISE`, mapping to answer/retrieve/abstain/clarify/correct-premise.

First train with supervised counterfactual evidence states (full evidence,
critical-evidence deletion, distractor insertion, wrong chapter, false premise).
Only after a strong supervised baseline use offline constrained RL or a
contextual bandit.  Reward combines grounded answer quality, improper
abstention, useless retrieval and measured latency.  RL rollout time may be
unbounded; the deployed policy must keep end-to-end P95 under 20 seconds.

## Resource allocation

- A100: full-parameter/listwise BGE and 8B adapter experiments.
- One 4090: 4B vLLM inference and end-to-end evaluation.
- One 4090: retrieval/reranking measurements.
- One 4090: verifier or parallel ablations.

Use the actual free devices rather than fixed assignments; never stop another
user's process.  The proposed BGE, small policy/verifier and adapter experiments
fit the available 3 x 4090 + 1 x A100 budget.  Large online RL and pretraining
from scratch do not.

## Execution status at 2026-08-21 04:40 +08:00

- Canonical 241 mount is online.  The old incomplete-pair exporter was stopped
  before it could cross its health gate.
- Exact frozen-input preflight proves train coverage is 513/533 (20 missing)
  and validation coverage is 116/116.
- GLM-5.3 coding-plan endpoint is reachable; the normal BigModel endpoint and
  the supplied DeepSeek key have insufficient balance.
- Missing labels are written only to `completion_v3_5`; old DeepSeek and GLM
  shards remain untouched.  One completion row has already succeeded.
- 241 has 256 CPUs but is under heavy shared load.  API completion may run below
  load 220 because it is network-bound; BGE training remains gated at load 160
  plus an idle physical GPU1/GPU2 with at least 18 GiB free.
