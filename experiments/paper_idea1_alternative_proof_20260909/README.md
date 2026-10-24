# Alternative-proof-aware post-training pilot

> Audit update, 2026-09-09: see [Phase 2](phase2_deep_audit/REPORT_20260909.md).
> The original semantic totals and recovery/novelty claims below are historical,
> not validated conclusions. Raw evidence and predictions are preserved.

Date: 2026-09-09

This directory is intentionally isolated from the other dirty-worktree
experiments. It studies the first proposal in
`ASA_Paper_Research_Ideas_20260907.docx`:

> If one annotated proof is removed, the label must depend on whether another
> complete proof remains. A reader should preserve the answer under proof
> substitution, identify a genuinely broken proof, and recover after missing
> support is restored.

The first stage is a behavioral feasibility pilot, not a training result. It
uses the four already reviewed strict9 families that contain an explicit
alternative support group. It compares the untouched Qwen3.5-4B base reader
with the historical clean-SFT adapter under the same canonical Exx
non-thinking pipeline.

Directory layout:

- `input/`: extracted idea, source hashes, and compact frozen evidence.
- `tools/`: matrix construction and result analysis.
- `data/`: the generated intervention matrix.
- `runs/`: raw model outputs and runtime metadata.
- `audits/`: protocol and semantic review artifacts.
- `summary/`: machine-readable aggregate results.
- `REPORT_20260909.md`: final research report.

The matrix contains seven states per question family:

1. proof A only;
2. proof B only;
3. both complete proofs;
4. proof B after evidence renumbering and a matched distractor;
5. a genuinely broken state in round 1;
6. the same broken state at the final round;
7. restoration of the missing relation at the final round.

No evidence text is truncated. Model prompts contain only the first
conversation turn. Reference targets and review metadata are retained for
offline scoring but are never passed to the reader.
