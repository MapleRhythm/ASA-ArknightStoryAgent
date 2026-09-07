# Gold evidence-binding ambiguity findings (2026-09-04)

> 2026-09-07 correction: the first-pass judge saw at most 600 characters per
> E-ID, and this third-pass script saw at most 1800. Three of the 86 rows
> exceeded 1800 characters. These counts remain historical results, but the
> sidecar is not a full-evidence training authority. It is superseded by
> `docs/GOLD_RECALIBRATION_SAFETY_AUDIT_20260907.md`.

## Scope

The strict recalibration judged 1,929 fact rows with one merged-evidence
decision and one decision per cited E-ID. The strict SFT dataset was frozen
and was not modified by this historical audit.

## Contradiction audit

There are 86 rows where merged and individual decisions disagree:

| contradiction | count |
|---|---:|
| merged unsupported, at least one individual supported | 23 |
| merged supported, no individual supported | 63 |

A third GLM-5.3 adjudication was run with an ambiguity-aware protocol.  It
returned:

| label | count |
|---|---:|
| supported_by_union | 38 |
| supported_by_some | 13 |
| ambiguous | 13 |
| unsupported | 22 |

Breakdown:

- Of the 23 `merged_unsupported_ind_supported` rows: 3 union-supported, 4
  partially supported, 4 ambiguous, 12 unsupported.
- Of the 63 `merged_supported_no_ind_supported` rows: 35 union-supported, 9
  partially supported, 9 ambiguous, 10 unsupported.

## Interpretation

1. Individual E-ID support is not a sufficient gold criterion.  The 35
   union-supported rows are genuine multi-evidence chains where no single
   chunk contains the complete claim.
2. The strict merged criterion is also not sufficient.  The 10 rows labelled
   `unsupported` inside the merged-supported group are residual positive-label
   noise.
3. The 13 ambiguous rows mostly involve identity/coreference, implied
   motivation, or a claim that combines directly stated and inferred clauses.
   They should not be converted into hard negatives.

## Training policy

- Keep the current strict SFT run intact only as a controlled historical
  comparison.
- For the next run, construct an ambiguity-preserving label set:
  - `supported_by_union`: retain the union of cited E-IDs;
  - `supported_by_some`: exclude the original compound claim from positive
    training until its supported clause is atomized and independently checked;
  - `ambiguous`: exclude from hard-negative construction, or retain with a
    soft/low weight;
  - `unsupported`: remove cited facts and flip the action only when no
    supported fact remains.
- Never use the 86 contradiction rows as confirmed negatives without a new
  full-evidence adjudication whose evidence SHA matches the manifest.

The sidecar judgments are stored at
`/mnt/store/zhb/exx_grounding_v1/data/gold_recalibrated_20260904/ambiguity_adjudications.jsonl`.
