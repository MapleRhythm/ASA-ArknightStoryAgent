# Publication boundaries for research artifacts

This repository publishes source code, tests, methodology, aggregate findings, and URL/hash manifests. It does not publish raw evidence prompts, model predictions, training/evaluation row data, runtime logs, local machine or storage metadata, credentials, or third-party full-text literature unless a separate review establishes redistribution permission.

## Public GitHub scope

The following belong in normal source commits when accompanied by their direct tests:

- runtime and offline-audit source code;
- test cases that do not embed protected source text or credentials;
- experiment protocols, READMEs, research reports, and aggregate summaries;
- bibliographic URL/hash manifests that identify, but do not copy, third-party sources.

## Private research archive scope

The following remain preserved in a private Hugging Face archive or equivalent access-controlled storage, rather than a public Git commit:

- rendered evidence and complete prompt captures;
- raw model predictions, review-row exports, candidate training rows, and evaluation matrices;
- runtime logs and machine-specific run state;
- downloaded PDFs and full-text XML/TXT copies of third-party literature;
- files containing local infrastructure, account-operation, credential, or billing context.

Private archival preserves reproducibility without treating source-derived text or operational material as public code. A release of any private artifact requires a separate provenance, licensing, and data-review decision.

## Commit review rule

Before staging an experiment directory, use explicit file paths. Include tools, tests, reports, and source manifests only after checking that they contain no raw evidence payloads or operational metadata. Do not use a directory-wide add command for experiment output trees.
