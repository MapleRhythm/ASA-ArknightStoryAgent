# Hugging Face research-archive publisher

`scripts/publish_research_archive_hf.py` derives and validates the controlled
Hugging Face handoff payload. It keeps credentials out of the payload and does
not contact Hugging Face unless `publish --execute` is explicitly supplied.

The public research source, protocols, tests, and reports are already published
in GitHub at [`research_archive/20261025`](../research_archive/20261025/README.md).
Hugging Face receives only a compact public handoff plus the separately governed
private archival dataset; it does not duplicate the 1,948-file GitHub archive.

## Scope and remote targets

The script uses these fixed targets:

| Batch | Target | Revision | Remote directory |
| --- | --- | --- | --- |
| Public handoff | model `MapleRhythm/ASA-ArknightStoryAgent` | `pi-handoff-20261025` | `archive/2026-10-25/` |
| Private archive | dataset `MapleRhythm/asa-research-archive` | `main` | `archive/2026-10-25/` |

On execution, an absent public model branch is created from the model's `main`.
The script does **not** change the model repository's `main` branch or existing
model weights. The private dataset batch does write its new `archive/2026-10-25/`
directory to the private dataset's `main` branch.

## Inputs and derived output

Set paths for the approved publication root and an owned, fresh derived-payload
directory. `SOURCE` is immutable input. `PAYLOAD` is generated output and must
not be a historical dry-run directory with older manifest metadata.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python3}"
PUBLICATION_ROOT="${ASA_PUBLICATION_ROOT:?set the approved publication root}"
SOURCE="$PUBLICATION_ROOT/2026-10-25"
PAYLOAD="$PUBLICATION_ROOT/hf-upload-20261025-r1"
PUBLISHER="$REPO_ROOT/scripts/publish_research_archive_hf.py"
```

`SOURCE` must contain the audited publication inputs:

```text
2026-10-25/
  public-research-code/          # published GitHub preservation source
  private-research-archive/      # 1,343 reviewed private archival files
  PUBLIC_MANIFEST.jsonl
  MANIFEST.jsonl
  MISSING-REMOTE.tsv
```

`prepare` derives `PAYLOAD/public/` and `PAYLOAD/private/`. It preserves the
private files under their archive-relative paths and derives strict SHA-256
manifests. The public batch contains only its generated README, public manifest,
and current handoff/publication-boundary documentation snapshots. Credential
materials, secret-scan outputs, excluded credential files, and excluded/held
manifest items are not copied.

A previously prepared 1,343-file dry-run payload is a historical snapshot. Do
not claim that it matches the final re-signed public manifest and do not upload
it. Run `prepare` with the current audited source and a fresh payload directory;
if a payload path already exists, the script compares it exactly and refuses to
overwrite a mismatch.

## Local preparation and validation

These commands are local-only. They do not load a token, connect to Hugging
Face, create a repository or branch, or upload files.

```bash
"$PYTHON" "$PUBLISHER" prepare \
  --source "$SOURCE" \
  --payload "$PAYLOAD" \
  --docs-dir "$REPO_ROOT/docs"

"$PYTHON" "$PUBLISHER" publish --payload "$PAYLOAD"
```

`prepare` validates the source allowlist and derives the new payload.
`publish` without `--execute` validates exact membership, rejects symlinks,
recomputes every manifest SHA-256, and prints only batch counts and byte totals.

## Explicit remote execution

Run this only after local validation succeeds and official Hugging Face
connectivity, repository permission, provenance, licensing, and data-review
approval are established:

```bash
HF_OFFICIAL_PROXY="http://127.0.0.1:7890" \
"$PYTHON" "$PUBLISHER" publish \
  --payload "$PAYLOAD" \
  --token-name "${HF_TOKEN_NAME:-dexrobot}" \
  --execute
```

Omit `HF_OFFICIAL_PROXY` when direct official connectivity works. The optional
proxy must resolve to loopback. `--execute` reads the named locally stored token
only in process memory and communicates only with `https://huggingface.co`.

Before writing, the script verifies the authenticated identity, explicit write
authority for both targets, target visibility, and the existing remote run
directory. It aborts if an existing remote file differs or an unexpected path is
present; matching files are skipped. An absent private dataset is created only
after an authenticated official 404 and only as `private=True`.

## Current status

Hugging Face official connectivity remains blocked and no Hugging Face upload
has been performed. This runbook adds a reproducible local preparation and
validation path; it does not authorize or perform a network retry.
