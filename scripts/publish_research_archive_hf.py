#!/usr/bin/env python3
"""Prepare or publish the 2026-10-25 ASA handoff without exposing credentials.

Default mode is local-only validation. --execute is required for every remote
operation. The named Hugging Face token stays in this process and is used only
against https://huggingface.co (optionally through an explicit loopback proxy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

OFFICIAL_ENDPOINT = "https://huggingface.co"
OWNER = "MapleRhythm"
MODEL_REPO = f"{OWNER}/ASA-ArknightStoryAgent"
DATASET_REPO = f"{OWNER}/asa-research-archive"
BRANCH = "pi-handoff-20261025"
RUN_DIR = "archive/2026-10-25"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9_-]{8,}"),
    re.compile(rb"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----"),
    re.compile(rb"(?:OPENAI|HF|HUGGINGFACE)_(?:API_)?KEY\s*[:=]", re.I),
    re.compile(rb"SECRET[-_ ]?SCAN", re.I),
    re.compile(rb"(?:^|[^A-Za-z])source_root(?:[^A-Za-z]|$)", re.I),
    re.compile(rb"PRIVATE_MANIFEST", re.I),
    re.compile(rb"/(?:home|root|tmp|Users)/", re.I),
    re.compile(rb"[A-Za-z]:\\\\"),
)


class PublicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlannedFile:
    local_path: Path
    remote_path: str
    sha256: str
    git_blob_sha1: str


@dataclass(frozen=True)
class Batch:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    files: tuple[PlannedFile, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise PublicationError(f"unsafe manifest path: {value!r}")
    return path


def read_manifest(path: Path, required_name: str) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        raise PublicationError(f"missing regular manifest: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid manifest {path.name}: {type(exc).__name__}") from exc
    if set(value) != {"schema_version", "files"} or value.get("schema_version") != 1 or not isinstance(value.get("files"), list):
        raise PublicationError(f"{path.name} permits only schema_version=1 and a files list")
    records: dict[str, str] = {}
    for item in value["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise PublicationError(f"{path.name} entries must contain only path and sha256")
        relative = safe_relative(item["path"])
        digest = item["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PublicationError(f"invalid sha256 declaration for {relative}")
        if relative.as_posix() == required_name or relative.as_posix() in records:
            raise PublicationError(f"manifest cannot declare itself or duplicate a path: {relative}")
        records[relative.as_posix()] = digest
    if not records:
        raise PublicationError(f"{path.name} may not be empty")
    return records


def verify_regular_file(root: Path, relative: str, expected_sha256: str) -> PlannedFile:
    local = root / relative
    if not local.is_file() or local.is_symlink():
        raise PublicationError(f"manifest item is not a regular non-symlink file: {relative}")
    actual = sha256_file(local)
    if actual != expected_sha256:
        raise PublicationError(f"sha256 mismatch for {relative}")
    return PlannedFile(local_path=local, remote_path="", sha256=actual, git_blob_sha1=git_blob_sha1(local))


def public_path_allowed(relative: str) -> bool:
    path = PurePosixPath(relative)
    return relative in {"README.md", "PUBLIC_MANIFEST.jsonl"} or (
        len(path.parts) >= 2 and path.parts[0] == "docs" and path.suffix == ".md"
    )


def scan_public_file(path: Path) -> None:
    content = path.read_bytes()
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            raise PublicationError(f"public policy scan rejected {path.name}")


def make_batch(payload: Path, kind: str) -> Batch:
    if kind == "public":
        root = payload / "public"
        manifest_name = "PUBLIC_MANIFEST.json"
        repo_id, repo_type, revision = MODEL_REPO, "model", BRANCH
    else:
        root = payload / "private"
        manifest_name = "PRIVATE_MANIFEST.json"
        repo_id, repo_type, revision = DATASET_REPO, "dataset", "main"
    if not root.is_dir() or root.is_symlink():
        raise PublicationError(f"missing regular {kind} payload directory")

    declared = read_manifest(root / manifest_name, manifest_name)
    if kind == "public":
        if "README.md" not in declared:
            raise PublicationError("PUBLIC_MANIFEST.json must declare README.md")
        if any(not public_path_allowed(relative) for relative in declared):
            raise PublicationError("public manifest permits only README.md and docs/**/*.md")

    planned: list[PlannedFile] = []
    manifest = root / manifest_name
    if kind == "public":
        scan_public_file(manifest)
    planned.append(
        PlannedFile(
            local_path=manifest,
            remote_path=f"{RUN_DIR}/{manifest_name}",
            sha256=sha256_file(manifest),
            git_blob_sha1=git_blob_sha1(manifest),
        )
    )
    for relative, expected_sha256 in sorted(declared.items()):
        verified = verify_regular_file(root, relative, expected_sha256)
        if kind == "public":
            scan_public_file(verified.local_path)
        planned.append(
            PlannedFile(
                local_path=verified.local_path,
                remote_path=f"{RUN_DIR}/{relative}",
                sha256=verified.sha256,
                git_blob_sha1=verified.git_blob_sha1,
            )
        )

    actual_files = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and not candidate.is_symlink()
    }
    permitted_files = set(declared) | {manifest_name}
    if actual_files != permitted_files:
        extra = sorted(actual_files - permitted_files)
        missing = sorted(permitted_files - actual_files)
        raise PublicationError(f"{kind} payload must exactly match manifest allowlist; extra={extra}, missing={missing}")
    return Batch(kind, repo_id, repo_type, revision, tuple(planned))


def local_plan(payload: Path) -> tuple[Batch, Batch]:
    if not payload.is_dir() or payload.is_symlink():
        raise PublicationError("payload path does not exist or is not a regular directory")
    return make_batch(payload, "public"), make_batch(payload, "private")


def canonical_manifest(entries: dict[str, str]) -> bytes:
    return (json.dumps(
        {"schema_version": 1, "files": [{"path": path, "sha256": digest} for path, digest in sorted(entries.items())]},
        ensure_ascii=False,
        indent=2,
    ) + "\n").encode("utf-8")


def write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PublicationError(f"refusing to overwrite prepared file: {path.name}")
    path.write_bytes(content)


def source_private_entries(source: Path) -> dict[str, str]:
    archive = source / "private-research-archive"
    manifest = source / "MANIFEST.jsonl"
    if not archive.is_dir() or not manifest.is_file():
        raise PublicationError("source must contain private-research-archive and MANIFEST.jsonl")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("target") != "private-research-archive":
            continue
        destination = record.get("destination_relative_path", "")
        prefix = "private-research-archive/"
        if not destination.startswith(prefix):
            raise PublicationError("private archive manifest path has an invalid destination prefix")
        relative = safe_relative(destination[len(prefix):]).as_posix()
        digest = record.get("destination_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or relative in expected:
            raise PublicationError("private archive manifest has an invalid or duplicate SHA entry")
        expected[relative] = digest
    if not expected:
        raise PublicationError("source private archive manifest contains no files")
    actual = {
        item.relative_to(archive).as_posix()
        for item in archive.rglob("*")
        if item.is_file() and not item.is_symlink()
    }
    if actual != set(expected):
        raise PublicationError("source private archive does not exactly match MANIFEST.jsonl private entries")
    return expected


def generated_public_readme() -> bytes:
    return b"""# ASA research handoff archive (2026-10-25)

This Hugging Face branch stores a compact public handoff: this README, the
public provenance manifest, and documentation snapshots. It deliberately does
not duplicate the 1,948-file research-code archive here.

The public research source, protocols, tests, and reports are published in the
GitHub project under
[`research_archive/20261025`](https://github.com/MapleRhythm/ASA-ArknightStoryAgent/tree/fix/grounding-pool-chain-priority/research_archive/20261025).

The companion private dataset `MapleRhythm/asa-research-archive` preserves
reviewed private research artifacts and complete provenance metadata. It is not
a model-weight release. Existing model weights and the model repository `main`
branch are intentionally unchanged.

The JSONL manifest records the published GitHub archive's relative paths and
sanitized SHA-256 values. The two Markdown snapshots define current handoff and
publication boundaries.
"""


def generated_private_readme() -> bytes:
    return b"""# ASA private research archive (2026-10-25)

This private dataset preserves the reviewed research-archive files under their
original archive-relative paths. `metadata/MANIFEST.jsonl` is the complete
provenance ledger; `metadata/MISSING-REMOTE.tsv` records known artifacts that
were not present locally. Credentials, secret scans, excluded credential files,
and operational request receipts are not included.

The public code, tests, protocols, and reports live in the linked GitHub
research archive. This dataset is private archival storage, not a model release.
"""


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def expected_prepared(source: Path, docs_dir: Path) -> tuple[dict[str, bytes], dict[str, str], dict[str, str]]:
    private_entries = source_private_entries(source)
    public_manifest = source / "PUBLIC_MANIFEST.jsonl"
    missing_remote = source / "MISSING-REMOTE.tsv"
    full_manifest = source / "MANIFEST.jsonl"
    for required in (public_manifest, missing_remote, full_manifest):
        if not required.is_file() or required.is_symlink():
            raise PublicationError(f"missing regular source metadata: {required.name}")
    snapshots = {
        "docs/PI_HANDOFF_20261025.md": docs_dir / "PI_HANDOFF_20261025.md",
        "docs/PUBLICATION_BOUNDARIES_20261025.md": docs_dir / "PUBLICATION_BOUNDARIES_20261025.md",
    }
    public_bytes: dict[str, bytes] = {
        "README.md": generated_public_readme(),
        "PUBLIC_MANIFEST.jsonl": public_manifest.read_bytes(),
    }
    for relative, path in snapshots.items():
        if not path.is_file() or path.is_symlink():
            raise PublicationError(f"missing regular documentation snapshot: {path.name}")
        public_bytes[relative] = path.read_bytes()
    for relative, content in public_bytes.items():
        temp = source / ".publication-public-scan"
        # Scan in memory to avoid creating files in the audited source payload.
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                raise PublicationError(f"public policy scan rejected {relative}")
    public_entries = {relative: hashlib.sha256(content).hexdigest() for relative, content in public_bytes.items()}

    metadata_bytes = {
        "metadata/MANIFEST.jsonl": full_manifest.read_bytes(),
        "metadata/MISSING-REMOTE.tsv": missing_remote.read_bytes(),
        "metadata/README.md": generated_private_readme(),
    }
    private_all = dict(private_entries)
    private_all.update({relative: hashlib.sha256(content).hexdigest() for relative, content in metadata_bytes.items()})
    return public_bytes, private_entries, metadata_bytes | {"PRIVATE_MANIFEST.json": canonical_manifest(private_all)}


def expected_output_files(source: Path, docs_dir: Path) -> tuple[dict[str, bytes], dict[str, str], dict[str, bytes]]:
    public_bytes, private_entries, private_bytes = expected_prepared(source, docs_dir)
    public_all = dict(public_bytes)
    public_all["PUBLIC_MANIFEST.json"] = canonical_manifest({
        relative: hashlib.sha256(content).hexdigest() for relative, content in public_bytes.items()
    })
    return public_all, private_entries, private_bytes


def compare_prepared(payload: Path, source: Path, docs_dir: Path) -> tuple[Batch, Batch]:
    public_bytes, private_entries, private_bytes = expected_output_files(source, docs_dir)
    public_root, private_root = payload / "public", payload / "private"
    expected_private = set(private_entries) | set(private_bytes)
    actual_public = {p.relative_to(public_root).as_posix() for p in public_root.rglob("*") if p.is_file() and not p.is_symlink()} if public_root.is_dir() else set()
    actual_private = {p.relative_to(private_root).as_posix() for p in private_root.rglob("*") if p.is_file() and not p.is_symlink()} if private_root.is_dir() else set()
    if actual_public != set(public_bytes) or actual_private != expected_private:
        raise PublicationError("existing prepared payload differs from the required allowlist; refusing overwrite")
    for relative, content in public_bytes.items():
        if (public_root / relative).read_bytes() != content:
            raise PublicationError("existing prepared public metadata differs; refusing overwrite")
    for relative, content in private_bytes.items():
        if (private_root / relative).read_bytes() != content:
            raise PublicationError("existing prepared private metadata differs; refusing overwrite")
    return local_plan(payload)


def prepare_payload(source: Path, payload: Path, docs_dir: Path) -> tuple[Batch, Batch, str]:
    if payload.exists():
        public, private = compare_prepared(payload, source, docs_dir)
        return public, private, "existing_verified"
    if payload.parent.exists() and payload.parent.is_symlink():
        raise PublicationError("prepared payload parent may not be a symlink")
    public_bytes, private_entries, private_bytes = expected_output_files(source, docs_dir)
    staging = payload.with_name(f".{payload.name}.prepare-{os.getpid()}")
    if staging.exists():
        raise PublicationError("stale prepare staging path exists; refusing to reuse it")
    try:
        for relative, content in public_bytes.items():
            write_new(staging / "public" / relative, content)
        source_archive = source / "private-research-archive"
        for relative in sorted(private_entries):
            link_or_copy(source_archive / relative, staging / "private" / relative)
        for relative, content in private_bytes.items():
            write_new(staging / "private" / relative, content)
        os.rename(staging, payload)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    public, private = compare_prepared(payload, source, docs_dir)
    return public, private, "created"


def token_from_named_store(token_name: str) -> str:
    from huggingface_hub.utils._auth import get_stored_tokens

    token = get_stored_tokens().get(token_name)
    if not token:
        raise PublicationError(f"named token is unavailable: {token_name}")
    return token


def configure_transport(loopback_proxy: str | None) -> None:
    for key in PROXY_KEYS:
        os.environ.pop(key, None)
    if loopback_proxy is None:
        return
    parsed = urlparse(loopback_proxy)
    if parsed.scheme not in {"http", "https", "socks5", "socks5h"} or parsed.hostname not in {
        "127.0.0.1", "::1", "localhost"
    }:
        raise PublicationError("the optional proxy must be a loopback URL")


class BoundedSession(requests.Session):
    def __init__(self, proxy: str | None):
        super().__init__()
        self.trust_env = False
        if proxy:
            self.proxies = {"http": proxy, "https": proxy}

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", (5, 20))
        return super().request(*args, **kwargs)


def fine_grained_allows_write(account: dict[str, Any], repo_id: str) -> bool:
    """Accept only explicit repo-scoped write/admin evidence from whoami-v2."""
    access = ((account.get("auth") or {}).get("accessToken") or {})
    fine_grained = access.get("fineGrained") or access.get("fine_grained") or access.get("permissions")
    if fine_grained is None:
        return False
    normalized = json.dumps(fine_grained, sort_keys=True).lower()
    return repo_id.lower() in normalized and ("write" in normalized or "admin" in normalized)


def permission_for(account: dict[str, Any], repo_id: str) -> bool:
    role = (((account.get("auth") or {}).get("accessToken") or {}).get("role") or "").lower()
    if role in {"write", "admin"}:
        return True
    if role in {"finegrained", "fine-grained", "fine_grained"}:
        return fine_grained_allows_write(account, repo_id)
    return False


def branch_target(refs: Any, branch: str) -> str | None:
    for item in refs.branches:
        if item.name == branch:
            return item.target_commit
    return None


def remote_run_check(api: Any, batch: Batch, revision: str) -> tuple[list[PlannedFile], str]:
    """Abort on any unexpected or different file already under this run directory."""
    refs = api.list_repo_refs(batch.repo_id, repo_type=batch.repo_type)
    target = branch_target(refs, revision)
    if target is None:
        return list(batch.files), "revision_missing"
    existing = [name for name in api.list_repo_files(batch.repo_id, repo_type=batch.repo_type, revision=revision)
                if name.startswith(f"{RUN_DIR}/")]
    desired = {file.remote_path: file for file in batch.files}
    if set(existing) - set(desired):
        raise PublicationError(f"remote run already contains unexpected paths in {batch.name} batch")
    remote_info = {
        item.path: item
        for item in api.get_paths_info(batch.repo_id, paths=existing, repo_type=batch.repo_type, revision=revision)
    } if existing else {}
    missing: list[PlannedFile] = []
    for remote_path, planned in desired.items():
        info = remote_info.get(remote_path)
        if info is None:
            missing.append(planned)
            continue
        remote_digest = info.lfs.sha256 if info.lfs else info.blob_id
        local_digest = planned.sha256 if info.lfs else planned.git_blob_sha1
        if remote_digest != local_digest:
            raise PublicationError(f"remote run content differs; refusing overwrite in {batch.name} batch")
    return missing, target


def commit_batch(api: Any, batch: Batch, files: Iterable[PlannedFile], parent_commit: str) -> str | None:
    from huggingface_hub import CommitOperationAdd
    from huggingface_hub.errors import HfHubHTTPError

    operations = [CommitOperationAdd(path_in_repo=file.remote_path, path_or_fileobj=file.local_path) for file in files]
    if not operations:
        return None
    try:
        info = api.create_commit(
            repo_id=batch.repo_id,
            repo_type=batch.repo_type,
            revision=batch.revision,
            parent_commit=parent_commit,
            operations=operations,
            commit_message=f"archive: {RUN_DIR} {batch.name} metadata",
            commit_description="Exact manifest allowlist and local SHA-256 validation completed before commit.",
        )
    except HfHubHTTPError as exc:
        raise PublicationError(f"official Hub denied or raced {batch.name} commit; no success reported") from exc
    return info.oid


def execute(public: Batch, private: Batch, token_name: str, loopback_proxy: str | None) -> dict[str, Any]:
    configure_transport(loopback_proxy)
    token = token_from_named_store(token_name)
    from huggingface_hub import HfApi, configure_http_backend
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    configure_http_backend(lambda: BoundedSession(loopback_proxy))
    api = HfApi(endpoint=OFFICIAL_ENDPOINT, token=token, library_name="asa-handoff-publisher")
    try:
        account = api.whoami()
    except HfHubHTTPError as exc:
        raise PublicationError("official whoami failed; identity and permissions remain unverified") from exc
    identity = account.get("name")
    if identity != OWNER:
        raise PublicationError("named token identity is not MapleRhythm")
    if not permission_for(account, MODEL_REPO):
        raise PublicationError("whoami does not establish write authorization for the target model repository")
    if not permission_for(account, DATASET_REPO):
        raise PublicationError("whoami does not establish write authorization for the target dataset repository")

    model = api.repo_info(MODEL_REPO, repo_type="model")
    if model.private:
        raise PublicationError("target handoff model repository visibility changed unexpectedly")

    try:
        dataset = api.repo_info(DATASET_REPO, repo_type="dataset")
        if not dataset.private:
            raise PublicationError("existing archive dataset is not private; refusing visibility change")
        dataset_state = "existing_private"
    except RepositoryNotFoundError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status != 404:
            raise PublicationError("archive dataset is absent or unauthorized; refusing to create over an unverified target") from exc
        dataset = None
        dataset_state = "absent"
    except HfHubHTTPError as exc:
        raise PublicationError("cannot establish archive dataset existence/visibility") from exc

    # Both remote trees are inspected before any creation or commit.
    public_missing, public_parent = remote_run_check(api, public, BRANCH)
    if dataset is None:
        private_missing, private_parent = list(private.files), "revision_missing"
    else:
        private_missing, private_parent = remote_run_check(api, private, "main")

    # The first write to an absent branch is its intended creation from main, never main mutation.
    if public_parent == "revision_missing":
        try:
            api.create_branch(MODEL_REPO, repo_type="model", branch=BRANCH, revision="main", exist_ok=False)
        except HfHubHTTPError as exc:
            raise PublicationError("official Hub denied handoff branch creation; model write authorization is not established") from exc
        public_missing, public_parent = remote_run_check(api, public, BRANCH)
        if public_parent == "revision_missing":
            raise PublicationError("created handoff branch cannot be resolved")
    if dataset is None:
        try:
            api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=False)
        except HfHubHTTPError as exc:
            raise PublicationError("official Hub denied private dataset creation; dataset write authorization is not established") from exc
        private_parent = branch_target(api.list_repo_refs(DATASET_REPO, repo_type="dataset"), "main")
        if private_parent is None:
            raise PublicationError("created private archive dataset cannot be resolved")

    public_commit = commit_batch(api, public, public_missing, public_parent)
    private_commit = commit_batch(api, private, private_missing, private_parent)
    return {
        "identity": identity,
        "model_repo": MODEL_REPO,
        "model_branch": BRANCH,
        "dataset_repo": DATASET_REPO,
        "dataset_state_before": dataset_state,
        "public_commit": public_commit,
        "private_commit": private_commit,
        "write_operations": True,
    }


def batch_summary(batch: Batch) -> dict[str, Any]:
    return {
        "batch": batch.name,
        "repo_id": batch.repo_id,
        "repo_type": batch.repo_type,
        "revision": batch.revision,
        "run_dir": RUN_DIR,
        "file_count": len(batch.files),
        "total_bytes": sum(file.local_path.stat().st_size for file in batch.files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ASA Hugging Face 2026-10-25 publisher")
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="derive a verified upload payload without network access")
    prepare.add_argument("--source", required=True, type=Path, help="audited 2026-10-25 source publication root")
    prepare.add_argument("--payload", required=True, type=Path, help="owned derived upload directory")
    prepare.add_argument("--docs-dir", required=True, type=Path, help="current project documentation directory")
    publish = subcommands.add_parser("publish", help="validate a prepared payload; --execute enables official-Hub writes")
    publish.add_argument("--payload", required=True, type=Path, help="prepared payload root")
    publish.add_argument("--token-name", default="dexrobot")
    publish.add_argument("--execute", action="store_true", help="permit official-Hub writes after local validation")
    publish.add_argument("--loopback-proxy-env", default="HF_OFFICIAL_PROXY")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            public, private, state = prepare_payload(args.source, args.payload, args.docs_dir)
            result = {
                "ok": True,
                "mode": "prepare",
                "state": state,
                "write_operations": False,
                "batches": [batch_summary(public), batch_summary(private)],
            }
        else:
            public, private = local_plan(args.payload)
            result = {
                "ok": True,
                "mode": "execute" if args.execute else "dry-run",
                "write_operations": False,
                "batches": [batch_summary(public), batch_summary(private)],
            }
            if args.execute:
                proxy = os.environ.get(args.loopback_proxy_env)
                result.update(execute(public, private, args.token_name, proxy))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except PublicationError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "write_operations": False}, ensure_ascii=False, sort_keys=True))
        return 2
    except Exception as exc:
        # Do not serialize exception arguments because a client/library may include sensitive request details.
        print(json.dumps({"ok": False, "error": type(exc).__name__, "write_operations": False}, ensure_ascii=False, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
