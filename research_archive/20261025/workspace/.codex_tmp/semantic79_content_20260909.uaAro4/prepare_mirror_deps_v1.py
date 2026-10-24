"""Install only verified wheels into this staging directory, never system Python."""
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
import zipfile

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "pytest-9.1.1-py3-none-any.whl": "37a86b45efb9a47a61a36449063e8e18d0cab3161329fc099eb21783169c4f0c",
    "tqdm-4.67.3-py3-none-any.whl": "ee1e4c0e59148062281c49d80b25b67771a127c85fc9676d3be5f243206826bf",
}


def main():
    assert HERE.name == ".canonical_sync_20260908.lxnhWO"
    assert sys.version_info[:2] == (3, 10)
    wheel_dir, target = HERE / "wheels", HERE / "cpu_test_deps"
    assert not wheel_dir.exists() and not target.exists()
    wheel_dir.mkdir()
    subprocess.run([
        sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
        "download", "--no-cache-dir", "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
        "--dest", str(wheel_dir), "pytest==9.1.1", "tqdm==4.67.3",
    ], check=True, timeout=240)
    manifest = []
    for wheel in sorted(wheel_dir.iterdir()):
        assert wheel.suffix == ".whl" and not wheel.is_symlink()
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            assert len(names) == 1
            info = BytesParser().parsebytes(archive.read(names[0]))
        name, version = info["Name"], info["Version"]
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        with urlopen(url, timeout=30) as response:
            official = json.load(response)
        matching = [r for r in official["urls"] if r["filename"] == wheel.name]
        assert len(matching) == 1 and matching[0]["digests"]["sha256"] == digest
        assert not matching[0]["yanked"]
        if wheel.name in EXPECTED:
            assert digest == EXPECTED[wheel.name]
        manifest.append({
            "name": name, "version": version, "filename": wheel.name, "sha256": digest,
            "metadata_url": url, "requires_python": info["Requires-Python"],
        })
    assert set(EXPECTED) <= {r["filename"] for r in manifest}
    with (HERE / "verified_wheel_manifest_v1.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    subprocess.run([
        sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
        "install", "--no-index", "--no-compile", "--ignore-installed",
        "--find-links", str(wheel_dir), "--target", str(target), "pytest==9.1.1", "tqdm==4.67.3",
    ], check=True, timeout=120)
    installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(target)])}
    assert installed == {r["name"]: r["version"] for r in manifest}
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "target": str(target), "installed": installed,
        "system_or_training_environment_modified": False, "wheels": manifest,
        "context_correction": "Both wheel SHA256 values in the inherited narrative summary were incorrect. They were not used. Values were re-read from official PyPI and every downloaded wheel was checked against official metadata before installation.",
    }
    with (HERE / "isolated_dependencies_receipt_v1.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"target": str(target), "installed": installed}))


if __name__ == "__main__":
    main()
