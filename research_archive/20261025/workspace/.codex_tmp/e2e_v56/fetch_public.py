"""Freeze the official FairytaleQA repository, without executing its code."""
import io
import json
from pathlib import Path
import tarfile
import urllib.request
from export_final import sha, save

HERE = Path(__file__).resolve().parent


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ASA-public-eval/1.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read()


def main():
    dest = HERE / "public_source"
    dest.mkdir(exist_ok=False)
    commit_url = "https://api.github.com/repos/uci-soe/FairytaleQAData/commits/main"
    raw = fetch(commit_url)
    commit = json.loads(raw)["sha"]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
    with (dest / "commit.json").open("xb") as f:
        f.write(raw)
    url = f"https://codeload.github.com/uci-soe/FairytaleQAData/tar.gz/{commit}"
    data = fetch(url)
    with (dest / "official_repository.tgz").open("xb") as f:
        f.write(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or member.isdir():
                continue
            assert member.isfile() and ".." not in parts and not Path(*parts).is_absolute()
            path = dest / "repo" / Path(*parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                out.write(tar.extractfile(member).read())
    save(dest / "source_manifest.json", {
        "repository": "https://github.com/uci-soe/FairytaleQAData",
        "commit": commit, "archive_url": url, "archive_sha256": sha(dest / "official_repository.tgz"),
        "files": {str(p.relative_to(dest / "repo")): sha(p)
                  for p in sorted((dest / "repo").rglob("*")) if p.is_file()},
    })
    print(json.dumps({"commit": commit, "bytes": len(data), "destination": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
