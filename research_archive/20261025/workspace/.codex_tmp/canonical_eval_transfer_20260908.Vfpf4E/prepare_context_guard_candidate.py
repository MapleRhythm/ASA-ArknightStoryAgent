"""Build a byte-only candidate snapshot on 241, without chmod/chown/copy stat.

The experiment mount rejects cp's chmod operations. Existing failed candidate
copies are retained, and this script only creates a new exclusive directory.
"""
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
source = run / "after/src"
target = run / "context_guard_candidate_bytes_v1"
target.mkdir()
modified = Path("asa_arknight_story_agent/inference/model_runtime/vllm_runner.py")
patch = (run / "vllm_runner.context_guard.py").read_bytes()
records = []
for path in sorted(source.rglob("*")):
    relative = path.relative_to(source)
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        continue
    if path.is_symlink():
        raise ValueError(f"source snapshot unexpectedly contains a symlink: {relative}")
    if not path.is_file():
        continue
    original = path.read_bytes()
    data = patch if relative == modified else original
    destination = target / "src" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f:
        f.write(data)
    if digest(destination.read_bytes()) != digest(data):
        raise ValueError("candidate byte verification failed")
    records.append({
        "path": str(relative), "source_sha256": digest(original), "candidate_sha256": digest(data),
        "changed": data != original,
    })
changed = [r["path"] for r in records if r["changed"]]
if changed != [str(modified)]:
    raise ValueError(f"unexpected change set: {changed}")
report = {
    "evaluation_only": True, "source": str(source), "candidate": str(target),
    "script_sha256": digest(Path(__file__).read_bytes()), "files": records, "changed_paths": changed,
    "file_metadata_not_preserved": True,
}
with (target / "manifest.json").open("x") as f:
    f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"files_verified": len(records), "changed_paths": changed}))
