"""Byte-only CPU candidate; one change relative to the tested context guard."""
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


run = Path(__file__).resolve().parent
source = run / "context_guard_candidate_bytes_v1/src"
target = run / "quote_integrity_candidate_v1/src"
target.mkdir(parents=True)
changed = Path("asa_arknight_story_agent/inference/payload/utils.py")
replacement = (run / "payload_utils.quote_integrity.py").read_bytes()
canonical = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/release/ASA-ArknightStoryAgent/src")
records = []
for path in sorted(source.rglob("*")):
    relative = path.relative_to(source)
    if "__pycache__" in relative.parts or path.suffix == ".pyc" or not path.is_file():
        continue
    assert not path.is_symlink()
    previous = path.read_bytes()
    assert (canonical / relative).read_bytes() == previous, f"Concurrent source change: {relative}"
    content = replacement if relative == changed else previous
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(content)
    assert destination.read_bytes() == content
    records.append({
        "path": str(relative), "before_sha256": sha(previous), "after_sha256": sha(content),
        "changed": content != previous,
    })
assert [r["path"] for r in records if r["changed"]] == [str(changed)]
report = {
    "evaluation_only": True, "production_changed": False, "source_files": records,
    "candidate_source": str(target), "changes": [str(changed)],
}
with (target.parent / "manifest.json").open("x") as output:
    output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"source_files_verified": len(records), "changed": [str(changed)]}))
