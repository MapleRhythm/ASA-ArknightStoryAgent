"""Launch a bounded, logged experiment, never change the production release."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=["pilot5", "reader10_think"])
    args = p.parse_args()
    stage = Path(__file__).resolve().parent
    executable = "${REMOTE_PYTHON}"
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    repo = Path("${REMOTE_HOME}/ASA-ArknightStoryAgent/releases/exx-20260915-v4")
    assert (repo.parent.parent / "current-release").resolve() == repo.resolve()
    if args.name == "pilot5":
        command = [
            executable, str(stage / "run_pipeline.py"), "--repo", str(repo),
            "--model", str(stage / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
            "--questions", str(stage / "smoke5.txt"), "--out", str(stage / args.name),
        ]
    else:
        command = [
            executable, str(stage / "replay_reader.py"), "--baseline",
            "${REMOTE_HOME}/ASA-ArknightStoryAgent/outputs/experiments/qwen35_9b_20260916_WCkC8J/eval50_fix2",
            "--out", str(stage / args.name), "--thinking",
        ]
    assert not (stage / args.name).exists()
    assert not (stage / f"{args.name}.launch.json").exists()
    with (stage / f"{args.name}.log").open("x") as log:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    record = {"pid": process.pid, "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "production_modified": False}
    (stage / f"{args.name}.launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
