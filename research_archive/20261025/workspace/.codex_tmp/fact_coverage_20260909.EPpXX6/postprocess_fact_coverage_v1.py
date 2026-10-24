"""Wait for this experiment only, then analyze and audit immutable outputs once."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from build_fact_coverage_pairs_v1 import HERE, RUN, SOURCE, load


def main():
    log_path = HERE / "postprocess_v1.log"
    assert not log_path.exists()
    with log_path.open("x", buffering=1) as log:
        started, last_print = time.monotonic(), 0
        while not (HERE / "launch_receipt_v1.json").exists():
            elapsed = time.monotonic() - started
            assert elapsed < 7200, "Generation completion timeout; do not launch duplicate jobs"
            if elapsed - last_print >= 30 or not last_print:
                value = {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "waiting_for_existing_generation", "elapsed_seconds": int(elapsed),
                }
                line = json.dumps(value)
                print(line, flush=True)
                log.write(line + "\n")
                last_print = max(elapsed, 0.1)
            time.sleep(10)
        receipt = load(HERE / "launch_receipt_v1.json")
        assert receipt["returncode"] == 0 and receipt["source_unchanged"] and receipt["config_unchanged"]
        commands = [
            [
                sys.executable, str(RUN / "analyze_canonical_grounding_v4.py"),
                "--validator", "after", "--source-root", str(SOURCE),
                "--matrix", str(HERE / "fact_coverage_pairs116_v1.matrix.json"),
                "--predictions", str(HERE / "paired116_ctx12288_v1/predictions.json"),
                "--output", str(HERE / "validation_v1.json"),
            ],
            [
                "${REMOTE_PYTHON}",
                str(HERE / "check_fact_coverage_postflight_v1.py"),
            ],
            [sys.executable, str(HERE / "audit_fact_coverage_pairs_v1.py")],
        ]
        for index, command in enumerate(commands):
            stage = ["mechanical_analysis", "protocol_postflight", "blind_audits"][index]
            value = {"at_utc": datetime.now(timezone.utc).isoformat(), "status": "starting", "stage": stage}
            print(json.dumps(value), flush=True)
            log.write(json.dumps(value) + "\n")
            subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"status": "postprocess_completed"}), flush=True)


if __name__ == "__main__":
    main()
