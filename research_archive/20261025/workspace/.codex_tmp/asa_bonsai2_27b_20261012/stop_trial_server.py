"""Stop only the recorded trial server after all bounded tests have completed."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal


def main():
    stage = Path(__file__).resolve().parent
    for name in ("reader10_nothink", "reader10_think", "pilot5"):
        assert (stage / name / "completion.json").exists(), f"Test still incomplete: {name}"
    launch = json.loads((stage / "server.launch.json").read_text())
    pid = launch["pid"]
    actual = (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    assert [x.decode() for x in actual] == launch["command"], "PID no longer identifies trial server"
    os.kill(pid, signal.SIGTERM)
    record = {"stopped_utc": datetime.now(timezone.utc).isoformat(), "pid": pid,
              "action": "SIGTERM to exact recorded experiment process only",
              "weights_retained": True, "other_jobs_stopped": False}
    with (stage / "server.stop.json").open("x") as out:
        json.dump(record, out, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
