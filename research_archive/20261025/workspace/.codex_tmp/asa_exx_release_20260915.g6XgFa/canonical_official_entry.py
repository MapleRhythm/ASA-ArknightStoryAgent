#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
entry = root / "current-release/scripts/run_exx_official.py"
if not entry.is_file():
    raise SystemExit("Pinned Exx release missing; no legacy fallback.")
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name="__main__")
