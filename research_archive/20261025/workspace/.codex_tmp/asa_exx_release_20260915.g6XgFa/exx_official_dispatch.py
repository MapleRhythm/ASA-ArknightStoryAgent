"""Route default canonical CLI calls to the pinned Exx package."""
from pathlib import Path
import runpy
import sys


def dispatch_if_official(root, arguments):
    root = Path(root).resolve()
    forwarded = []
    selected = None
    i = 0
    while i < len(arguments):
        value = arguments[i]
        if value == "--runtime-config":
            if i + 1 == len(arguments):
                return False  # Let the existing argparse report the error.
            selected = arguments[i + 1]
            i += 2
            continue
        if value.startswith("--runtime-config="):
            selected = value.split("=", 1)[1]
        else:
            forwarded.append(value)
        i += 1
    if selected is not None:
        config_path = Path(selected)
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        official_configs = {
            root / "configs/runtime_inference.json",
            root / "configs/runtime_inference_gpu.json",
            root / "configs/runtime_exx_official.json",
        }
        if config_path not in official_configs:
            return False
        import hashlib
        pinned_config = root / "current-release/configs/runtime_exx_official.json"
        if config_path.read_bytes() != pinned_config.read_bytes():
            raise RuntimeError("Official config alias differs from the pinned release.")
    entry = root / "current-release/scripts/run_exx_official.py"
    if not entry.is_file():
        raise RuntimeError("Official Exx release is missing; refusing silent legacy fallback.")
    sys.argv = [str(entry), *forwarded]
    runpy.run_path(str(entry), run_name="__main__")
    return True
