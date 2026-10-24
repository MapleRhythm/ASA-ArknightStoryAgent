"""Reuse the audited idle-GPU launcher with explicit isolated artifact paths."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
SEM = RUN / "semantic79_content_v2.fPkkhM"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reused = SEM / "launch_slot_feedback_pairs_v1.py"
    old_preflight = json.loads((SEM / "slot_feedback_preflight_v1.json").read_text())
    assert sha(reused) == old_preflight["launcher_sha256"]
    spec = importlib.util.spec_from_file_location("audited_idle_gpu_launcher", reused)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    paths = {
        "HERE": HERE, "RUN": RUN,
        "MATRIX": HERE / "anchor_ablation_paired8_v1.matrix.json",
        "OUTPUT": HERE / "anchor_ablation_paired8_ctx12288_v1",
        "LOG": HERE / "anchor_ablation_paired8_v1.log",
        "RECEIPT": HERE / "anchor_ablation_launch_v1.json",
    }
    for key, value in paths.items():
        setattr(launcher, key, value)
    with (HERE / "launcher_reuse_manifest_v1.json").open("x", encoding="utf-8") as stream:
        json.dump({
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True, "training_ready": False,
            "wrapper_sha256": sha(Path(__file__)),
            "reused_launcher": str(reused), "reused_launcher_sha256": sha(reused),
            "path_overrides": {k: str(v) for k, v in paths.items()},
            "inherited_preflight_filename": "slot_feedback_preflight_v1.json",
            "note": "Only filename is inherited; this file in this isolated directory describes the new anchor ablation.",
        }, stream, indent=2)
    return launcher.main()


if __name__ == "__main__":
    raise SystemExit(main())
