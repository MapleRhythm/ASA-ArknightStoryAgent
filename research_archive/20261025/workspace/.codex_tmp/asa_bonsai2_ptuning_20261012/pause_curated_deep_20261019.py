"""Stop only the exact superseded experiment, retaining transactional checkpoints."""
import json
import os
from pathlib import Path
import signal
import time

import annotation_protocol as p


def main():
    root = Path(__file__).resolve().parent
    out = root / "ptuning_v2_deep_curated_20261019_attempt01"
    pid = 3664749
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    assert str(root / "train_curated_deep_20261019.py").encode() in command
    assert str(out).encode() in command and b"workflow" in command
    assert os.stat(f"/proc/{pid}").st_uid == os.getuid()
    pointer = json.loads((out / "latest_checkpoint.json").read_text())
    checkpoint = out / pointer["directory"]
    assert checkpoint.parent == out and pointer["step"] > 0
    assert p.file_sha(checkpoint / "training_state.pt") == pointer["training_state_sha256"]
    assert p.file_sha(checkpoint / "deep_prefix.safetensors") == pointer["adapter_sha256"]
    receipt = dict(reason="user_requested_input_layer_training_first",
                   pid=pid, validated_checkpoint=pointer, original_files_preserved=True,
                   training_complete=False, pause_requested_unix=time.time())
    p.atomic_json(out / "user_pause_20261019.json", receipt)
    os.kill(pid, signal.SIGTERM)
    print(p.compact(receipt))


if __name__ == "__main__":
    main()
