"""Bounded reference run with pinned official runtime, preserving other jobs."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "bonsai2_27b_20261012"
PROBES = [
    "The capital of France is",
    "用户：请用中文回答：一加一等于几？\n助手：",
    "<|im_start|>system\n你是剧情问答助手，只根据给定证据回答。<|im_end|>\n"
    "<|im_start|>user\n证据[E1]：阿米娅带领罗德岛救出了博士。\n"
    "问题：谁带领罗德岛救出了博士？<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n",
]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, payload):
    request = urllib.request.Request(
        "http://127.0.0.1:18128" + path,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with OPENER.open(request, timeout=180) as response:
        return json.load(response)


def main():
    free = int(subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip())
    assert free > 9500, f"Need safe reference GPU headroom, found {free} MiB"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 18128))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", LD_LIBRARY_PATH=str(ORIGINAL / "bin"))
    command = [
        str(ORIGINAL / "bin/llama-server"), "-m",
        str(ORIGINAL / "models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"),
        "--host", "127.0.0.1", "--port", "18128", "-ngl", "99", "-fa", "on",
        "-c", "2048", "-np", "1", "-b", "256", "-ub", "128", "-t", "8",
    ]
    log_path = ROOT / f"reference_server_{time.time_ns()}.log"
    with log_path.open("x") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        (ROOT / "reference_pid.json").write_text(json.dumps({"pid": proc.pid, "command": command}))
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    raise RuntimeError(f"Reference exited {proc.returncode}")
                try:
                    with OPENER.open("http://127.0.0.1:18128/health", timeout=2):
                        break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError("Reference startup")
            tokenizer = Tokenizer.from_file(str(ROOT / "official/tokenizer.json"))
            rows = []
            for prompt in PROBES:
                tokens = post("/tokenize", {"content": prompt, "add_special": False})["tokens"]
                assert tokens == tokenizer.encode(prompt, add_special_tokens=False).ids
                result = post("/completion", {
                    "prompt": tokens, "n_predict": 1, "temperature": 0, "n_probs": 20,
                    "return_tokens": True, "cache_prompt": False, "seed": 42,
                })
                rows.append({"prompt": prompt, "tokens": tokens, "native": result})
                (ROOT / "reference.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            # Only terminate the process started by this exact script.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == "__main__":
    main()
