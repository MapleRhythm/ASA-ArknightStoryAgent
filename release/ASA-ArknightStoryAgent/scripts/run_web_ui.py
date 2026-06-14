#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"

MODES = {
    "cpu-local": {
        "label": "CPU Local 4B",
        "description": "qwen3.5 4B GGUF + LoRA merged, no reranker",
        "script": PROJECT_ROOT / "scripts" / "run_cpu_qwen35_4b_no_reranker.sh",
    },
    "cpu-api": {
        "label": "CPU API",
        "description": "local retrieval + remote OpenAI-compatible model, no reranker",
        "script": PROJECT_ROOT / "scripts" / "run_cpu_api_no_reranker.sh",
    },
    "gpu-reranker": {
        "label": "GPU Reranker + 4B",
        "description": "CUDA retrieval/reranker + vLLM 4B LoRA",
        "script": PROJECT_ROOT / "scripts" / "run_gpu_reranker_qwen35_4b.sh",
    },
}


SERVICE_MODES = {"cpu-local", "gpu-reranker"}


def default_python_bin() -> str:
    candidates = [
        Path("/home/zhb/miniconda3/envs/train/bin/python3.11"),
        Path("/home/zhb/miniconda3/envs/reasoning/bin/python3.11"),
        PROJECT_ROOT / ".conda" / "bin" / "python3.11",
        PROJECT_ROOT.parents[1] / ".conda" / "bin" / "python3.11",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def select_free_cuda_device() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    best_index: str | None = None
    best_free = -1
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            free_mb = int(float(parts[1]))
        except ValueError:
            continue
        if free_mb > best_free:
            best_index = parts[0]
            best_free = free_mb
    return best_index


def build_child_env(mode: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = env.get("PYTHON_BIN") or default_python_bin()
    if mode == "gpu-reranker" and not env.get("CUDA_VISIBLE_DEVICES"):
        selected_devices = env.get("ASA_CUDA_VISIBLE_DEVICES") or env.get("ASA_GPU_DEVICES") or select_free_cuda_device()
        if selected_devices:
            env["CUDA_VISIBLE_DEVICES"] = selected_devices
    package_candidates = [
        PROJECT_ROOT / ".python_packages" / "train",
        PROJECT_ROOT.parents[1] / ".python_packages" / "train",
        PROJECT_ROOT / "model" / "lora" / ".python_packages" / "train",
        PROJECT_ROOT.parents[1] / "model" / "lora" / ".python_packages" / "train",
    ]
    pythonpath_entries = [str(path) for path in package_candidates if path.exists()]
    if pythonpath_entries:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries + ([existing] if existing else []))
    return env


class PersistentInferenceService:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.mode: str | None = None
        self.command: list[str] = []
        self.started_at: float | None = None
        self.ready_payload: dict[str, Any] | None = None
        self.stderr_lines: list[str] = []
        self.lock = threading.Lock()
        self.stderr_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def status(self) -> dict[str, Any]:
        running = self.is_running()
        return {
            "running": running,
            "mode": self.mode if running else None,
            "pid": self.process.pid if running and self.process else None,
            "uptime": round(time.time() - self.started_at, 1) if running and self.started_at else 0,
            "ready": bool(self.ready_payload) if running else False,
            "command": [Path(part).name if part.startswith(str(PROJECT_ROOT)) else part for part in self.command],
            "stderr_tail": self.stderr_lines[-80:],
        }

    def _collect_stderr(self, process: subprocess.Popen[str]) -> None:
        if process.stderr is None:
            return
        for line in process.stderr:
            line = line.rstrip("\n")
            if not line:
                continue
            self.stderr_lines.append(line)
            if len(self.stderr_lines) > 300:
                del self.stderr_lines[:100]

    def start(self, mode: str = "cpu-local", timeout: float = 900.0) -> dict[str, Any]:
        with self.lock:
            if mode not in SERVICE_MODES:
                raise ValueError(f"Persistent service is not supported for mode: {mode}")
            if self.is_running():
                if self.mode == mode:
                    return self.status()
                self.stop_locked()

            command = build_command({"mode": mode, "message": "__warmup__"})
            command = command[:-1]
            env = build_child_env(mode)
            env["ASA_PERSISTENT_SERVICE"] = "1"
            self.stderr_lines = []
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )
            self.process = process
            self.mode = mode
            self.command = command
            self.started_at = time.time()
            self.ready_payload = None
            self.stderr_thread = threading.Thread(target=self._collect_stderr, args=(process,), daemon=True)
            self.stderr_thread.start()

            deadline = time.time() + timeout
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Persistent inference service exited during startup: " + "\n".join(self.stderr_lines[-20:]))
                assert process.stdout is not None
                readable, _, _ = select.select([process.stdout], [], [], min(1.0, max(0.0, deadline - time.time())))
                if not readable:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    self.stderr_lines.append(line.rstrip("\n"))
                    continue
                if payload.get("event") == "ready":
                    self.ready_payload = payload
                    return self.status()
                self.stderr_lines.append(line.rstrip("\n"))
            self.stop_locked()
            raise TimeoutError(f"Persistent inference service startup timed out after {timeout}s")

    def stop_locked(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin:
                process.stdin.write(json.dumps({"command": "stop"}, ensure_ascii=False) + "\n")
                process.stdin.flush()
                process.wait(timeout=10)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        finally:
            self.process = None
            self.ready_payload = None

    def stop(self) -> dict[str, Any]:
        with self.lock:
            previous = self.status()
            self.stop_locked()
            return {"stopped": True, "previous": previous}

    def ask(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        with self.lock:
            if not self.is_running():
                raise RuntimeError("Persistent inference service is not running. Click '启动服务' first.")
            if self.mode != (payload.get("mode") or "cpu-local"):
                raise RuntimeError(f"Persistent service is running in mode {self.mode}, but request mode is {payload.get('mode')}.")
            process = self.process
            if process is None or process.stdin is None or process.stdout is None:
                raise RuntimeError("Persistent inference service pipe is not available.")
            request = {
                "message": payload.get("message"),
                "history": payload.get("history"),
                "max_retrieval_rounds": payload.get("max_retrieval_rounds"),
                "max_tokens": payload.get("max_tokens"),
            }
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()

            started = time.time()
            deadline = started + timeout
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Persistent inference service exited: " + "\n".join(self.stderr_lines[-20:]))
                readable, _, _ = select.select([process.stdout], [], [], min(1.0, max(0.0, deadline - time.time())))
                if not readable:
                    continue
                line = process.stdout.readline()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    self.stderr_lines.append(line.rstrip("\n"))
                    if len(self.stderr_lines) > 300:
                        del self.stderr_lines[:100]
                    continue
                if data.get("event") == "result":
                    result = data.get("result") if isinstance(data.get("result"), dict) else None
                    return {
                        "ok": True,
                        "returncode": 0,
                        "mode": self.mode,
                        "answer": data.get("answer") or (result or {}).get("answer") or "",
                        "result": result,
                        "stdout": "",
                        "stderr": "\n".join(self.stderr_lines[-80:]),
                        "stderr_lines": self.stderr_lines[-80:],
                        "command": [Path(part).name if part.startswith(str(PROJECT_ROOT)) else part for part in self.command],
                        "cwd": str(PROJECT_ROOT),
                        "service": self.status(),
                        "stages": data.get("stages") or [],
                        "elapsed": data.get("elapsed"),
                    }
                if data.get("event") == "error":
                    return {
                        "ok": False,
                        "returncode": 1,
                        "mode": self.mode,
                        "answer": data.get("message") or "生成失败。",
                        "result": None,
                        "stdout": "",
                        "stderr": data.get("message") or "",
                        "stderr_lines": self.stderr_lines[-80:] + [f"[error] {data.get('error')}: {data.get('message')}"],
                        "command": [Path(part).name if part.startswith(str(PROJECT_ROOT)) else part for part in self.command],
                        "cwd": str(PROJECT_ROOT),
                        "service": self.status(),
                    }
            raise subprocess.TimeoutExpired(self.command, timeout)


SERVICE = PersistentInferenceService()


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, status: int = 404) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def render_dialogue_context(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in history[-12:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"assistant: {content}")
        elif role == "user":
            lines.append(f"user: {content}")
    return "\n".join(lines)


def build_command(payload: dict[str, Any]) -> list[str]:
    mode = str(payload.get("mode") or "cpu-local")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    script = MODES[mode]["script"]
    if not script.exists():
        raise FileNotFoundError(f"Mode script not found: {script}")

    question = str(payload.get("message") or "").strip()
    if not question:
        raise ValueError("message cannot be empty")

    command = ["bash", str(script)]
    dialogue_context = render_dialogue_context(payload.get("history") if isinstance(payload.get("history"), list) else [])
    if dialogue_context:
        command.extend(["--dialogue-context", dialogue_context])

    max_retrieval_rounds = payload.get("max_retrieval_rounds")
    if mode != "cpu-api" and max_retrieval_rounds not in (None, ""):
        command.extend(["--max-retrieval-rounds", str(int(max_retrieval_rounds))])

    max_tokens = payload.get("max_tokens")
    if max_tokens not in (None, ""):
        command.extend(["--max-tokens", str(int(max_tokens))])

    command.append(question)
    return command


def run_inference(payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    if payload.get("use_persistent_service", True) and (payload.get("mode") or "cpu-local") in SERVICE_MODES:
        return SERVICE.ask(payload, timeout)

    command = build_command(payload)
    env = build_child_env(str(payload.get("mode") or "cpu-local"))
    started_cwd = str(PROJECT_ROOT)
    completed = subprocess.run(
        command,
        input="/exit\n",
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    parsed = extract_first_json_object(completed.stdout)
    answer = ""
    if parsed:
        answer = str(parsed.get("answer") or "").strip()
    if not answer:
        answer = completed.stdout.strip()

    return {
        "ok": completed.returncode == 0 and bool(answer),
        "returncode": completed.returncode,
        "mode": payload.get("mode") or "cpu-local",
        "answer": answer,
        "result": parsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stderr_lines": [line for line in completed.stderr.splitlines() if line.strip()],
        "command": [Path(part).name if part.startswith(str(PROJECT_ROOT)) else part for part in command],
        "cwd": started_cwd,
    }


class WebUIHandler(BaseHTTPRequestHandler):
    server_version = "ASAWebUI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            json_response(
                self,
                {
                    "ok": True,
                    "project_root": str(PROJECT_ROOT),
                    "service": SERVICE.status(),
                    "modes": {
                        key: {
                            "label": value["label"],
                            "description": value["description"],
                            "available": value["script"].exists(),
                            "readable": os.access(value["script"], os.R_OK),
                        }
                        for key, value in MODES.items()
                    },
                },
            )
            return
        if path in {"", "/"}:
            path = "/index.html"
        target = (WEB_ROOT / path.lstrip("/")).resolve()
        try:
            target.relative_to(WEB_ROOT.resolve())
        except ValueError:
            text_response(self, "Forbidden", HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            text_response(self, "Not found", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/api/service/start", "/api/service/stop"}:
            json_response(self, {"ok": False, "error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(raw_body or "{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            timeout = float(payload.get("timeout") or self.server.inference_timeout)
            if path == "/api/service/start":
                mode = str(payload.get("mode") or "cpu-local")
                response = SERVICE.start(mode=mode, timeout=timeout)
                json_response(self, {"ok": True, "service": response})
                return
            if path == "/api/service/stop":
                response = SERVICE.stop()
                json_response(self, {"ok": True, **response})
                return
            response = run_inference(payload, timeout=timeout)
            status = HTTPStatus.OK if response["ok"] else HTTPStatus.INTERNAL_SERVER_ERROR
            json_response(self, response, status)
        except subprocess.TimeoutExpired as exc:
            json_response(
                self,
                {
                    "ok": False,
                    "error": "timeout",
                    "message": f"Inference timed out after {exc.timeout}s",
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                },
                HTTPStatus.REQUEST_TIMEOUT,
            )
        except Exception as exc:  # Keep the UI useful during local setup.
            json_response(
                self,
                {"ok": False, "error": type(exc).__name__, "message": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}", flush=True)


class ASAWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], *, inference_timeout: float) -> None:
        super().__init__(server_address, handler_class)
        self.inference_timeout = inference_timeout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local ASA-ArknightStoryAgent Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-question inference timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not WEB_ROOT.exists():
        raise SystemExit(f"Web assets not found: {WEB_ROOT}")
    server = ASAWebServer((args.host, args.port), WebUIHandler, inference_timeout=args.timeout)
    url = f"http://{args.host}:{args.port}"
    print(f"ASA-ArknightStoryAgent Web UI running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
