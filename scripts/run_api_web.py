#!/usr/bin/env python3
"""Local API-only UI. No browser credentials, local generation, or paid retries."""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
RUNTIME_PATH = PROJECT_ROOT / "configs/runtime_api_stable.json"
REQUEST_LIMIT = 128 * 1024
OUTPUT_LIMIT = 16 * 1024 * 1024
STAGES = {
    "user_question_hypothesis_generation": "理解问题",
    "hypothesis_generation": "理解问题",
    "retrieval": "检索剧情原文",
    "conclusion_generation": "核对证据并生成回答",
    "follow_up_hypothesis_generation": "补充检索线索",
    "direct_answer_generation": "整理最终回答",
}


def runtime_config() -> dict[str, Any]:
    return json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))


def redact(text: str) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        text = text.replace(key, "[redacted]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[redacted]", text)


def health_payload() -> dict[str, Any]:
    cfg = runtime_config()
    index = PROJECT_ROOT / "indexes/arknights_story"
    checks = {
        "api_key": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()),
        "index": all((index / name).is_file() for name in ("documents.jsonl", "faiss.index", "bm25_tokens.pkl")),
        "embedding": (PROJECT_ROOT / "model/embeddings/bge-small-zh-v1.5/config.json").is_file(),
        "dependencies": all(importlib.util.find_spec(name) is not None for name in ("faiss", "sentence_transformers")),
    }
    instructions = {
        "api_key": "在启动服务的终端设置 DEEPSEEK_API_KEY，然后重新启动。",
        "index": "运行 python scripts/build_retrieval_index.py --device cpu 构建本地索引。",
        "embedding": "运行 python scripts/download_models.py --runtime cpu-api 下载检索模型。",
        "dependencies": "运行 bash scripts/setup_cpu_api_no_reranker.sh 并激活 .venv-api 环境。",
    }
    return {
        "ok": True, "ready": all(checks.values()), "checks": checks,
        "issues": [{"id": name, "message": instructions[name]} for name, ok in checks.items() if not ok],
        "version": cfg["release"]["version"], "model": cfg["generator"]["model"],
        "mode": "api", "thinking": False, "evidence_limit": 20, "max_rounds": 2,
        "retrieval": "CPU · BGE + BM25 · 无重排器", "api_connectivity": "not_tested",
    }


def validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("请求必须是 JSON 对象。")
    if set(payload) - {"message", "history"}:
        raise ValueError("API 固定版不接受模型、模式或推理参数覆盖。")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 4000:
        raise ValueError("请输入 1–4000 字的问题。")
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) > 10:
        raise ValueError("对话上下文最多 10 条。")
    clean_history = []
    for item in history:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("无效的对话角色。")
        content = item.get("content")
        if not isinstance(content, str) or len(content) > 12000:
            raise ValueError("单条历史记录不能超过 12000 字。")
        clean_history.append({"role": item["role"], "content": content})
    return {"message": message.strip(), "history": clean_history}


def build_command(payload: dict[str, Any]) -> list[str]:
    command = [sys.executable, str(PROJECT_ROOT / "api-mode/run_api_inference.py"),
               "--runtime-config", str(RUNTIME_PATH), "--no-save-run"]
    context = "\n".join(f"{item['role']}: {item['content']}" for item in payload["history"])
    if context:
        command.extend(["--dialogue-context", context])
    return [*command, "--", payload["message"]]


def extract_result(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("answer"), str):
            return value
    return None


def public_result(result: dict[str, Any], elapsed: float) -> dict[str, Any]:
    # Only final answers and source text, never raw prompts or intermediate reasoning.
    evidence = [
        {key: item.get(key) for key in ("id", "activity_name", "story_name", "stage_code", "clean_text")}
        for item in result.get("evidence") or [] if isinstance(item, dict)
    ]
    return {
        "type": "result", "ok": True, "answer": result["answer"],
        "evidence": evidence, "elapsed": round(elapsed, 2),
        "rounds": max((step.get("round", 0) for step in result.get("retrieval_trace", [])
                       if isinstance(step, dict) and isinstance(step.get("round", 0), int)), default=0),
        "model": "deepseek-flash",
    }


def stop_process(process: subprocess.Popen[str]) -> None:
    # This process group belongs exclusively to the current HTTP request.
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
    for pipe in (process.stdin, process.stdout, process.stderr):
        if pipe and not pipe.closed:
            pipe.close()


def inference_events(payload: dict[str, Any], timeout: float, cancelled: threading.Event | None = None):
    if cancelled is not None and cancelled.is_set():
        yield {"type": "error", "message": "服务正在关闭，未发送请求。"}
        return
    started = time.monotonic()
    process = subprocess.Popen(
        build_command(payload), cwd=PROJECT_ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", start_new_session=True, bufsize=1,
    )
    lines: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def consume(name: str, pipe: Any) -> None:
        try:
            for line in pipe:
                lines.put((name, line))
        finally:
            lines.put((name, None))

    workers = [threading.Thread(target=consume, args=(name, pipe), daemon=True)
               for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr))]
    for worker in workers:
        worker.start()
    stdout: list[str] = []
    stderr: list[str] = []
    ended: set[str] = set()
    size = 0
    try:
        assert process.stdin is not None
        process.stdin.write("/exit\n")
        process.stdin.flush()
        process.stdin.close()
        yield {"type": "stage", "message": "正在加载检索资源", "elapsed": 0}
        while len(ended) < 2:
            elapsed = time.monotonic() - started
            if cancelled is not None and cancelled.is_set():
                yield {"type": "error", "message": "服务正在关闭，本地任务已停止。"}
                return
            if elapsed > timeout:
                yield {"type": "error", "message": f"请求超过 {timeout:g} 秒，已停止本地任务。可稍后重试。"}
                return
            try:
                channel, line = lines.get(timeout=0.5)
            except queue.Empty:
                yield {"type": "heartbeat", "elapsed": round(elapsed, 1)}
                continue
            if line is None:
                ended.add(channel)
                continue
            size += len(line.encode("utf-8"))
            if size > OUTPUT_LIMIT:
                yield {"type": "error", "message": "推理输出超出安全限制，任务已停止。"}
                return
            (stdout if channel == "stdout" else stderr).append(line)
            if line.startswith("[stage] "):
                stage = line.removeprefix("[stage] ").strip()
                yield {"type": "stage", "message": STAGES.get(stage, "正在检索与核对证据"),
                       "elapsed": round(elapsed, 1)}
        process.wait(timeout=5)
        result = extract_result("".join(stdout))
        if process.returncode != 0 or result is None or not result["answer"].strip():
            error_text = "".join(stderr)
            if "401" in error_text:
                message = "DeepSeek 密钥无效或已过期，请在服务端更新。"
            elif "402" in error_text:
                message = "DeepSeek 账户余额不足，请检查服务商控制台。"
            elif "429" in error_text:
                message = "DeepSeek 请求限流，请稍后手动重试。"
            elif "finish_reason=length" in error_text:
                message = "模型未在输出预算内完成，未将残缺内容当作答案。"
            elif "timed out" in error_text.lower():
                message = "连接 DeepSeek 超时，请检查服务端网络后重试。"
            else:
                message = "暂时未能生成回答，请检查服务端配置或稍后重试。"
            print("[inference error] " + redact(error_text[-3000:]), file=sys.stderr, flush=True)
            yield {"type": "error", "message": message}
            return
        yield public_result(result, time.monotonic() - started)
    finally:
        stop_process(process)
        for worker in workers:
            worker.join(timeout=1)


class WebUIHandler(BaseHTTPRequestHandler):
    server_version = "ASAWebUI/1.0"

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; "
                         "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        super().end_headers()

    def respond(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed_host(self) -> bool:
        # Protect a loopback-bound service against browser DNS rebinding.
        if self.server.server_address[0] not in {"127.0.0.1", "::1"}:
            return True
        try:
            return urlparse("//" + self.headers.get("Host", "")).hostname in {"127.0.0.1", "localhost", "::1"}
        except ValueError:
            return False

    def do_GET(self) -> None:
        if not self.allowed_host():
            self.respond({"ok": False, "message": "Invalid host"}, 403)
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self.respond(health_payload())
            return
        files = {"/": "index.html", "/index.html": "index.html",
                 "/app.js": "app.js", "/styles.css": "styles.css"}
        if path not in files:
            self.respond({"ok": False, "message": "Not found"}, 404)
            return
        target = WEB_ROOT / files[path]
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(target.name)[0] or "text/plain") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self.allowed_host():
            self.respond({"ok": False, "message": "Invalid host"}, 403)
            return
        if urlparse(self.path).path != "/api/chat":
            self.respond({"ok": False, "message": "Not found"}, 404)
            return
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.respond({"ok": False, "message": "不接受跨站请求。"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond({"ok": False, "message": "请使用 application/json。"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= REQUEST_LIMIT:
                self.respond({"ok": False, "message": "请求体过大或为空。"}, 413)
                return
            self.connection.settimeout(15)
            payload = validate_payload(json.loads(self.rfile.read(length)))
        except (ValueError, TimeoutError, UnicodeError):
            self.respond({"ok": False, "message": "请求格式不正确，或超出问题／上下文限制。"}, 400)
            return
        status = health_payload()
        if not status["ready"]:
            self.respond({"ok": False, "message": "服务尚未就绪，请展开「连接与配置」查看准备步骤。",
                          "issues": status["issues"]}, 503)
            return
        if not self.server.inference_slot.acquire(blocking=False):
            self.respond({"ok": False, "message": "服务正在处理另一条问题，请稍后重试。"}, 429)
            return
        events = None
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            events = inference_events(payload, self.server.inference_timeout, self.server.stop_event)
            for event in events:
                self.wfile.write((redact(json.dumps(event, ensure_ascii=False)) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception:
            try:
                self.wfile.write(b'{"type":"error","message":"Request failed. Please retry."}\n')
                self.wfile.flush()
            except OSError:
                pass
        finally:
            if events:
                events.close()
            self.server.inference_slot.release()

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}", flush=True)


class ASAWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler] = WebUIHandler,
                 *, inference_timeout: float = 300):
        self.inference_timeout = inference_timeout
        self.inference_slot = threading.BoundedSemaphore(1)
        self.stop_event = threading.Event()
        super().__init__(address, handler)

    def server_close(self) -> None:
        self.stop_event.set()
        super().server_close()
        # Give the in-flight request time to reap its own child, including on Ctrl+C.
        if self.inference_slot.acquire(timeout=7):
            self.inference_slot.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASA API-only local web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--ask-key", action="store_true", help="Read the API key without echoing or storing it on disk.")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 900:
        parser.error("--timeout must be between 1 and 900 seconds")
    if args.ask_key:
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("DeepSeek API key (hidden): ").strip()
    server = ASAWebServer((args.host, args.port), inference_timeout=args.timeout)
    print(f"ASA api-v1.0.0 · http://{args.host}:{server.server_port}", flush=True)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: no user authentication; protect the service before exposing publicly.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
