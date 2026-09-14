import http.client
import json
import subprocess
import sys
import threading

import pytest

import run_api_web as web


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(web, "health_payload", lambda connection=None: {
        "ok": True, "ready": True, "model": "deepseek-flash", "checks": {}, "issues": [],
    })
    instance = web.ASAWebServer(("127.0.0.1", 0))
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=2)


def request(server, method, path, payload=None, **headers):
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    body = json.dumps(payload) if payload is not None else None
    if body is not None:
        headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body, headers)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_static_assets_and_security_headers(server):
    for path in ("/", "/styles.css", "/app.js"):
        status, headers, body = request(server, "GET", path)
        assert status == 200 and body
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


@pytest.mark.parametrize("path", ["/.env", "/configs/runtime_api_stable.json", "/../scripts/run_api_web.py",
                                 "/%2e%2e/.env", "/api/service/start"])
def test_non_public_paths_are_not_exposed(server, path):
    assert request(server, "GET", path)[0] == 404


def test_cross_origin_rebinding_and_wrong_content_type_rejected(server):
    assert request(server, "POST", "/api/chat", {"message": "问题"}, Origin="https://evil.example")[0] == 403
    assert request(server, "GET", "/api/health", Host="evil.example")[0] == 403
    assert request(server, "POST", "/api/chat", {"message": "问题"}, **{"Content-Type": "text/plain"})[0] == 415


def test_request_overrides_never_reach_inference(server, monkeypatch):
    def forbidden(*args):
        raise AssertionError("must not invoke inference")
    monkeypatch.setattr(web, "inference_events", forbidden)
    assert request(server, "POST", "/api/chat", {"message": "问题", "model": "local-4b"})[0] == 400


def test_missing_setup_is_503_not_a_successful_answer(server, monkeypatch):
    monkeypatch.setattr(web, "health_payload", lambda connection=None: {"ready": False, "issues": [{"id": "api_key"}]})
    status, _, body = request(server, "POST", "/api/chat", {"message": "问题"})
    assert status == 503
    assert json.loads(body)["ok"] is False


def test_no_invisible_queue_when_inference_is_busy(server):
    server.inference_slot.acquire()
    try:
        assert request(server, "POST", "/api/chat", {"message": "问题"})[0] == 429
    finally:
        server.inference_slot.release()


def test_complete_ndjson_response_and_slot_release(server, monkeypatch):
    received = []
    def events(payload, timeout, cancelled=None, connection=None):
        received.append(payload)
        yield {"type": "stage", "message": "检索"}
        yield {"type": "result", "ok": True, "answer": "完整答案", "evidence": []}
    monkeypatch.setattr(web, "inference_events", events)
    status, headers, body = request(server, "POST", "/api/chat", {"message": "真实路由测试", "history": []})
    assert status == 200
    assert "ndjson" in headers["Content-Type"]
    frames = [json.loads(line) for line in body.splitlines()]
    assert frames[-1]["answer"] == "完整答案"
    assert received == [{"message": "真实路由测试", "history": []}]
    assert server.inference_slot.acquire(blocking=False)
    server.inference_slot.release()


def test_real_subprocess_stdout_is_parsed_without_loading_models(monkeypatch):
    program = 'import json; print("API ready"); print(json.dumps({"answer":"mock result","evidence":[]}))'
    monkeypatch.setattr(web, "build_command", lambda payload, connection=None: [sys.executable, "-u", "-c", program])
    events = list(web.inference_events({"message": "测试", "history": []}, 5))
    assert events[-1]["type"] == "result"
    assert events[-1]["answer"] == "mock result"


def test_closing_stream_stops_only_its_own_subprocess(monkeypatch):
    original = subprocess.Popen
    children = []
    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(web.subprocess, "Popen", track)
    monkeypatch.setattr(web, "build_command", lambda payload, connection=None: [
        sys.executable, "-u", "-c", 'import time; print("ready",flush=True); time.sleep(8)',
    ])
    stream = web.inference_events({"message": "测试", "history": []}, 5)
    assert next(stream)["type"] == "stage"
    stream.close()
    assert len(children) == 1
    assert children[0].poll() is not None


def test_timeout_is_an_error_and_stops_the_request_process(monkeypatch):
    monkeypatch.setattr(web, "build_command", lambda payload, connection=None: [
        sys.executable, "-u", "-c", 'import time; print("ready",flush=True); time.sleep(8)',
    ])
    events = list(web.inference_events({"message": "测试", "history": []}, 0.1))
    assert events[-1]["type"] == "error"
    assert not any(event["type"] == "result" for event in events)


def test_shutdown_prevents_starting_a_new_subprocess(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("shutdown must not start a subprocess")
    monkeypatch.setattr(web.subprocess, "Popen", forbidden)
    cancelled = threading.Event()
    cancelled.set()
    events = list(web.inference_events({"message": "测试", "history": []}, 5, cancelled))
    assert events == [{"type": "error", "message": "服务正在关闭，未发送请求。"}]
