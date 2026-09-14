"""Custom API contract; only synthetic credentials and loopback mock endpoints."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

import run_api_inference as api
import run_api_web as web
from test_api_web_http import request, server  # noqa: F401


def current():
    return web.APIConnection("https://api.deepseek.com/chat/completions", "deepseek-flash",
                             "synthetic-original-credential", request_profile="deepseek")


def settings(**changes):
    return {"api_base_url": "https://compatible.example/v1", "model": "vendor/model",
            "api_key": "synthetic-new-credential", "request_profile": "standard", **changes}


@pytest.mark.parametrize("url,expected", [
    ("https://API.Example:443/v1/", "https://api.example/v1/chat/completions"),
    ("https://custom-api.example/prefix/v1/chat/completions/", "https://custom-api.example/prefix/v1/chat/completions"),
    ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ("http://[::1]:8000/v1", "http://[::1]:8000/v1/chat/completions"),
])
def test_endpoint_normalization(url, expected):
    assert web.normalize_endpoint(url) == expected


@pytest.mark.parametrize("url", [
    None, "", "file:///tmp/secret", "https://user:pass@api.example/v1",
    "https://api.example/v1?key=secret", "https://api.example/v1#secret",
    "https://api.example/v1?", "https://api.example:70000/v1",
    "https://api.example:0/v1", "https://api.example/\nother",
    "https://api.example\\@evil.example/v1", "http://api.example/v1",
    "http://192.168.1.20/v1", "http://127.0.0.1.evil.example/v1",
])
def test_unsafe_or_malformed_endpoints_rejected(url):
    with pytest.raises(ValueError):
        web.normalize_endpoint(url)


@pytest.mark.parametrize("changes", [
    {"model": ""}, {"model": "a\nb"}, {"model": "a" * 201},
    {"api_key": "space in key"}, {"api_key": "非ASCII"}, {"api_key": "a" * 4097},
    {"no_auth": "true"}, {"request_profile": "arbitrary"}, {"runtime_config": "/tmp/config"},
    {"no_auth": True, "api_key": ""},
])
def test_bad_settings_rejected(changes):
    with pytest.raises(ValueError):
        web.validate_settings(settings(**changes), current())


def test_address_change_cannot_reuse_key_even_on_same_host():
    for endpoint in ("https://other.example/v1", "https://api.deepseek.com/other-path"):
        with pytest.raises(ValueError, match="地址已改变"):
            web.validate_settings(settings(api_base_url=endpoint, api_key=""), current())
    updated = web.validate_settings(settings(api_base_url="https://api.deepseek.com/", api_key=""), current())
    assert updated.api_key == current().api_key
    assert updated.model == "vendor/model"
    assert updated.request_profile == "standard"


def test_local_no_auth_is_explicit_and_clears_key():
    updated = web.validate_settings(settings(api_base_url="http://localhost:8000/v1",
                                            no_auth=True, api_key=""), current())
    assert updated.no_auth and not updated.api_key
    assert "--allow-no-api-key" in web.build_command({"message": "question", "history": []}, updated)
    with pytest.raises(ValueError):
        web.validate_settings(settings(api_base_url="http://localhost:8000/v1", no_auth=True), current())


def test_repr_and_public_state_never_return_credentials():
    connection = current()
    assert connection.api_key not in repr(connection)
    assert connection.api_key not in json.dumps(connection.public())
    assert connection.public()["key_configured"]
    key = 'synthetic-"escaped"-credential'
    assert key not in web.redact(key, key)
    assert "escaped" not in web.redact(json.dumps(key), key)
    value = web.redact_values({"type": "result", "answer": "result", "evidence": []}, "result")
    assert value == {"type": "result", "answer": "[redacted]", "evidence": []}


def test_saved_settings_affect_health_and_each_request_without_leaking_keys(server, monkeypatch):
    server.api_connection = current()
    initial = json.loads(request(server, "GET", "/api/settings")[2])
    assert initial["model"] == "deepseek-flash" and "api_key" not in initial
    status, _, body = request(server, "POST", "/api/settings", settings())
    assert status == 200 and b"synthetic-new-credential" not in body
    assert json.loads(body)["api_base_url"] == "https://compatible.example/v1/chat/completions"
    snapshots = []

    def events(payload, timeout, cancelled, connection):
        snapshots.append(connection)
        yield {"type": "result", "ok": True, "answer": connection.api_key, "model": connection.model}

    monkeypatch.setattr(web, "inference_events", events)
    status, _, body = request(server, "POST", "/api/chat", {"message": "synthetic question"})
    assert status == 200 and b"synthetic-new-credential" not in body
    assert json.loads(body)["model"] == "vendor/model"
    assert snapshots[0].api_key == "synthetic-new-credential"
    assert snapshots[0].request_profile == "standard"


def test_failed_change_keeps_original_settings_and_busy_blocks_changes(server):
    server.api_connection = current()
    status, _, body = request(server, "POST", "/api/settings", settings(api_key=""))
    assert status == 400 and "重新输入" in json.loads(body)["message"]
    assert server.api_connection == current()
    server.inference_slot.acquire()
    try:
        assert request(server, "POST", "/api/settings", settings())[0] == 409
        assert server.api_connection == current()
    finally:
        server.inference_slot.release()


def test_settings_reject_cross_site_and_remote_clients(server, monkeypatch):
    for headers in ({"Origin": "https://evil.example"}, {"Sec-Fetch-Site": "cross-site"},
                    {"Host": "evil.example"}, {"Origin": "null"}):
        assert request(server, "POST", "/api/settings", settings(), **headers)[0] == 403
    assert request(server, "POST", "/api/settings", settings(), **{"Content-Type": "text/plain"})[0] == 415
    monkeypatch.setattr(web, "is_loopback", lambda host: False)
    assert request(server, "GET", "/api/settings")[0] == 403
    assert request(server, "POST", "/api/settings", settings())[0] == 403


def test_health_uses_current_api_not_default_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-env-credential")
    connection = web.APIConnection("http://localhost:8000/v1/chat/completions", "local-test", no_auth=True)
    data = web.health_payload(connection)
    assert data["model"] == "local-test" and data["api_host"] == "localhost:8000"
    assert data["checks"]["api_key"] and data["thinking"] is None
    assert "synthetic-env-credential" not in json.dumps(data)
    empty = web.APIConnection("https://other.example/v1/chat/completions", "other-test")
    assert not web.health_payload(empty)["checks"]["api_key"]


def test_actual_child_receives_only_selected_key_and_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "synthetic-original-credential")
    connection = web.validate_settings(settings(), current())
    command = web.build_command({"message": "--model accidental-option", "history": []}, connection)
    assert command[-2:] == ["--", "--model accidental-option"]
    assert "--model=vendor/model" in command
    assert "--api-extra-body={}" in command
    assert connection.api_key not in " ".join(command)
    program = (
        "import os,json; "
        "assert os.environ['ASA_REQUEST_API_KEY']=='synthetic-new-credential'; "
        "assert 'DEEPSEEK_API_KEY' not in os.environ; "
        "print(json.dumps({'answer':'validated child environment','evidence':[]}))"
    )
    monkeypatch.setattr(web, "build_command", lambda payload, config: [sys.executable, "-c", program])
    result = list(web.inference_events({"message": "test", "history": []}, 5, connection=connection))[-1]
    assert result["type"] == "result" and result["model"] == "vendor/model"


@pytest.fixture
def mock_provider():
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), payload))
            if self.path.startswith("/redirect"):
                self.send_response(307)
                self.send_header("Location", "/should-not-receive")
                self.end_headers()
                return
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "mock answer"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_port}", requests
    instance.shutdown()
    instance.server_close()
    thread.join(2)


@pytest.mark.parametrize("profile,no_auth", [("standard", False), ("deepseek", False), ("standard", True)])
def test_selected_command_reaches_real_mock_http_provider(mock_provider, monkeypatch, profile, no_auth):
    base, received = mock_provider
    connection = web.validate_settings(settings(api_base_url=base + "/v1", no_auth=no_auth,
                                                api_key="" if no_auth else "synthetic-new-credential",
                                                request_profile=profile), current())
    command = web.build_command({"message": "question", "history": []}, connection)
    with patch.object(sys, "argv", command[1:]):
        args = api.parse_args()
    runner = api.OpenAICompatibleAPIRunner(
        api_base_url=args.api_base_url, api_key=connection.api_key,
        api_key_env=args.api_key_env, model=args.model,
        extra_body=args.api_extra_body, response_format_json=False, prompt_appendix=False,
    )
    assert runner.generate("mock evidence") == "mock answer"
    assert len(received) == 1
    path, authorization, payload = received[0]
    assert path == "/v1/chat/completions"
    assert authorization == (None if no_auth else "Bearer synthetic-new-credential")
    assert payload["model"] == "vendor/model"
    assert (payload.get("thinking") == {"type": "disabled"}) if profile == "deepseek" else ("thinking" not in payload)
    assert "response_format" not in payload
    assert payload["messages"][-1]["content"] == "mock evidence"


def test_api_redirects_do_not_forward_keys_or_evidence(mock_provider):
    base, received = mock_provider
    runner = api.OpenAICompatibleAPIRunner(api_base_url=base + "/redirect",
                                          api_key="synthetic-only-credential", api_key_env="TEST",
                                          model="mock-model")
    with pytest.raises(RuntimeError, match="HTTP 307"):
        runner.generate("mock evidence")
    assert len(received) == 1
