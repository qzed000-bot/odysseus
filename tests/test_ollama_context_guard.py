"""Dispatch tests for the opt-in local Ollama context ceiling.

The capture server binds an ephemeral loopback port and never uses 11434.
No Ollama process is started.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import HTTPException

from src import llm_core
from src.ollama_context_guard import (
    ENV_CONTEXT_BASES,
    ENV_MAX_NUM_CTX,
    is_guarded_local_ollama,
)

MODEL = "qwen2.5:7b-instruct"


class _Recorder(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.server.gets.append(self.path.split("?", 1)[0])
        self._send(200, b'{"models":[],"data":[]}', "application/json")

    def do_POST(self):
        raw = self._read_body()
        try:
            body = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            body = {"_raw": raw.decode(errors="replace")}
        path = self.path.split("?", 1)[0]
        self.server.posts.append({"path": path, "body": body})
        script = self.server.script
        status = script.pop(0) if script else 200
        if status != 200:
            self._send(status, b'{"error":"unavailable"}', "application/json")
            return
        if path.startswith("/api/"):
            line = b'{"message":{"content":"ok"},"done":true}\n'
            payload = line if body.get("stream") else line.strip()
            self._send(200, payload, "application/x-ndjson")
            return
        if body.get("stream"):
            payload = (
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self._send(200, payload, "text/event-stream")
            return
        self._send(
            200,
            b'{"choices":[{"message":{"content":"ok"}}]}',
            "application/json",
        )


@pytest.fixture
def capture():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    server.posts = []
    server.gets = []
    server.script = []
    assert server.server_address[1] != 11434
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    llm_core._response_cache.clear()
    llm_core._response_model_cache.clear()
    llm_core._dead_hosts.clear()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        llm_core._response_cache.clear()
        llm_core._dead_hosts.clear()


def _base(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def _num_ctx(body: dict):
    options = body.get("options") or {}
    return options.get("num_ctx", body.get("num_ctx"))


def test_port_11434_v1_is_local_ollama_without_a_connection(monkeypatch):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.delenv(ENV_CONTEXT_BASES, raising=False)
    assert is_guarded_local_ollama("http://127.0.0.1:11434/v1") is True
    assert is_guarded_local_ollama("http://127.0.0.1:11434/v1/chat/completions") is True
    assert llm_core._dispatch_provider("http://127.0.0.1:11434/v1") == "ollama"
    assert llm_core._ollama_post_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/api/chat"
    assert is_guarded_local_ollama("http://127.0.0.1:1234/v1") is False
    assert is_guarded_local_ollama("https://ollama.com/api") is False
    assert is_guarded_local_ollama("https://api.openai.com/v1") is False
    monkeypatch.delenv(ENV_MAX_NUM_CTX, raising=False)
    assert llm_core._dispatch_provider("http://127.0.0.1:11434/v1") == "openai"


def test_broken_base_list_refuses_local_ollama_before_post(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.setenv(ENV_CONTEXT_BASES, "not-a-url")
    with pytest.raises(HTTPException) as caught:
        llm_core.llm_call(
            _base(capture) + "/api",
            MODEL,
            [{"role": "user", "content": "bad bases"}],
            max_tokens=4,
        )
    assert caught.value.status_code == 400
    assert capture.posts == []


def test_guard_off_native_still_sends_mapped_window(monkeypatch, capture):
    monkeypatch.delenv(ENV_MAX_NUM_CTX, raising=False)
    monkeypatch.delenv(ENV_CONTEXT_BASES, raising=False)
    result = llm_core.llm_call(
        _base(capture) + "/api",
        MODEL,
        [{"role": "user", "content": "guard off"}],
        max_tokens=8,
    )
    assert result == "ok"
    body = capture.posts[-1]["body"]
    assert capture.posts[-1]["path"] == "/api/chat"
    assert _num_ctx(body) == 131072


def test_native_root_clamps_stubbed_mapped_context(monkeypatch, capture):
    """Root URLs probe context on port 80; stub that lookup and check the post."""
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.delenv(ENV_CONTEXT_BASES, raising=False)
    monkeypatch.setattr(llm_core, "get_context_length", lambda url, model: 131072)
    result = llm_core.llm_call(
        _base(capture),
        MODEL,
        [{"role": "user", "content": "native root"}],
        max_tokens=4,
    )
    assert result == "ok"
    post = capture.posts[-1]
    assert post["path"] == "/api/chat"
    assert _num_ctx(post["body"]) == 2048


def test_native_api_clamps_real_qwen_mapping(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.delenv(ENV_CONTEXT_BASES, raising=False)
    result = llm_core.llm_call(
        _base(capture) + "/api",
        MODEL,
        [{"role": "user", "content": "native api"}],
        max_tokens=4,
    )
    assert result == "ok"
    post = capture.posts[-1]
    assert post["path"] == "/api/chat"
    assert _num_ctx(post["body"]) == 2048


def test_v1_is_rewritten_to_native_and_capped(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    base = _base(capture)
    monkeypatch.setenv(ENV_CONTEXT_BASES, base + "/v1")
    result = llm_core.llm_call(
        base + "/v1",
        MODEL,
        [{"role": "user", "content": "v1 shape"}],
        max_tokens=4,
    )
    assert result == "ok"
    post = capture.posts[-1]
    assert post["path"] == "/api/chat"
    assert _num_ctx(post["body"]) == 2048
    assert "num_ctx" not in post["body"] or post["body"].get("num_ctx") is None


def test_smaller_discovered_window_is_not_raised(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.setattr(llm_core, "get_context_length", lambda url, model: 512)
    result = llm_core.llm_call(
        _base(capture) + "/api",
        MODEL,
        [{"role": "user", "content": "small window"}],
        max_tokens=4,
    )
    assert result == "ok"
    assert _num_ctx(capture.posts[-1]["body"]) == 512


def test_requested_override_cannot_exceed_cap(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.setattr(llm_core, "get_context_length", lambda url, model: 999999)
    llm_core.llm_call(
        _base(capture),
        "custom-model",
        [{"role": "user", "content": "huge request"}],
        max_tokens=4,
    )
    assert _num_ctx(capture.posts[-1]["body"]) == 2048


@pytest.mark.parametrize("raw", ["4096", "0", "no", "2048abc"])
def test_invalid_configuration_rejects_before_dispatch(monkeypatch, capture, raw):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, raw)
    with pytest.raises(HTTPException) as caught:
        llm_core.llm_call(
            _base(capture) + "/api",
            MODEL,
            [{"role": "user", "content": f"invalid {raw}"}],
            max_tokens=4,
        )
    assert caught.value.status_code == 400
    assert "2048" in str(caught.value.detail)
    assert capture.posts == []


def test_blank_whitespace_enables_when_value_is_2048(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, " 2048 ")
    llm_core.llm_call(
        _base(capture) + "/api",
        MODEL,
        [{"role": "user", "content": "padded 2048"}],
        max_tokens=4,
    )
    assert _num_ctx(capture.posts[-1]["body"]) == 2048


def test_other_loopback_openai_server_is_not_rerouted(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.delenv(ENV_CONTEXT_BASES, raising=False)
    result = llm_core.llm_call(
        _base(capture) + "/v1",
        "lmstudio-model",
        [{"role": "user", "content": "leave this openai"}],
        max_tokens=4,
    )
    assert result == "ok"
    post = capture.posts[-1]
    assert post["path"] == "/v1/chat/completions"
    assert "num_ctx" not in post["body"]
    assert "options" not in post["body"]


def test_public_openai_is_unchanged_when_guard_is_on(monkeypatch):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "cloud"}}]},
        )

    monkeypatch.setattr(llm_core.httpx, "post", fake_post)
    llm_core._response_cache.clear()
    result = llm_core.llm_call(
        "https://api.openai.com/v1",
        "gpt-4o",
        [{"role": "user", "content": "remote"}],
        max_tokens=4,
    )
    assert result == "cloud"
    assert seen["url"].startswith("https://api.openai.com/")
    assert "num_ctx" not in seen["json"]
    assert "options" not in seen["json"]


def test_ollama_cloud_is_not_capped(monkeypatch):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"content": "cloud"}, "done": True},
        )

    monkeypatch.setattr(llm_core.httpx, "post", fake_post)
    llm_core._response_cache.clear()
    result = llm_core.llm_call(
        "https://ollama.com/api",
        "gpt-oss:120b-test",
        [{"role": "user", "content": "cloud ollama"}],
        temperature=0.2,
        max_tokens=7,
    )
    assert result == "cloud"
    assert seen["url"] == "https://ollama.com/api/chat"
    assert "num_ctx" not in (seen["json"].get("options") or {})


def test_fallback_caps_only_the_local_candidate(monkeypatch):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    monkeypatch.setattr(llm_core, "get_context_length", lambda url, model: 131072)
    seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        seen.append((url, json))
        request = httpx.Request("POST", url)
        if "127.0.0.1" in url:
            return httpx.Response(503, request=request, json={"error": "down"})
        return httpx.Response(
            200,
            request=request,
            json={"choices": [{"message": {"content": "next"}}]},
        )

    monkeypatch.setattr(llm_core.httpx, "post", fake_post)
    llm_core._response_cache.clear()
    result = llm_core.llm_call_with_fallback(
        [
            ("http://127.0.0.1:18081/api", MODEL, None),
            ("https://api.openai.com/v1", "gpt-4o", None),
        ],
        [{"role": "user", "content": "fallback"}],
        max_tokens=4,
    )
    assert result == "next"
    assert seen[0][0] == "http://127.0.0.1:18081/api/chat"
    assert _num_ctx(seen[0][1]) == 2048
    assert "num_ctx" not in seen[1][1]
    assert seen[1][0].startswith("https://api.openai.com/")


async def test_streaming_native_and_v1_are_capped(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    base = _base(capture)
    monkeypatch.setenv(ENV_CONTEXT_BASES, base + "/v1")
    for endpoint, prompt in ((base + "/api", "stream native"), (base + "/v1", "stream v1")):
        llm_core._response_cache.clear()
        chunks = []
        async for chunk in llm_core.stream_llm(
            endpoint,
            MODEL,
            [{"role": "user", "content": prompt}],
            max_tokens=4,
        ):
            chunks.append(chunk)
        assert any("ok" in chunk for chunk in chunks)
    assert [post["path"] for post in capture.posts] == ["/api/chat", "/api/chat"]
    assert all(post["body"].get("stream") is True for post in capture.posts)
    assert all(_num_ctx(post["body"]) == 2048 for post in capture.posts)


async def test_async_retry_reuses_capped_body(monkeypatch, capture):
    monkeypatch.setenv(ENV_MAX_NUM_CTX, "2048")
    capture.script = [503]
    result = await llm_core.llm_call_async(
        _base(capture) + "/api",
        MODEL,
        [{"role": "user", "content": "retry"}],
        max_tokens=4,
        max_retries=2,
    )
    assert result == "ok"
    assert len(capture.posts) == 2
    assert all(_num_ctx(post["body"]) == 2048 for post in capture.posts)
