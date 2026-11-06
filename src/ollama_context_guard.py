"""Opt-in maximum context for configured local Ollama chat requests.

Ollama 0.40.14 treats ``OLLAMA_CONTEXT_LENGTH`` and a Modelfile ``num_ctx`` as
defaults. A native ``options.num_ctx`` replaces both. The OpenAI-compatible
``/v1/chat/completions`` body cannot carry ``num_ctx``; that server build drops
unknown fields, so writing the field there does not enforce anything.

When ``ODYSSEUS_OLLAMA_MAX_NUM_CTX`` is exactly ``2048``, local Ollama chat is
sent on the native ``/api/chat`` dialect with ``options.num_ctx`` set to
``min(discovered, 2048)``. A smaller discovered window is left alone. Unset
means the guard is off and existing provider behavior is unchanged. Any other
value is invalid: local Ollama requests are rejected before the HTTP call
rather than sent uncapped.

The guard does not apply to Ollama Cloud or to other loopback OpenAI servers
(LM Studio, llama.cpp, vLLM) unless their base URL is listed in
``ODYSSEUS_OLLAMA_CONTEXT_BASES`` or they are already addressed as native
Ollama. Port 11434 ``/v1`` is treated as local Ollama because that is Ollama's
OpenAI-compatible listener.
"""

from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlparse, urlunparse

ENV_MAX_NUM_CTX = "ODYSSEUS_OLLAMA_MAX_NUM_CTX"
ENV_CONTEXT_BASES = "ODYSSEUS_OLLAMA_CONTEXT_BASES"
ENFORCED_MAX_NUM_CTX = 2048

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


class OllamaContextGuardRejected(Exception):
    """Local Ollama chat was refused before any model request was sent."""


def guard_mode() -> str:
    """Return ``off``, ``enforce``, or ``invalid``.

    Blank or unset is off. Only the exact integer 2048 turns enforcement on.
    """
    raw = os.environ.get(ENV_MAX_NUM_CTX)
    if raw is None or raw.strip() == "":
        return "off"
    if raw.strip() == str(ENFORCED_MAX_NUM_CTX):
        return "enforce"
    return "invalid"


def _reject(message: str) -> None:
    raise OllamaContextGuardRejected(message)


def _invalid_cap_message() -> str:
    raw = os.environ.get(ENV_MAX_NUM_CTX, "")
    return (
        f"{ENV_MAX_NUM_CTX} is {raw!r}, not {ENFORCED_MAX_NUM_CTX}. "
        "Refusing this local Ollama request instead of sending it without "
        f"a context cap. Set {ENV_MAX_NUM_CTX}={ENFORCED_MAX_NUM_CTX} or unset it."
    )


def _parse_bases() -> list[str]:
    raw = os.environ.get(ENV_CONTEXT_BASES)
    if raw is None or raw.strip() == "":
        return []
    bases: list[str] = []
    for part in raw.split(","):
        item = part.strip().rstrip("/")
        if not item:
            continue
        parsed = urlparse(item)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            _reject(
                f"{ENV_CONTEXT_BASES} contains invalid URL {part.strip()!r}. "
                "Refusing local Ollama requests instead of ignoring that entry."
            )
        bases.append(item)
    return bases


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().rstrip(".")
    except Exception:
        return ""


def _is_ollama_cloud(url: str) -> bool:
    host = _hostname(url)
    return host == "ollama.com" or host.endswith(".ollama.com")


def _path(url: str) -> str:
    try:
        return (urlparse(url).path or "").rstrip("/")
    except Exception:
        return ""


def _is_native_ollama_shape(url: str) -> bool:
    """Match ``llm_core._is_ollama_native_url`` for local listeners only.

    Ollama Cloud is excluded here even though the chat detector treats it as
    native. This guard is a local context ceiling, not a cloud control.
    """
    if _is_ollama_cloud(url):
        return False
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").rstrip("/")
    if path.startswith("/v1"):
        return False
    local = host in _LOOPBACK_HOSTS or parsed.port == 11434
    return local and (path == "" or path == "/api" or path.startswith("/api/"))


def _is_port_11434_v1(url: str) -> bool:
    try:
        parsed = urlparse(url or "")
    except Exception:
        return False
    if parsed.port != 11434:
        return False
    path = (parsed.path or "").rstrip("/")
    return path == "/v1" or path.startswith("/v1/")


def _origin_and_path(url: str) -> tuple[str, int, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    return host, port, (parsed.path or "").rstrip("/")


def _matches_base(url: str, base: str) -> bool:
    try:
        host, port, path = _origin_and_path(url)
        bhost, bport, bpath = _origin_and_path(base)
    except Exception:
        return False
    if host != bhost or port != bport:
        return False
    if not bpath:
        return True
    return path == bpath or path.startswith(bpath + "/")


def is_guarded_local_ollama(url: str) -> bool:
    """Whether this URL is a local Ollama chat target while the guard is active.

    A broken ``ODYSSEUS_OLLAMA_CONTEXT_BASES`` entry refuses loopback and other
    unambiguous local Ollama URLs. Public providers are left alone.
    """
    if guard_mode() == "off" or _is_ollama_cloud(url):
        return False
    unambiguous = _is_native_ollama_shape(url) or _is_port_11434_v1(url)
    try:
        bases = _parse_bases()
    except OllamaContextGuardRejected:
        if unambiguous or _hostname(url) in _LOOPBACK_HOSTS:
            raise
        return False
    if unambiguous:
        return True
    return any(_matches_base(url, base) for base in bases)


def dispatch_provider(url: str, detected: str) -> str:
    """Provider to use for this chat call.

    Guarded local Ollama ``/v1`` URLs are forced onto the native provider so
    the request can carry ``options.num_ctx``. Other URLs keep ``detected``.
    """
    mode = guard_mode()
    if mode == "off":
        return detected
    if not is_guarded_local_ollama(url):
        return detected
    if mode == "invalid":
        _reject(_invalid_cap_message())
    return "ollama"


def native_chat_url(url: str, normalized_native: str) -> str:
    """Native ``/api/chat`` URL for a guarded or already-native Ollama endpoint.

    ``_normalize_ollama_url`` leaves a ``/v1`` base in place and would post to
    ``/v1/chat``, which is not Ollama's native API. Strip that prefix first.
    """
    if _is_native_ollama_shape(url):
        return normalized_native
    parsed = urlparse(url)
    origin = urlunparse((parsed.scheme or "http", parsed.netloc, "", "", "", ""))
    return origin.rstrip("/") + "/api/chat"


def num_ctx_for_payload(url: str, num_ctx: Optional[int]) -> Optional[int]:
    """Value to put in ``options.num_ctx``, or None to omit it.

    Omission is the historical behavior and is not a cap: Ollama then applies
    its own default, which on an 8 GB GPU is 4096 and can be overridden upward
    by a later request. Enforcement always emits an integer ``<= 2048``.
    """
    mode = guard_mode()
    if mode == "off" or not is_guarded_local_ollama(url):
        if num_ctx is not None and num_ctx > 0 and num_ctx != _default_context():
            return int(num_ctx)
        return None
    if mode == "invalid":
        _reject(_invalid_cap_message())
    if num_ctx is None or num_ctx <= 0 or num_ctx == _default_context():
        return ENFORCED_MAX_NUM_CTX
    return min(int(num_ctx), ENFORCED_MAX_NUM_CTX)


def _default_context() -> int:
    from src.model_context import DEFAULT_CONTEXT
    return DEFAULT_CONTEXT
