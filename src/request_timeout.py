"""Fallback timeout for ordinary HTTP handlers.

Streaming and other known long routes stay exempt. The role-workflow run
route is also exempt: its own model client already stops at 120 seconds, and
a cold local load can pass 45 seconds and still commit. Cancelling that
request in this middleware returns 504 while the handler thread keeps going,
so the page can show a blank running step after the output was saved.

Nothing else under /api/role-workflow is exempt. This module does not call a
model.
"""

from __future__ import annotations

import asyncio
import os

from starlette.responses import JSONResponse

REQUEST_HARD_TIMEOUT = float(os.getenv("REQUEST_HARD_TIMEOUT", "45"))
_TIMEOUT_EXEMPT_PREFIXES = (
    "/api/chat",            # streaming
    "/api/shell/stream",    # SSE
    "/api/research",        # multi-minute jobs
    "/api/model/download",  # tmux setup may run pip installs
    "/api/model/probe",     # SSE; iterates models with up to 8s timeout each
    "/api/model-endpoints", # /probe sub-route also iterates models
    "/api/cookbook/setup",  # remote pacman/apt installs
    "/api/upload",          # large files
    "/api/image",           # diffusion proxies (inpaint/harmonize/upscale/etc.) — own 120s httpx timeout
    "/api/memory/audit",    # retains own 120s LLM inactivity timeout
)


def is_role_workflow_run(path: str) -> bool:
    """True only for one explicit role-workflow run URL."""
    prefix = "/api/role-workflow/projects/"
    suffix = "/run"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return False
    project_id = path[len(prefix):-len(suffix)]
    if len(project_id) != 32:
        return False
    return all(character in "0123456789abcdef" for character in project_id)


def request_hard_timeout_exempt(path: str) -> bool:
    text = path or ""
    if any(text.startswith(prefix) for prefix in _TIMEOUT_EXEMPT_PREFIXES):
        return True
    return is_role_workflow_run(text)


async def call_with_hard_timeout(request, call_next):
    path = request.url.path or ""
    if request_hard_timeout_exempt(path):
        return await call_next(request)
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_HARD_TIMEOUT)
    except asyncio.TimeoutError:
        return JSONResponse(
            {"detail": f"Request exceeded {REQUEST_HARD_TIMEOUT:.0f}s timeout"},
            status_code=504,
        )
