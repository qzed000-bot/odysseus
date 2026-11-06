"""Simulated tests for the sequential text workflow.

These tests never open a real network connection and never call a model.
Most completion tests replace ``httpx.post``. The proxy test uses the real
client and records the address the socket layer was asked to connect to.
Port 11434 is not bound.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.database import ModelEndpoint, ScheduledTask, SessionLocal
from routes.role_workflow_routes import setup_role_workflow_routes
from src import auth_helpers, role_workflow
from src.role_workflow import (
    RoleProject,
    RoleStep,
    WorkflowClosed,
    cancel_project,
    create_project,
    ensure_tables,
    feature_enabled,
    get_project,
    recover_project,
    run_next,
)

ROOT = Path(__file__).resolve().parents[1]
PIN = "http://127.0.0.1:11434/v1"
MODEL = "qwen2.5:7b-instruct"
OWNER = "daily"


@pytest.fixture(scope="module", autouse=True)
def shared_sqlite(tmp_path_factory):
    """One file database for the main thread and request threads.

    The process-wide test engine is sqlite memory, and that pool is one
    database per thread. These tests cross threads, so they need one file.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from core.database import Base

    path = tmp_path_factory.mktemp("role-workflow") / "app.db"
    eng = create_engine(
        "sqlite:///" + path.as_posix(),
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    Base.metadata.create_all(eng)
    patch = pytest.MonkeyPatch()
    patch.setattr(role_workflow, "SessionLocal", session_factory)
    patch.setattr(role_workflow, "engine", eng)
    patch.setattr(sys.modules[__name__], "SessionLocal", session_factory)
    role_workflow._TABLES_READY = False
    ensure_tables()
    yield
    patch.undo()
    role_workflow._TABLES_READY = False
    eng.dispose()


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("network socket blocked")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(autouse=True)
def workflow_env(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_ROLE_WORKFLOW", "1")
    monkeypatch.setenv("ODYSSEUS_OLLAMA_MAX_NUM_CTX", "2048")
    monkeypatch.delenv("ODYSSEUS_OLLAMA_CONTEXT_BASES", raising=False)
    monkeypatch.delenv("ODYSSEUS_ROLE_WORKFLOW_MODELS", raising=False)
    ensure_tables()
    yield
    if role_workflow._RUN_LOCK.locked():
        try:
            role_workflow._RUN_LOCK.release()
        except RuntimeError:
            pass


@pytest.fixture
def client():
    app = FastAPI()

    @app.middleware("http")
    async def test_user(request, call_next):
        user = request.headers.get("x-test-user")
        if user:
            request.state.current_user = user
        if request.headers.get("x-test-api-token") == "1":
            request.state.api_token = True
        return await call_next(request)

    app.include_router(setup_role_workflow_routes())
    with TestClient(app) as test_client:
        test_client.headers.update({"x-test-user": OWNER})
        yield test_client


def _endpoint(base_url=PIN, owner=None, enabled=True, model_type="llm") -> str:
    db = SessionLocal()
    try:
        endpoint_id = uuid.uuid4().hex
        db.add(ModelEndpoint(
            id=endpoint_id,
            name="local-ollama",
            base_url=base_url,
            is_enabled=enabled,
            owner=owner,
            model_type=model_type,
        ))
        db.commit()
        return endpoint_id
    finally:
        db.close()


def _reply(url, text):
    return httpx.Response(
        200,
        json={"message": {"role": "assistant", "content": text}},
        request=httpx.Request("POST", url),
    )


def _install_replies(monkeypatch, replies):
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, "json": kwargs.get("json"), "kwargs": kwargs})
        if not replies:
            raise AssertionError("unexpected completion")
        item = replies.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(url, **kwargs)
        return _reply(url, item)

    monkeypatch.setattr(role_workflow.httpx, "post", post)
    return calls


def _boom(*_args, **_kwargs):
    raise AssertionError("hidden model path was called")


def _block_hidden_paths(monkeypatch):
    import src.llm_core as llm_core
    import src.model_context as model_context

    for name in (
        "llm_call",
        "llm_call_async",
        "llm_call_with_fallback",
        "llm_call_async_with_fallback",
        "get_context_length",
    ):
        if hasattr(llm_core, name):
            monkeypatch.setattr(llm_core, name, _boom)
    monkeypatch.setattr(model_context, "get_context_length", _boom)


def _create(endpoint_id=None, brief="Explain in a few sentences how rain forms.", title="Rain notes"):
    return create_project(
        owner=OWNER,
        title=title,
        brief=brief,
        endpoint_id=endpoint_id or _endpoint(),
        model=MODEL,
    )


def _step(project, role):
    return next(step for step in project["steps"] if step["role"] == role)


def test_feature_is_off_unless_exact_one(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_ROLE_WORKFLOW", raising=False)
    assert feature_enabled() is False
    for value in ("", "0", "true", "yes", "2"):
        monkeypatch.setenv("ODYSSEUS_ROLE_WORKFLOW", value)
        assert feature_enabled() is False
    monkeypatch.setenv("ODYSSEUS_ROLE_WORKFLOW", " 1 ")
    assert feature_enabled() is True


def test_feature_default_off_makes_no_model_call(monkeypatch, client):
    monkeypatch.delenv("ODYSSEUS_ROLE_WORKFLOW", raising=False)
    before = SessionLocal().query(RoleProject).count()
    SessionLocal().close()
    assert client.get("/api/role-workflow/status").json() == {"enabled": False}
    created = client.post("/api/role-workflow/projects", json={
        "title": "Rain notes",
        "brief": "Explain rain.",
        "endpoint_id": "missing",
        "model": MODEL,
        "owner": "root",
    })
    assert created.status_code == 404
    after = SessionLocal().query(RoleProject).count()
    SessionLocal().close()
    assert after == before


def test_unauthenticated_nonadmin_and_injected_owner(monkeypatch, client):
    def privilege_denied(*_args, **_kwargs):
        raise AssertionError("privilege was consulted")

    monkeypatch.setattr(auth_helpers, "require_privilege", privilege_denied)
    endpoint_id = _endpoint()
    body = {
        "title": "<script>alert(1)</script>",
        "brief": "Explain rain.",
        "endpoint_id": endpoint_id,
        "model": MODEL,
        "owner": "root",
        "status": "completed",
        "output_text": "should-not-stick",
        "endpoint_url": "https://example.com/v1",
    }
    with TestClient(client.app) as anon:
        assert anon.get("/api/role-workflow/projects").status_code == 401
    with TestClient(client.app) as token:
        token.headers["x-test-api-token"] = "1"
        assert token.get("/api/role-workflow/projects").status_code == 403

    created = client.post("/api/role-workflow/projects", json=body)
    assert created.status_code == 200
    project = created.json()
    assert "owner" not in project
    assert project["title"] == "<script>alert(1)</script>"
    assert _step(project, "writer")["status"] == "pending"
    assert _step(project, "writer")["output_text"] == ""
    assert project["endpoint_url"] == PIN

    client.headers["x-test-user"] = "someone-else"
    assert client.get("/api/role-workflow/projects").json()["projects"] == []
    assert client.get(f"/api/role-workflow/projects/{project['id']}").status_code == 404
    client.headers["x-test-user"] = OWNER
    assert client.get(f"/api/role-workflow/projects/{project['id']}").status_code == 200

    db = SessionLocal()
    try:
        row = db.query(RoleProject).filter(RoleProject.id == project["id"]).one()
        assert row.owner == OWNER
    finally:
        db.close()


def test_writer_critic_organiser_handoff_and_cap(monkeypatch):
    _block_hidden_paths(monkeypatch)
    calls = _install_replies(monkeypatch, ["DRAFT-ALPHA", "DECISION-BETA", "PLAN-GAMMA"])
    project = _create()
    before_tasks = SessionLocal().query(ScheduledTask).count()
    SessionLocal().close()

    writer = run_next(OWNER, project["id"])
    critic = run_next(OWNER, project["id"])
    organiser = run_next(OWNER, project["id"])

    assert _step(writer, "writer")["output_text"] == "DRAFT-ALPHA"
    assert _step(writer, "critic")["status"] == "pending"
    assert "DRAFT-ALPHA" in calls[1]["json"]["messages"][1]["content"]
    assert "DECISION-BETA" not in calls[1]["json"]["messages"][1]["content"]
    organiser_input = calls[2]["json"]["messages"][1]["content"]
    assert "DRAFT-ALPHA" in organiser_input
    assert "DECISION-BETA" in organiser_input
    assert _step(organiser, "writer")["output_text"] == "DRAFT-ALPHA"
    assert _step(organiser, "critic")["output_text"] == "DECISION-BETA"
    assert _step(organiser, "organiser")["output_text"] == "PLAN-GAMMA"
    assert organiser["current_artifact"] == "PLAN-GAMMA"
    assert organiser["status"] == "completed"
    assert len(calls) == 3
    for call in calls:
        assert call["url"] == "http://127.0.0.1:11434/api/chat"
        assert call["kwargs"]["follow_redirects"] is False
        assert call["kwargs"]["trust_env"] is False
        assert call["kwargs"]["proxy"] is None
        assert call["json"]["model"] == MODEL
        assert call["json"]["stream"] is False
        assert "tools" not in call["json"]
        assert call["json"]["options"]["num_ctx"] == 2048
        assert call["json"]["options"]["num_predict"] == 512
        assert all("tool_calls" not in message for message in call["json"]["messages"])
    with pytest.raises(WorkflowClosed) as done:
        run_next(OWNER, project["id"])
    assert done.value.status_code == 409
    assert len(calls) == 3
    after_tasks = SessionLocal().query(ScheduledTask).count()
    SessionLocal().close()
    assert after_tasks == before_tasks


def test_failure_cancel_and_oversize_do_not_advance_or_erase(monkeypatch):
    calls = _install_replies(monkeypatch, [
        "DRAFT-KEEP",
        httpx.ConnectError("down"),
        "DECISION-KEEP",
    ])
    project = _create()
    saved = run_next(OWNER, project["id"])
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    failed = get_project(OWNER, project["id"])
    assert _step(failed, "writer")["output_text"] == "DRAFT-KEEP"
    assert _step(failed, "critic")["status"] == "failed"
    assert _step(failed, "critic")["output_text"] == ""
    assert _step(failed, "organiser")["status"] == "pending"
    assert failed["current_artifact"] == "DRAFT-KEEP"
    assert len(calls) == 2

    retried = run_next(OWNER, project["id"])
    assert _step(retried, "writer")["output_text"] == "DRAFT-KEEP"
    assert _step(retried, "critic")["output_text"] == "DECISION-KEEP"
    assert len(calls) == 3

    started = threading.Event()
    release = threading.Event()
    recorded = []

    def post(url, **kwargs):
        recorded.append(kwargs.get("json"))
        started.set()
        assert release.wait(5)
        return _reply(url, "DISCARD-ME")

    monkeypatch.setattr(role_workflow.httpx, "post", post)
    held = {"error": None}

    def worker():
        try:
            run_next(OWNER, project["id"])
        except WorkflowClosed as exc:
            held["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(5)
    assert cancel_project(OWNER, project["id"])["cancel_requested"] is True
    release.set()
    thread.join(5)
    assert held["error"] is not None
    cancelled = get_project(OWNER, project["id"])
    assert _step(cancelled, "writer")["output_text"] == "DRAFT-KEEP"
    assert _step(cancelled, "critic")["output_text"] == "DECISION-KEEP"
    assert _step(cancelled, "organiser")["status"] == "interrupted"
    assert _step(cancelled, "organiser")["output_text"] == ""
    assert len(recorded) == 1

    huge = "n" * 4000
    oversized = _create(brief=huge, title="Too long to send")
    with pytest.raises(WorkflowClosed) as rejected:
        run_next(OWNER, oversized["id"])
    assert rejected.value.status_code == 400
    kept = get_project(OWNER, oversized["id"])
    assert kept["brief"] == huge
    assert _step(kept, "writer")["status"] == "failed"
    assert _step(kept, "writer")["output_text"] == ""
    assert len(recorded) == 1


def test_tool_like_text_is_inert(monkeypatch):
    payload = '{"name":"bash","arguments":{"cmd":"rm -rf /"}}'

    def toolish(url, **_kwargs):
        return httpx.Response(
            200,
            json={"message": {"content": payload, "tool_calls": [{"function": {"name": "bash"}}]}},
            request=httpx.Request("POST", url),
        )

    calls = _install_replies(monkeypatch, [toolish])
    project = _create()
    saved = run_next(OWNER, project["id"])
    assert _step(saved, "writer")["output_text"] == payload
    assert _step(saved, "critic")["status"] == "pending"

    def empty_tools(url, **_kwargs):
        return httpx.Response(
            200,
            json={"message": {"content": "", "tool_calls": [{"function": {"name": "bash"}}]}},
            request=httpx.Request("POST", url),
        )

    calls = _install_replies(monkeypatch, [empty_tools])
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    failed = get_project(OWNER, project["id"])
    assert _step(failed, "writer")["output_text"] == payload
    assert _step(failed, "critic")["status"] == "failed"
    assert _step(failed, "critic")["output_text"] == ""


def test_redirect_and_http_error_do_not_follow_or_advance(monkeypatch):
    def redirect(url, **_kwargs):
        return httpx.Response(
            302,
            headers={"location": "http://example.com/v1/chat/completions"},
            request=httpx.Request("POST", url),
        )

    calls = _install_replies(monkeypatch, [redirect])
    project = _create()
    with pytest.raises(WorkflowClosed) as redirected:
        run_next(OWNER, project["id"])
    assert "redirect" in redirected.value.message
    assert len(calls) == 1
    kept = get_project(OWNER, project["id"])
    assert _step(kept, "writer")["status"] == "failed"
    assert _step(kept, "writer")["output_text"] == ""

    def crashed(url, **_kwargs):
        return httpx.Response(500, text="nope", request=httpx.Request("POST", url))

    calls = _install_replies(monkeypatch, ["DRAFT-STAYS", crashed])
    project = _create(title="Error path")
    run_next(OWNER, project["id"])
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    failed = get_project(OWNER, project["id"])
    assert _step(failed, "writer")["output_text"] == "DRAFT-STAYS"
    assert _step(failed, "critic")["status"] == "failed"
    assert len(calls) == 2


def test_concurrent_runs_and_projects_serialize(monkeypatch):
    first = _create(title="First")
    second = _create(title="Second")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs.get("json"))
        started.set()
        assert release.wait(5)
        return _reply(url, "ONLY-ONE")

    monkeypatch.setattr(role_workflow.httpx, "post", post)
    results = []

    def worker(project_id):
        try:
            run_next(OWNER, project_id)
            results.append((project_id, 200))
        except WorkflowClosed as exc:
            results.append((project_id, exc.status_code))

    threads = [
        threading.Thread(target=worker, args=(first["id"],)),
        threading.Thread(target=worker, args=(first["id"],)),
        threading.Thread(target=worker, args=(second["id"],)),
    ]
    try:
        threads[0].start()
        assert started.wait(5)
        threads[1].start()
        threads[2].start()
        threads[1].join(5)
        threads[2].join(5)
        assert sorted(code for _pid, code in results) == [409, 409]
        assert len(calls) == 1
    finally:
        release.set()
        threads[0].join(5)
    assert [code for _pid, code in results].count(200) == 1
    followed = run_next(OWNER, second["id"])
    assert _step(followed, "writer")["output_text"] == "ONLY-ONE"
    assert len(calls) == 2
    busy = role_workflow._RUN_LOCK
    assert busy.acquire(blocking=False)
    try:
        with pytest.raises(WorkflowClosed) as blocked:
            run_next(OWNER, first["id"])
        assert blocked.value.status_code == 409
    finally:
        busy.release()


def test_pin_tamper_allowlist_and_local_url_policy(monkeypatch):
    calls = _install_replies(monkeypatch, ["SHOULD-NOT-SEND"])
    endpoint_id = _endpoint()
    project = _create(endpoint_id=endpoint_id)

    db = SessionLocal()
    try:
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).one()
        endpoint.base_url = "http://192.168.1.5:11434/v1"
        db.commit()
    finally:
        db.close()
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    assert calls == []
    assert get_project(OWNER, project["id"])["brief"].startswith("Explain")

    db = SessionLocal()
    try:
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).one()
        endpoint.base_url = PIN
        row = db.query(RoleProject).filter(RoleProject.id == project["id"]).one()
        row.model = "other:model"
        row_step = db.query(RoleStep).filter(RoleStep.project_id == project["id"], RoleStep.role == "writer").one()
        row_step.status = "pending"
        row_step.error = ""
        db.commit()
    finally:
        db.close()
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    assert calls == []

    db = SessionLocal()
    try:
        row = db.query(RoleProject).filter(RoleProject.id == project["id"]).one()
        row.model = MODEL
        endpoint = db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).one()
        endpoint.is_enabled = False
        step = db.query(RoleStep).filter(RoleStep.project_id == project["id"], RoleStep.role == "writer").one()
        step.status = "pending"
        db.commit()
    finally:
        db.close()
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])

    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == endpoint_id).delete()
        step = db.query(RoleStep).filter(RoleStep.project_id == project["id"], RoleStep.role == "writer").one()
        step.status = "pending"
        db.commit()
    finally:
        db.close()
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, project["id"])
    assert calls == []

    monkeypatch.setenv("ODYSSEUS_ROLE_WORKFLOW_MODELS", "other:model")
    fresh = _endpoint()
    with pytest.raises(WorkflowClosed):
        _create(endpoint_id=fresh)
    monkeypatch.delenv("ODYSSEUS_ROLE_WORKFLOW_MODELS", raising=False)

    rejected_urls = [
        "https://127.0.0.1:11434/v1",
        "http://user:pass@127.0.0.1:11434/v1",
        "http://127.0.0.1:11434/v1?next=http://example.com",
        "http://127.0.0.1:11435/v1",
        "http://0.0.0.0:11434/v1",
        "http://127.0.0.1:1234/v1",
        "http://example.com:11434/v1",
        "http://10.1.2.3:80/v1",
        "http://127.0.0.1:11434/admin",
    ]
    for url in rejected_urls:
        with pytest.raises(WorkflowClosed):
            _create(endpoint_id=_endpoint(base_url=url))

    monkeypatch.setenv("ODYSSEUS_OLLAMA_CONTEXT_BASES", "http://127.0.0.1:11435/v1,http://10.1.2.3:11434/v1")
    with pytest.raises(WorkflowClosed):
        _create(endpoint_id=_endpoint(base_url="http://127.0.0.1:11435/v1"))
    with pytest.raises(WorkflowClosed):
        _create(endpoint_id=_endpoint(base_url="http://10.1.2.3:11434/v1"))
    assert calls == []

    monkeypatch.delenv("ODYSSEUS_OLLAMA_MAX_NUM_CTX", raising=False)
    with pytest.raises(WorkflowClosed):
        _create(endpoint_id=_endpoint())
    monkeypatch.setenv("ODYSSEUS_OLLAMA_MAX_NUM_CTX", "4096")
    with pytest.raises(WorkflowClosed):
        _create(endpoint_id=_endpoint())
    assert calls == []


def test_guard_invalid_and_deleted_pin_fail_before_post(monkeypatch):
    calls = _install_replies(monkeypatch, ["NO"])
    project = _create()
    monkeypatch.setenv("ODYSSEUS_OLLAMA_MAX_NUM_CTX", "4096")
    with pytest.raises(WorkflowClosed) as invalid:
        run_next(OWNER, project["id"])
    assert invalid.value.status_code == 400
    assert calls == []
    kept = get_project(OWNER, project["id"])
    assert kept["brief"].startswith("Explain")
    assert _step(kept, "writer")["output_text"] == ""


def test_crash_recovery_does_not_replay(monkeypatch):
    project = _create()
    db = SessionLocal()
    try:
        step = db.query(RoleStep).filter(RoleStep.project_id == project["id"], RoleStep.role == "writer").one()
        step.status = "running"
        step.input_text = "Brief:\nExplain in a few sentences how rain forms."
        db.commit()
    finally:
        db.close()

    calls = _install_replies(monkeypatch, ["SHOULD-NOT-RUN-ON-RECOVER"])
    reopened = get_project(OWNER, project["id"])
    assert _step(reopened, "writer")["status"] == "running"
    with pytest.raises(WorkflowClosed) as blocked:
        run_next(OWNER, project["id"])
    assert blocked.value.status_code == 409
    assert calls == []

    recovered = recover_project(OWNER, project["id"])
    assert _step(recovered, "writer")["status"] == "interrupted"
    assert _step(recovered, "writer")["output_text"] == ""
    assert calls == []

    _install_replies(monkeypatch, ["AFTER-RECOVER"])
    resumed = run_next(OWNER, project["id"])
    assert _step(resumed, "writer")["status"] == "completed"
    assert _step(resumed, "writer")["output_text"] == "AFTER-RECOVER"
    assert _step(resumed, "critic")["status"] == "pending"


def test_reopen_in_new_session_and_fresh_process(tmp_path):
    project = _create(title="Persist me", brief="Explain rain.")
    db = SessionLocal()
    try:
        row = db.query(RoleProject).filter(RoleProject.id == project["id"]).one()
        assert row.owner == OWNER
        assert row.brief == "Explain rain."
    finally:
        db.close()
    again = get_project(OWNER, project["id"])
    assert again["title"] == "Persist me"
    assert [step["role"] for step in again["steps"]] == ["writer", "critic", "organiser"]

    script = textwrap.dedent(
        """
        import os
        import socket
        import sys
        import types
        import uuid

        os.environ["DATABASE_URL"] = sys.argv[1]
        os.environ["ODYSSEUS_ROLE_WORKFLOW"] = "1"
        os.environ["ODYSSEUS_OLLAMA_MAX_NUM_CTX"] = "2048"
        os.environ.pop("ODYSSEUS_OLLAMA_CONTEXT_BASES", None)
        os.environ["ODYSSEUS_DATA_DIR"] = os.path.dirname(sys.argv[1].replace("sqlite:///", "", 1))

        def blocked(*_args, **_kwargs):
            raise RuntimeError("network blocked")

        socket.socket.connect = blocked
        socket.create_connection = blocked
        for name in ("pyotp", "bcrypt"):
            sys.modules.setdefault(name, types.ModuleType(name))

        from core.database import ModelEndpoint, SessionLocal
        from src.role_workflow import RoleStep, create_project, get_project, recover_project

        mode, project_id = sys.argv[2], sys.argv[3]
        if mode == "create":
            db = SessionLocal()
            endpoint_id = uuid.uuid4().hex
            db.add(ModelEndpoint(
                id=endpoint_id,
                name="local-ollama",
                base_url="http://127.0.0.1:11434/v1",
                is_enabled=True,
                owner=None,
                model_type="llm",
            ))
            db.commit()
            db.close()
            project = create_project(
                owner="daily",
                title="Fresh process",
                brief="Explain rain.",
                endpoint_id=endpoint_id,
                model="qwen2.5:7b-instruct",
            )
            db = SessionLocal()
            step = db.query(RoleStep).filter(
                RoleStep.project_id == project["id"], RoleStep.role == "writer"
            ).one()
            step.status = "running"
            db.commit()
            db.close()
            print(project["id"])
        elif mode == "read":
            project = get_project("daily", project_id)
            assert project["steps"][0]["status"] == "running"
            assert project["steps"][0]["output_text"] == ""
            print("STILL_RUNNING")
        elif mode == "recover":
            project = recover_project("daily", project_id)
            assert project["steps"][0]["status"] == "interrupted"
            assert project["steps"][0]["output_text"] == ""
            print("RECOVERED")
        else:
            raise SystemExit("unknown mode")
        """
    )
    db_url = "sqlite:///" + (tmp_path / "role-workflow.db").as_posix()
    env = os_env()
    created = subprocess.run(
        [sys.executable, "-c", script, db_url, "create", ""],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    project_id = created.stdout.strip().splitlines()[-1]
    reread = subprocess.run(
        [sys.executable, "-c", script, db_url, "read", project_id],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert reread.returncode == 0, reread.stderr
    assert "STILL_RUNNING" in reread.stdout
    recovered = subprocess.run(
        [sys.executable, "-c", script, db_url, "recover", project_id],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert "RECOVERED" in recovered.stdout


def os_env():
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ODYSSEUS_ROLE_WORKFLOW"] = "1"
    env["ODYSSEUS_OLLAMA_MAX_NUM_CTX"] = "2048"
    env.pop("ODYSSEUS_OLLAMA_CONTEXT_BASES", None)
    return env


def test_malformed_input_and_other_owners_endpoint(client):
    endpoint_id = _endpoint()
    too_big = client.post("/api/role-workflow/projects", json={
        "title": "Rain notes",
        "brief": "x" * 4001,
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert too_big.status_code == 400
    assert "not shortened" in too_big.json()["detail"]
    missing = client.post("/api/role-workflow/projects", json={
        "title": "",
        "brief": "Explain rain.",
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert missing.status_code == 400
    wrong_type = client.post("/api/role-workflow/projects", json={
        "title": "Rain notes",
        "brief": 12,
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert wrong_type.status_code == 400
    private = _endpoint(owner="someone-else")
    hidden = client.post("/api/role-workflow/projects", json={
        "title": "Rain notes",
        "brief": "Explain rain.",
        "endpoint_id": private,
        "model": MODEL,
    })
    assert hidden.status_code == 400
    listed = client.get("/api/role-workflow/endpoints").json()["endpoints"]
    assert all(item["base_url"] == PIN for item in listed)
    assert private not in {item["id"] for item in listed}


def test_ui_is_text_only_and_linked():
    js = (ROOT / "static/js/role-workflow.js").read_text(encoding="utf-8")
    html = (ROOT / "static/role-workflow.html").read_text(encoding="utf-8")
    page = (ROOT / "static/index.html").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    source = (ROOT / "src/role_workflow.py").read_text(encoding="utf-8")
    routes = (ROOT / "routes/role_workflow_routes.py").read_text(encoding="utf-8")
    for token in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert token not in js
        assert token not in html
    assert "textContent" in js
    assert "not a reviewed judgement" in html
    assert 'id="connect-local"' in html
    assert "/api/role-workflow/connection" in js
    assert "require_admin" not in routes
    assert "Run next step" in html
    assert 'href="/static/role-workflow.html"' in page
    assert "setup_role_workflow_routes" in app
    assert "run_next(" not in app
    assert "recover_project(" not in app
    forbidden = (
        "agent_loop",
        "do_pipeline",
        "send_to_session",
        "chromadb",
        "fastembed",
        "auto_memory",
        "schedule",
    )
    for token in forbidden:
        assert token not in source
        assert token not in routes
        assert token not in js


def test_route_run_contract(monkeypatch, client):
    _block_hidden_paths(monkeypatch)
    calls = _install_replies(monkeypatch, ["ROUTE-DRAFT"])
    endpoint_id = _endpoint()
    created = client.post("/api/role-workflow/projects", json={
        "title": "Route",
        "brief": "Explain rain.",
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert created.status_code == 200
    project_id = created.json()["id"]
    ran = client.post(f"/api/role-workflow/projects/{project_id}/run", json={"owner": "root"})
    assert ran.status_code == 200
    assert _step(ran.json(), "writer")["output_text"] == "ROUTE-DRAFT"
    assert calls[0]["json"]["options"]["num_ctx"] == 2048
    assert calls[0]["kwargs"]["trust_env"] is False
    assert "root" not in ran.text


def test_utf8_budget_rejects_before_post(monkeypatch):
    from src.role_workflow import (
        INPUT_TOKEN_BUDGET,
        MESSAGE_OVERHEAD_TOKENS,
        build_messages,
        complete_text,
        prompt_token_bound,
    )

    calls = _install_replies(monkeypatch, ["SHOULD-NOT-SEND"])
    newlines = _create(brief="Explain rain.", title="Newlines")
    db = SessionLocal()
    try:
        row = db.query(RoleProject).filter(RoleProject.id == newlines["id"]).one()
        row.brief = "\n" * 2200
        db.commit()
    finally:
        db.close()
    with pytest.raises(WorkflowClosed) as rejected:
        run_next(OWNER, newlines["id"])
    assert rejected.value.status_code == 400
    kept = get_project(OWNER, newlines["id"])
    assert kept["brief"] == "\n" * 2200
    assert _step(kept, "writer")["output_text"] == ""
    assert prompt_token_bound(build_messages("writer", kept["brief"], {})) > INPUT_TOKEN_BUDGET

    dense = _create(brief="n" * 2000, title="Dense letters")
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, dense["id"])
    assert get_project(OWNER, dense["id"])["brief"] == "n" * 2000

    emoji_brief = "😀" * 700
    emoji = _create(brief=emoji_brief, title="Emoji brief")
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, emoji["id"])
    assert get_project(OWNER, emoji["id"])["brief"] == emoji_brief
    assert len(emoji_brief.encode("utf-8")) == 2800
    assert calls == []

    draft = _create(title="Emoji draft")
    _install_replies(monkeypatch, ["SHORT-DRAFT"])
    saved = run_next(OWNER, draft["id"])
    assert _step(saved, "writer")["output_text"] == "SHORT-DRAFT"
    db = SessionLocal()
    try:
        step = db.query(RoleStep).filter(
            RoleStep.project_id == draft["id"], RoleStep.role == "writer"
        ).one()
        step.output_text = "😀" * 700
        project = db.query(RoleProject).filter(RoleProject.id == draft["id"]).one()
        project.current_artifact = step.output_text
        db.commit()
    finally:
        db.close()
    calls = _install_replies(monkeypatch, ["SHOULD-NOT-SEND"])
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, draft["id"])
    after = get_project(OWNER, draft["id"])
    assert _step(after, "writer")["output_text"] == "😀" * 700
    assert _step(after, "critic")["status"] == "failed"
    assert _step(after, "critic")["output_text"] == ""
    assert calls == []

    control = _create(title="Control bytes")
    _install_replies(monkeypatch, ["PLAIN"])
    run_next(OWNER, control["id"])
    db = SessionLocal()
    try:
        step = db.query(RoleStep).filter(
            RoleStep.project_id == control["id"], RoleStep.role == "writer"
        ).one()
        step.output_text = "\x01" * 1800
        db.commit()
    finally:
        db.close()
    calls = _install_replies(monkeypatch, ["SHOULD-NOT-SEND"])
    with pytest.raises(WorkflowClosed):
        run_next(OWNER, control["id"])
    assert get_project(OWNER, control["id"])["steps"][0]["output_text"] == "\x01" * 1800
    assert calls == []

    overhead = MESSAGE_OVERHEAD_TOKENS * 2
    user = "u" * (INPUT_TOKEN_BUDGET - overhead - 1)
    exact = [{"role": "system", "content": "s"}, {"role": "user", "content": user}]
    assert prompt_token_bound(exact) == INPUT_TOKEN_BUDGET
    over = [{"role": "system", "content": "s"}, {"role": "user", "content": user + "u"}]
    assert prompt_token_bound(over) == INPUT_TOKEN_BUDGET + 1
    sent = _install_replies(monkeypatch, ["BOUNDARY-OK"])
    assert complete_text(PIN, MODEL, exact) == "BOUNDARY-OK"
    assert len(sent) == 1
    assert sent[0]["json"]["options"]["num_ctx"] == 2048
    assert sent[0]["json"]["options"]["num_predict"] == 512
    with pytest.raises(WorkflowClosed):
        complete_text(PIN, MODEL, over)
    assert len(sent) == 1

    handoff = build_messages(
        "organiser",
        "Explain in a few sentences how rain forms.",
        {"writer": "Moist air rises and cools.", "critic": "Keep both sentences."},
    )
    assert "Moist air rises and cools." in handoff[1]["content"]
    assert "Keep both sentences." in handoff[1]["content"]
    assert prompt_token_bound(handoff) <= INPUT_TOKEN_BUDGET
    fat = build_messages(
        "organiser",
        "Explain rain.",
        {"writer": "w" * 1200, "critic": "c" * 1200},
    )
    assert prompt_token_bound(fat) > INPUT_TOKEN_BUDGET


def test_account_model_policy_blocks_before_post(monkeypatch, client):
    from routes.chat_helpers import _allowed_models_from_privileges
    from src.role_workflow import account_models_from_privileges

    samples = [
        {"block_all_models": True, "allowed_models": ["qwen2.5:7b-instruct"], "allowed_models_restricted": False},
        {"block_all_models": False, "allowed_models": [], "allowed_models_restricted": False},
        {"block_all_models": False, "allowed_models": [], "allowed_models_restricted": True},
        {"block_all_models": False, "allowed_models": [MODEL], "allowed_models_restricted": True},
        {"block_all_models": False, "allowed_models": ["other"], "allowed_models_restricted": False},
    ]
    for privs in samples:
        assert account_models_from_privileges(privs) == _allowed_models_from_privileges(privs)

    flags_off = {
        "can_use_agent": False,
        "can_use_browser": False,
        "can_use_bash": False,
        "can_use_documents": False,
        "can_use_research": False,
        "can_generate_images": False,
        "can_manage_memory": False,
        "block_all_models": True,
        "allowed_models": [],
        "allowed_models_restricted": False,
    }

    class Account:
        def __init__(self):
            self.privs = dict(flags_off)
            self.calls = 0

        def get_privileges(self, username):
            self.calls += 1
            assert username == OWNER
            return dict(self.privs)

    account = Account()
    client.app.state.auth_manager = account
    calls = _install_replies(monkeypatch, ["SHOULD-NOT-SEND"])
    endpoint_id = _endpoint()
    blocked = client.post("/api/role-workflow/projects", json={
        "title": "Blocked",
        "brief": "Explain rain.",
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert blocked.status_code == 403
    assert account.calls >= 1
    assert calls == []

    account.privs["block_all_models"] = False
    account.privs["allowed_models"] = [MODEL]
    account.privs["allowed_models_restricted"] = True
    calls = _install_replies(monkeypatch, ["DRAFT"])
    created = client.post("/api/role-workflow/projects", json={
        "title": "Allowed",
        "brief": "Explain rain.",
        "endpoint_id": endpoint_id,
        "model": MODEL,
    })
    assert created.status_code == 200
    project_id = created.json()["id"]
    ran = client.post(f"/api/role-workflow/projects/{project_id}/run")
    assert ran.status_code == 200
    assert _step(ran.json(), "writer")["output_text"] == "DRAFT"
    assert account.calls >= 3

    account.privs["allowed_models"] = []
    account.privs["block_all_models"] = True
    withdrawn = client.post(f"/api/role-workflow/projects/{project_id}/run")
    assert withdrawn.status_code == 403
    kept = client.get(f"/api/role-workflow/projects/{project_id}").json()
    assert _step(kept, "writer")["output_text"] == "DRAFT"
    assert _step(kept, "critic")["output_text"] == ""
    assert len(calls) == 1
    admin_owned = _endpoint(owner="admin")
    hidden = client.post("/api/role-workflow/projects", json={
        "title": "Hidden endpoint",
        "brief": "Explain rain.",
        "endpoint_id": admin_owned,
        "model": MODEL,
    })
    assert hidden.status_code == 400
    assert admin_owned not in {item["id"] for item in client.get("/api/role-workflow/endpoints").json()["endpoints"]}


def test_malformed_host_is_clean_400_and_localhost_is_numeric(client):
    from src.role_workflow import normalize_loopback_ollama_url

    for raw in (
        "http://localhost:11434.evil.com/v1",
        "http://127.0.0.1:99999/v1",
        "http://127.0.0.1:11434abc/v1",
    ):
        with pytest.raises(WorkflowClosed) as rejected:
            normalize_loopback_ollama_url(raw)
        assert rejected.value.status_code == 400
        endpoint_id = _endpoint(base_url=raw)
        response = client.post("/api/role-workflow/projects", json={
            "title": "Bad URL",
            "brief": "Explain rain.",
            "endpoint_id": endpoint_id,
            "model": MODEL,
        })
        assert response.status_code == 400
    assert normalize_loopback_ollama_url("http://localhost:11434/v1") == PIN
    created = client.post("/api/role-workflow/projects", json={
        "title": "Localhost pin",
        "brief": "Explain rain.",
        "endpoint_id": _endpoint(base_url="http://localhost:11434/v1"),
        "model": MODEL,
    })
    assert created.status_code == 200
    assert created.json()["endpoint_url"] == PIN


def test_proxy_env_connects_to_numeric_loopback(monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://203.0.113.9:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    seen = []

    def create_connection(address, *args, **kwargs):
        seen.append(address)
        raise OSError("intercepted before connect")

    def sock_connect(self, address):
        seen.append(address)
        raise OSError("intercepted before connect")

    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket.socket, "connect", sock_connect)
    project = _create(endpoint_id=_endpoint(base_url="http://localhost:11434/v1"))
    with pytest.raises(WorkflowClosed) as rejected:
        run_next(OWNER, project["id"])
    assert rejected.value.status_code == 502
    assert seen
    for address in seen:
        assert isinstance(address, tuple)
        assert address[0] in {"127.0.0.1", "::1"}
        assert address[1] == 11434
        assert "203.0.113.9" not in str(address)
    kept = get_project(OWNER, project["id"])
    assert kept["endpoint_url"] == PIN
    assert _step(kept, "writer")["status"] == "failed"
    assert _step(kept, "writer")["output_text"] == ""
    assert kept["brief"].startswith("Explain")


def test_stale_attempt_cannot_replace_saved_output():
    from src.role_workflow import _persist_outcome

    project = _create(title="Stale attempt")
    db = SessionLocal()
    try:
        step = db.query(RoleStep).filter(
            RoleStep.project_id == project["id"], RoleStep.role == "writer"
        ).one()
        step.status = "running"
        step.attempt_id = "attempt-a"
        db.commit()
        step_id = step.id
    finally:
        db.close()
    assert _persist_outcome(step_id, "attempt-a", "completed", output="KEEP") is True
    assert _persist_outcome(step_id, "attempt-b", "completed", output="REPLACE") is False
    assert _persist_outcome(step_id, "attempt-a", "failed", error="late failure") is False
    kept = get_project(OWNER, project["id"])
    assert _step(kept, "writer")["output_text"] == "KEEP"
    assert _step(kept, "writer")["status"] == "completed"
    assert _step(kept, "critic")["status"] == "pending"


def test_attempt_column_migration_keeps_rows():
    from sqlalchemy import inspect, text

    role_workflow._TABLES_READY = False
    try:
        with role_workflow.engine.begin() as conn:
            conn.execute(text("ALTER TABLE role_workflow_steps DROP COLUMN attempt_id"))
        role_workflow.ensure_tables()
        columns = {
            col["name"]
            for col in inspect(role_workflow.engine).get_columns("role_workflow_steps")
        }
        assert "attempt_id" in columns
        project = _create(title="After migration")
        assert project["title"] == "After migration"
    finally:
        role_workflow._TABLES_READY = False
        role_workflow.ensure_tables()


def _child_env(data_dir: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["ODYSSEUS_ROLE_WORKFLOW"] = "1"
    env["ODYSSEUS_OLLAMA_MAX_NUM_CTX"] = "2048"
    env.pop("ODYSSEUS_OLLAMA_CONTEXT_BASES", None)
    env["ODYSSEUS_DATA_DIR"] = str(data_dir)
    return env


_PROCESS_SCRIPT = textwrap.dedent(
    """
    import os
    import socket
    import sys
    import time
    import types
    import uuid

    os.environ["DATABASE_URL"] = sys.argv[1]
    os.environ["ODYSSEUS_DATA_DIR"] = sys.argv[2]
    os.environ["ODYSSEUS_ROLE_WORKFLOW"] = "1"
    os.environ["ODYSSEUS_OLLAMA_MAX_NUM_CTX"] = "2048"
    os.environ.pop("ODYSSEUS_OLLAMA_CONTEXT_BASES", None)

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network blocked")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    for name in ("pyotp", "bcrypt"):
        sys.modules.setdefault(name, types.ModuleType(name))

    from core.database import ModelEndpoint, SessionLocal
    from src import role_workflow
    from src.role_workflow import WorkflowClosed, create_project, recover_project, run_next

    db_url, data_dir, mode, project_id, other_id, log_path, release_path = sys.argv[1:8]

    def slow(url, model, messages):
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("post\\n")
            handle.flush()
        if mode == "die":
            os._exit(9)
        if mode == "once":
            return "REPLY-ONCE"
        deadline = time.time() + 8
        while time.time() < deadline and not os.path.exists(release_path):
            time.sleep(0.05)
        return "REPLY-HOLD"

    if mode != "create":
        role_workflow.complete_text = slow

    if mode == "create":
        db = SessionLocal()
        first = uuid.uuid4().hex
        second = uuid.uuid4().hex
        for endpoint_id in (first, second):
            db.add(ModelEndpoint(
                id=endpoint_id,
                name="local-ollama",
                base_url="http://127.0.0.1:11434/v1",
                is_enabled=True,
                owner=None,
                model_type="llm",
            ))
        db.commit()
        db.close()
        one = create_project(
            owner="daily", title="One", brief="Explain rain.",
            endpoint_id=first, model="qwen2.5:7b-instruct",
        )
        two = create_project(
            owner="daily", title="Two", brief="Explain rain.",
            endpoint_id=second, model="qwen2.5:7b-instruct",
        )
        print(one["id"])
        print(two["id"])
    elif mode == "read":
        project = role_workflow.get_project("daily", project_id)
        writer = project["steps"][0]
        print(writer["status"])
        print(writer["output_text"])
        print(project["steps"][1]["status"])
    elif mode == "recover":
        try:
            recover_project("daily", project_id)
            print("RECOVERED")
        except WorkflowClosed as exc:
            print(f"ERR {exc.status_code}")
    elif mode == "cancel":
        print("CANCEL" if role_workflow.cancel_project("daily", project_id)["cancel_requested"] else "NOCANCEL")
        project = role_workflow.get_project("daily", project_id)
        print(project["steps"][0]["status"])
    else:
        target = other_id if mode == "other" else project_id
        try:
            run_next("daily", target)
            print("OK")
        except WorkflowClosed as exc:
            print(f"ERR {exc.status_code}")
    """
)


def _spawn(db_url, data_dir, mode, project_id="", other_id="", log_path="", release_path=""):
    return subprocess.run(
        [
            sys.executable, "-c", _PROCESS_SCRIPT,
            db_url, str(data_dir), mode, project_id, other_id, log_path, release_path,
        ],
        cwd=str(ROOT),
        env=_child_env(data_dir),
        text=True,
        capture_output=True,
        timeout=40,
        check=False,
    )


def test_separate_processes_share_one_model_call(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_url = "sqlite:///" + (tmp_path / "shared.db").as_posix()
    log_path = tmp_path / "posts.txt"
    release_path = tmp_path / "release"
    created = _spawn(db_url, data_dir, "create", log_path=str(log_path), release_path=str(release_path))
    assert created.returncode == 0, created.stderr
    first_id, second_id = created.stdout.strip().splitlines()[-2:]
    holder = subprocess.Popen(
        [
            sys.executable, "-c", _PROCESS_SCRIPT,
            db_url, str(data_dir), "hold", first_id, second_id, str(log_path), str(release_path),
        ],
        cwd=str(ROOT),
        env=_child_env(data_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.time() + 15
        while time.time() < deadline and log_path.read_text(encoding="utf-8").count("post") < 1 if log_path.exists() else True:
            if holder.poll() is not None:
                stdout, stderr = holder.communicate()
                raise AssertionError(f"holder exited early\n{stdout}\n{stderr}")
            time.sleep(0.05)
        assert log_path.read_text(encoding="utf-8").count("post") == 1
        duplicate = _spawn(
            db_url, data_dir, "hold", first_id, second_id, str(log_path), str(release_path)
        )
        other = _spawn(
            db_url, data_dir, "other", first_id, second_id, str(log_path), str(release_path)
        )
        recovered = _spawn(
            db_url, data_dir, "recover", first_id, second_id, str(log_path), str(release_path)
        )
        cancelled = _spawn(
            db_url, data_dir, "cancel", first_id, second_id, str(log_path), str(release_path)
        )
        assert duplicate.returncode == 0, duplicate.stderr
        assert other.returncode == 0, other.stderr
        assert recovered.returncode == 0, recovered.stderr
        assert cancelled.returncode == 0, cancelled.stderr
        assert "ERR 409" in duplicate.stdout
        assert "ERR 409" in other.stdout
        assert "ERR 409" in recovered.stdout
        assert "NOCANCEL" in cancelled.stdout
        assert "running" in cancelled.stdout
        assert log_path.read_text(encoding="utf-8").count("post") == 1
    finally:
        release_path.write_text("go", encoding="utf-8")
        stdout, stderr = holder.communicate(timeout=15)
    assert holder.returncode == 0, stderr
    assert "OK" in stdout
    assert log_path.read_text(encoding="utf-8").count("post") == 1
    read = _spawn(db_url, data_dir, "read", first_id, second_id, str(log_path), str(release_path))
    assert read.returncode == 0, read.stderr
    status, output, critic = read.stdout.strip().splitlines()[-3:]
    assert status == "completed"
    assert output == "REPLY-HOLD"
    assert critic == "pending"


def test_process_death_releases_lock_without_a_second_call(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_url = "sqlite:///" + (tmp_path / "shared.db").as_posix()
    log_path = tmp_path / "posts.txt"
    release_path = tmp_path / "release"
    created = _spawn(db_url, data_dir, "create", log_path=str(log_path), release_path=str(release_path))
    assert created.returncode == 0, created.stderr
    first_id = created.stdout.strip().splitlines()[-2]
    dead = subprocess.Popen(
        [
            sys.executable, "-c", _PROCESS_SCRIPT,
            db_url, str(data_dir), "die", first_id, "", str(log_path), str(release_path),
        ],
        cwd=str(ROOT),
        env=_child_env(data_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 15
    while time.time() < deadline and not (log_path.exists() and "post" in log_path.read_text(encoding="utf-8")):
        time.sleep(0.05)
    dead.wait(timeout=10)
    assert dead.returncode == 9
    recovered = _spawn(db_url, data_dir, "recover", first_id, "", str(log_path), str(release_path))
    assert recovered.returncode == 0, recovered.stderr
    assert "RECOVERED" in recovered.stdout
    assert log_path.read_text(encoding="utf-8").count("post") == 1
    read = _spawn(db_url, data_dir, "read", first_id, "", str(log_path), str(release_path))
    assert read.returncode == 0, read.stderr
    status, output, _critic = read.stdout.strip().splitlines()[-3:]
    assert status == "interrupted"
    assert output == ""
    resumed = _spawn(db_url, data_dir, "once", first_id, "", str(log_path), str(release_path))
    assert resumed.returncode == 0, resumed.stderr
    assert "OK" in resumed.stdout
    assert log_path.read_text(encoding="utf-8").count("post") == 2
    final = _spawn(db_url, data_dir, "read", first_id, "", str(log_path), str(release_path))
    assert final.returncode == 0, final.stderr
    status, output, critic = final.stdout.strip().splitlines()[-3:]
    assert status == "completed"
    assert output == "REPLY-ONCE"
    assert critic == "pending"


def test_private_connection_is_owner_scoped_and_does_not_probe(monkeypatch, client):
    from core.database import ModelEndpoint
    from src.role_workflow import PRIVATE_LOOPBACK_URL

    model_routes = (ROOT / "routes/model_routes.py").read_text(encoding="utf-8")
    create_at = model_routes.find("def create_model_endpoint")
    assert create_at > 0
    assert "require_admin(request)" in model_routes[create_at:create_at + 2500]

    calls = _install_replies(monkeypatch, ["PRIVATE-DRAFT"])
    other = _endpoint(owner="someone-else")
    shared = _endpoint(owner=None)
    spoof = client.post("/api/role-workflow/connection", json={
        "owner": "root",
        "id": other,
        "base_url": "https://example.com/v1",
        "api_key": "secret",
        "shared": True,
        "model": "other:model",
    })
    assert spoof.status_code == 200
    body = spoof.json()
    assert body["base_url"] == PRIVATE_LOOPBACK_URL
    assert body["model"] == MODEL
    assert body["id"] != other
    assert calls == []

    again = client.post("/api/role-workflow/connection", json={"id": body["id"], "owner": "someone-else"})
    assert again.status_code == 200
    assert again.json()["id"] == body["id"]
    db = SessionLocal()
    try:
        rows = db.query(ModelEndpoint).filter(ModelEndpoint.owner == OWNER).all()
        assert len(rows) == 1
        assert rows[0].id == body["id"]
        assert rows[0].api_key in (None, "")
        assert rows[0].is_enabled is True
        assert rows[0].base_url == PRIVATE_LOOPBACK_URL
        untouched = db.query(ModelEndpoint).filter(ModelEndpoint.id == other).one()
        assert untouched.owner == "someone-else"
        assert untouched.base_url == PIN
    finally:
        db.close()

    listed = client.get("/api/role-workflow/endpoints").json()["endpoints"]
    assert {item["id"] for item in listed} >= {body["id"], shared}
    assert other not in {item["id"] for item in listed}

    client.headers["x-test-user"] = "other-daily"
    theirs = client.post("/api/role-workflow/connection", json={"id": body["id"], "owner": OWNER})
    assert theirs.status_code == 200
    assert theirs.json()["id"] != body["id"]
    their_list = {item["id"] for item in client.get("/api/role-workflow/endpoints").json()["endpoints"]}
    assert body["id"] not in their_list
    assert theirs.json()["id"] in their_list
    client.headers["x-test-user"] = OWNER

    db = SessionLocal()
    try:
        row = db.query(ModelEndpoint).filter(ModelEndpoint.id == body["id"]).one()
        row.is_enabled = False
        db.commit()
    finally:
        db.close()
    blocked = client.post("/api/role-workflow/connection")
    assert blocked.status_code == 403
    db = SessionLocal()
    try:
        row = db.query(ModelEndpoint).filter(ModelEndpoint.id == body["id"]).one()
        assert row.is_enabled is False
        assert db.query(ModelEndpoint).filter(ModelEndpoint.owner == OWNER, ModelEndpoint.is_enabled == True).count() == 0
    finally:
        db.close()

    monkeypatch.delenv("ODYSSEUS_ROLE_WORKFLOW", raising=False)
    off = client.post("/api/role-workflow/connection", json={"owner": "root"})
    assert off.status_code == 404
    monkeypatch.setenv("ODYSSEUS_ROLE_WORKFLOW", "1")

    with TestClient(client.app) as anon:
        assert anon.post("/api/role-workflow/connection").status_code == 401
    with TestClient(client.app) as token:
        token.headers["x-test-api-token"] = "1"
        assert token.post("/api/role-workflow/connection").status_code == 403

    class Account:
        def __init__(self):
            self.calls = 0
            self.privs = {
                "block_all_models": True,
                "allowed_models": [],
                "allowed_models_restricted": False,
                "can_use_agent": False,
                "can_use_bash": False,
            }

        def get_privileges(self, username):
            self.calls += 1
            return dict(self.privs)

    account = Account()
    client.app.state.auth_manager = account
    db = SessionLocal()
    try:
        db.query(ModelEndpoint).filter(ModelEndpoint.id == body["id"]).update({"is_enabled": True})
        db.commit()
    finally:
        db.close()
    denied = client.post("/api/role-workflow/connection")
    assert denied.status_code == 403
    assert account.calls >= 1
    account.privs["block_all_models"] = False
    account.privs["allowed_models_restricted"] = True
    account.privs["allowed_models"] = [MODEL]
    allowed = client.post("/api/role-workflow/connection")
    assert allowed.status_code == 200
    created = client.post("/api/role-workflow/projects", json={
        "title": "Private",
        "brief": "Explain rain.",
        "endpoint_id": allowed.json()["id"],
        "model": MODEL,
        "owner": "root",
    })
    assert created.status_code == 200
    ran = client.post(f"/api/role-workflow/projects/{created.json()['id']}/run")
    assert ran.status_code == 200
    assert _step(ran.json(), "writer")["output_text"] == "PRIVATE-DRAFT"
    assert calls[0]["json"]["options"]["num_ctx"] == 2048
    assert calls[0]["json"]["options"]["num_predict"] == 512
    assert len(calls) == 1


def test_late_completion_is_readable_without_another_run(monkeypatch):
    project = _create(title="Late save")
    started = threading.Event()
    release = threading.Event()
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs.get("json"))
        started.set()
        assert release.wait(5)
        return _reply(url, "SAVED-LATE")

    monkeypatch.setattr(role_workflow.httpx, "post", post)
    holder = {}

    def worker():
        try:
            holder["project"] = run_next(OWNER, project["id"])
        except WorkflowClosed as exc:
            holder["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert started.wait(5)
    midway = get_project(OWNER, project["id"])
    assert _step(midway, "writer")["status"] == "running"
    assert _step(midway, "writer")["output_text"] == ""
    assert len(calls) == 1
    release.set()
    thread.join(5)
    assert "error" not in holder
    saved = get_project(OWNER, project["id"])
    assert _step(saved, "writer")["output_text"] == "SAVED-LATE"
    assert _step(saved, "writer")["status"] == "completed"
    reread = get_project(OWNER, project["id"])
    assert _step(reread, "writer")["output_text"] == "SAVED-LATE"
    assert len(calls) == 1
    assert calls[0]["options"]["num_ctx"] == 2048


def test_role_run_timeout_is_narrow(monkeypatch):
    import asyncio

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    import src.request_timeout as request_timeout

    project_id = "ab" * 16
    run_path = f"/api/role-workflow/projects/{project_id}/run"
    assert request_timeout.is_role_workflow_run(run_path)
    assert not request_timeout.is_role_workflow_run(f"/api/role-workflow/projects/{project_id}/recover")
    assert not request_timeout.is_role_workflow_run("/api/role-workflow/connection")
    assert not request_timeout.is_role_workflow_run("/api/role-workflow/projects")
    assert request_timeout.request_hard_timeout_exempt("/api/chat")
    assert request_timeout.request_hard_timeout_exempt("/api/memory/audit")

    monkeypatch.setattr(request_timeout, "REQUEST_HARD_TIMEOUT", 0.05)
    app = FastAPI()

    @app.middleware("http")
    async def timeout_gate(request, call_next):
        return await request_timeout.call_with_hard_timeout(request, call_next)

    @app.post(run_path)
    async def slow_run():
        await asyncio.sleep(0.2)
        return {"saved": True}

    @app.post("/api/role-workflow/projects/{project_id}/recover")
    async def slow_recover(project_id: str):
        await asyncio.sleep(0.2)
        return {"saved": True}

    @app.get("/api/slow-ordinary")
    async def slow_ordinary():
        await asyncio.sleep(0.2)
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post(run_path).status_code == 200
        recover = client.post(f"/api/role-workflow/projects/{project_id}/recover")
        assert recover.status_code == 504
        ordinary = client.get("/api/slow-ordinary")
        assert ordinary.status_code == 504
        assert "45" not in ordinary.text
        assert isinstance(ordinary.json()["detail"], str)


def test_role_dependency_closures_are_pinned():
    safety = (ROOT / "requirements-role-safety.txt").read_text(encoding="utf-8")
    runtime = (ROOT / "requirements-role-runtime.txt").read_text(encoding="utf-8")
    bootstrap = (ROOT / "requirements-bootstrap.txt").read_text(encoding="utf-8")
    safety_reqs = "\n".join(
        line for line in safety.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    assert "fastapi==0.162.0" in safety_reqs
    assert "opentelemetry-api==1.45.2" in safety_reqs
    assert "opentelemetry-sdk==" not in safety_reqs
    assert "opentelemetry-exporter" not in safety_reqs
    for banned in ("fastembed", "faster-whisper", "chromadb", "yt-dlp", "uvicorn"):
        assert banned not in safety_reqs
    assert "e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1" in safety
    assert "e9b54de76e530bb6cf0ccd036787103acb58fd42ba930ea8bf948ddbaa2f61d1" in bootstrap
    runtime_reqs = "\n".join(
        line for line in runtime.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )
    assert "-r requirements-role-safety.txt" in runtime_reqs
    assert "uvicorn==0.54.0" in runtime_reqs
    assert "python-dotenv==1.2.4" in runtime_reqs
    assert "numpy==2.4.6" in runtime_reqs
    assert "numpy==2.5.4" not in runtime_reqs
    assert "fastembed" not in runtime_reqs

