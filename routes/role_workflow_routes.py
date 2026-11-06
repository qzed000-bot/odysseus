"""HTTP API for the opt-in sequential text workflow.

Owner identity comes from the authenticated request. Client-supplied owner
fields are not read. The routes do not consult agent, admin, or tool
privileges, and they do not start background work.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.auth_helpers import require_user
from src.role_workflow import (
    WorkflowClosed,
    account_models_from_privileges,
    cancel_project,
    create_project,
    feature_enabled,
    get_project,
    list_pinned_endpoints,
    list_projects,
    recover_project,
    run_next,
    status_payload,
)


class ProjectCreate(BaseModel):
    title: Any = None
    brief: Any = None
    endpoint_id: Any = None
    model: Any = None


def _owner(request: Request) -> str:
    user = require_user(request)
    if not user or user in {"api", "internal-tool"}:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _account_models(request: Request, owner: str):
    """Current account model gate. Missing auth state means no account list.

    Capability flags are not consulted. A present account that cannot be
    read fails closed.
    """
    auth_manager = getattr(getattr(request, "app", None).state, "auth_manager", None)
    if auth_manager is None or not hasattr(auth_manager, "get_privileges"):
        return None
    try:
        privs = auth_manager.get_privileges(owner)
    except Exception as exc:
        raise WorkflowClosed("model access could not be verified", 403) from exc
    return account_models_from_privileges(privs)


def _feature_or_404() -> None:
    if not feature_enabled():
        raise HTTPException(status_code=404, detail="Not found")


def _closed(exc: WorkflowClosed) -> HTTPException:
    status = exc.status_code
    if not feature_enabled() and status == 404:
        return HTTPException(status_code=404, detail="Not found")
    return HTTPException(status_code=status, detail=exc.message)


def setup_role_workflow_routes() -> APIRouter:
    router = APIRouter(prefix="/api/role-workflow", tags=["role-workflow"])

    @router.get("/status")
    def workflow_status(request: Request):
        if not feature_enabled():
            return {"enabled": False}
        _owner(request)
        try:
            return status_payload()
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.get("/endpoints")
    def workflow_endpoints(request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return {"endpoints": list_pinned_endpoints(owner)}
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.get("/projects")
    def workflow_projects(request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return {"projects": list_projects(owner)}
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.post("/projects")
    def workflow_create(body: ProjectCreate, request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return create_project(
                owner=owner,
                title=body.title,
                brief=body.brief,
                endpoint_id=body.endpoint_id,
                model=body.model,
                account_models=_account_models(request, owner),
            )
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.get("/projects/{project_id}")
    def workflow_get(project_id: str, request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return get_project(owner, project_id)
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.post("/projects/{project_id}/run")
    def workflow_run(project_id: str, request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return run_next(owner, project_id, _account_models(request, owner))
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.post("/projects/{project_id}/recover")
    def workflow_recover(project_id: str, request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return recover_project(owner, project_id)
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    @router.post("/projects/{project_id}/cancel")
    def workflow_cancel(project_id: str, request: Request):
        _feature_or_404()
        owner = _owner(request)
        try:
            return cancel_project(owner, project_id)
        except WorkflowClosed as exc:
            raise _closed(exc) from exc

    return router
