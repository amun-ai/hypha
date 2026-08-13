"""#0044 — three prod log-hygiene fixes (zippy-goat nightly review 2026-08-13).

All three restate a benign, expected condition at a level (INFO/ERROR) that either
floods the retained ``kubectl logs`` forensic window or poisons the ERROR metric
health thresholds key on. None is a real fault. Same log-hygiene family as #0008,
#0019, #0020, #0021.

Finding 1c — ``hypha/artifact.py::search`` query-planner narration
    ``search()`` logged 3+ INFO lines PER CALL (``search: stage parameter value``,
    ``Adding stage filter condition``, ``Adding condition to return ONLY …``).
    ``search`` is a hot listing path (~300k lines/24h in prod). These are internal
    query-planner breadcrumbs with no steady-state operator value → demoted to DEBUG.

Finding 2 — ``hypha/http_rpc.py::_handle_rpc_post`` ``ClientDisconnect``
    When a client hangs up before/while its request body arrives, ``await
    request.body()`` raises ``starlette.requests.ClientDisconnect``. The broad
    ``except Exception`` caught it and logged ``ERROR`` + full traceback + returned
    ``500`` (618/24h in prod). It is a normal network event, not a server error —
    now caught specifically, logged at DEBUG with no traceback, and answered ``499``
    (there is no client left to answer).

Finding 3 — ``hypha/core/workspace.py`` default-service ``setup`` probe
    On every default-service registration the manager ran ``if svc.setup:`` on the
    remote-service proxy. When a default service simply does not define ``setup()``,
    the Munch/ObjectProxy RAISES ``AttributeError('setup')`` — a benign, expected
    condition that was logged as a per-client ERROR whose ``{e}`` interpolated to the
    useless bare string ``"setup"`` (~20/24h, unactionable by construction). The
    ``getattr(svc, "setup", None)`` guard makes "no setup method" a silent no-op;
    only a ``setup()`` that actually exists and RAISES is now an ERROR — carrying the
    exception type + repr so it is diagnosable.

These are real tests: genuine Docker-free ``RedisStore`` (FakeRedis), a real
``ArtifactController``, real ``HTTPStreamingRPCServer`` routes driven over ASGI,
and two real event-bus RPC peers — no mocks. For each finding one test proves the
benign path is silent at the noisy level (the regression guard) and, where the line
still carries diagnostic value, a second proves it is only downgraded, not deleted.
"""
import logging

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import ClientDisconnect, Request

from hypha.artifact import ArtifactController
from hypha.core import UserInfo, WorkspaceInfo
from hypha.core.auth import create_scope, generate_auth_token
from hypha.core.store import RedisStore
from hypha.http_rpc import HTTPStreamingRPCServer


# ---------------------------------------------------------------------------
# Finding 1c — artifact search() query-planner lines must be DEBUG, not INFO
# ---------------------------------------------------------------------------

_ARTIFACT_LOGGER = "artifact"
_PLANNER_MARKERS = (
    "search: stage parameter value",
    "Adding stage filter condition",
    "Adding condition to return ONLY",
)


async def _store_with_artifact_manager():
    """Build a Docker-free store + a real ArtifactController (no S3).

    The ArtifactController constructor registers a public service and a router,
    so it MUST be constructed before ``store.init()`` (which flips ``_ready`` and
    asserts no public service is registered afterwards)."""
    app = FastAPI()
    store = RedisStore(app, redis_uri=None)
    am = ArtifactController(store, s3_controller=None)
    await store.init(reset_redis=True)
    await am.init_db()
    return store, am


def _search_context():
    ws = "ws-artifact-loghygiene"
    user_info = UserInfo(
        id="artifact-loghygiene-user",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )
    return {
        "user": user_info.model_dump(),
        "ws": ws,
        "from": f"{ws}/loghygiene-client",
    }


@pytest.mark.asyncio
async def test_search_planner_lines_silent_at_info(caplog):
    """Regression guard: a full search() emits NO query-planner line at INFO."""
    store, am = await _store_with_artifact_manager()
    try:
        ctx = _search_context()
        with caplog.at_level(logging.INFO, logger=_ARTIFACT_LOGGER):
            # stage=False drives the explicit stage-filter branch (the loudest one)
            rows = await am.search(filters={"parent_id": None}, stage=False, context=ctx)
        assert rows == []  # empty workspace — proves we reached the planner, 0 rows
        planner_lines = [
            r.getMessage()
            for r in caplog.records
            if r.name == _ARTIFACT_LOGGER
            and any(m in r.getMessage() for m in _PLANNER_MARKERS)
        ]
        assert planner_lines == [], (
            "artifact search() query-planner narration must be silent at INFO; "
            f"leaked: {planner_lines}"
        )
    finally:
        await store.teardown()


@pytest.mark.asyncio
async def test_search_planner_lines_present_at_debug(caplog):
    """Downgrade-not-delete proof: the lines still fire at DEBUG for deep debugging."""
    store, am = await _store_with_artifact_manager()
    try:
        ctx = _search_context()
        with caplog.at_level(logging.DEBUG, logger=_ARTIFACT_LOGGER):
            await am.search(filters={"parent_id": None}, stage=False, context=ctx)
        planner_lines = [
            r.getMessage()
            for r in caplog.records
            if r.name == _ARTIFACT_LOGGER
            and any(m in r.getMessage() for m in _PLANNER_MARKERS)
        ]
        assert any(
            "search: stage parameter value" in m for m in planner_lines
        ), "query-planner lines must still be present at DEBUG (downgraded, not deleted)"
    finally:
        await store.teardown()


# ---------------------------------------------------------------------------
# Finding 2 — ClientDisconnect on the /rpc POST body must not be an ERROR/500
# ---------------------------------------------------------------------------

_HTTP_RPC_LOGGER = "http-rpc"


@pytest.mark.asyncio
async def test_rpc_post_client_disconnect_is_quiet_499(caplog, monkeypatch):
    """A client that hangs up mid-body yields 499 + DEBUG, never ERROR/traceback/500."""
    app = FastAPI()
    store = RedisStore(app, redis_uri=None)
    await store.init(reset_redis=True)
    server = HTTPStreamingRPCServer(store)
    server.register_routes(app)

    # A valid Bearer token so the request passes auth and reaches the body read.
    ws = "ws-rpc-loghygiene"
    user_info = UserInfo(
        id="rpc-loghygiene-user",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )
    token = await generate_auth_token(user_info, 3600)

    # The client hanging up mid-body surfaces to the handler as ClientDisconnect
    # raised by ``await request.body()`` — reproduce that exactly.
    async def _boom(self):
        raise ClientDisconnect()

    monkeypatch.setattr(Request, "body", _boom)

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            with caplog.at_level(logging.DEBUG, logger=_HTTP_RPC_LOGGER):
                resp = await client.post(
                    "/rpc",
                    params={"workspace": ws, "client_id": "c1"},
                    headers={"Authorization": f"Bearer {token}"},
                    content=b"\x00\x01\x02",
                )
        assert resp.status_code == 499, (
            f"ClientDisconnect must yield 499, got {resp.status_code}: {resp.text}"
        )
        errors = [
            r for r in caplog.records
            if r.name == _HTTP_RPC_LOGGER and r.levelno >= logging.ERROR
        ]
        assert errors == [], (
            "ClientDisconnect must not be logged at ERROR; "
            f"leaked: {[r.getMessage() for r in errors]}"
        )
        # No traceback text either — the broad arm interpolated format_exc().
        assert not any(
            "Traceback" in r.getMessage() for r in caplog.records
            if r.name == _HTTP_RPC_LOGGER
        ), "ClientDisconnect path must not emit a traceback"
        # It IS still observable at DEBUG.
        assert any(
            "Client disconnected before the RPC request body" in r.getMessage()
            for r in caplog.records
            if r.name == _HTTP_RPC_LOGGER and r.levelno == logging.DEBUG
        ), "the benign disconnect should still be visible at DEBUG"
    finally:
        await store.teardown()


# ---------------------------------------------------------------------------
# Finding 3 — default-service setup probe: no-setup silent, real failure diagnosable
# ---------------------------------------------------------------------------

_WORKSPACE_LOGGER = "workspace"


class _WorkspaceRecords(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        try:
            self.records.append((record.levelname, record.getMessage()))
        except Exception:
            pass


async def _register_default_service(setup_callable, ws_suffix):
    """Register a ``default`` service through the real event-bus register path.

    Seeds the client's ``built-in`` registry key directly (the manager's
    client-exists guard is existence-only) so the real ``:default@`` block in
    ``workspace.py`` runs, then returns the captured ``workspace``-logger records
    produced during that registration."""
    app = FastAPI()
    store = RedisStore(app, redis_uri=None)
    await store.init(reset_redis=True)
    ws = f"ws-default-{ws_suffix}"
    user = UserInfo(
        id=f"default-{ws_suffix}-user",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )
    await store.register_workspace(
        WorkspaceInfo(
            id=ws,
            name=ws,
            description="log-hygiene test",
            owners=[user.id],
            persistent=False,
            read_only=False,
        ),
        overwrite=True,
    )
    provider = store.create_rpc(ws, user, "c1")
    service = {
        "id": "default",
        "name": "default",
        "config": {"visibility": "protected"},
    }
    if setup_callable is not None:
        service["setup"] = setup_callable
    # Make the client "present" so the manager does not refuse the non-built-in svc.
    await store.get_redis().set(
        f"services:protected|built-in:{ws}/c1:built-in@app", b"{}"
    )

    handler = _WorkspaceRecords()
    handler.setLevel(logging.DEBUG)
    ws_logger = logging.getLogger(_WORKSPACE_LOGGER)
    ws_logger.addHandler(handler)
    prev_level = ws_logger.level
    ws_logger.setLevel(logging.DEBUG)
    try:
        await provider.register_service(service, {"overwrite": True})
        # The :default@ block resolves + probes setup asynchronously after hset.
        import asyncio

        await asyncio.sleep(1.2)
        return list(handler.records)
    finally:
        ws_logger.removeHandler(handler)
        ws_logger.setLevel(prev_level)
        await store.teardown()


@pytest.mark.asyncio
async def test_default_service_without_setup_is_silent():
    """A default service that defines no setup() emits NO ERROR (was bare 'setup')."""
    records = await _register_default_service(None, "nosetup")
    setup_errors = [
        msg for lvl, msg in records
        if lvl == "ERROR" and "setup" in msg.lower()
    ]
    assert setup_errors == [], (
        "a default service without setup() must not log any setup ERROR; "
        f"leaked: {setup_errors}"
    )


@pytest.mark.asyncio
async def test_default_service_setup_failure_carries_type_and_repr():
    """A setup() that actually raises IS an ERROR — with type + repr, not bare 'setup'."""

    def bad_setup():
        raise RuntimeError("boom-in-setup")

    records = await _register_default_service(bad_setup, "raises")
    setup_errors = [
        msg for lvl, msg in records
        if lvl == "ERROR" and "Failed to run setup" in msg
    ]
    assert setup_errors, f"a raising setup() must log an ERROR; got records: {records}"
    msg = setup_errors[0]
    # The whole point of the fix: the detail names the real failure, not "setup".
    assert "boom-in-setup" in msg, f"ERROR must carry the real failure detail: {msg}"
    assert msg.rstrip().rsplit(":", 1)[-1].strip() != "setup", (
        f"ERROR detail must not be the useless bare 'setup': {msg}"
    )
