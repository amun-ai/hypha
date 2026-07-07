"""Log-hygiene regression tests (0.21.107): benign WS lifecycle events must not
surface as ERROR and mask real errors.

Flagged during the 0.21.106 prod rollout (valiant-goat): the loudest benign-at-
ERROR was a *consequence of the 0.21.105 supersede-orphan fix* — cancelling the
old generation's task raises `asyncio.CancelledError` in its parked
`await asyncio.wait_for(websocket.receive(), ...)`, which propagated uncaught out
of the ASGI handler so uvicorn logged "Exception in ASGI application" at ERROR
(~320/min during the rollout's reconnect wave). Plus three plain benign-at-ERROR
log lines (disconnect / send-fail / close-race).
"""
import asyncio
import logging
import os

import pytest

from hypha.core import RedisRPCConnection, UserInfo
from hypha.core.auth import create_scope
from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


class _FakeState:
    name = "CONNECTED"


class _PingingWS:
    """Keeps delivering client 'ping' frames so the receive loop stays parked
    (so a supersede-cancel lands in the wait_for(receive))."""

    def __init__(self):
        self.client_state = _FakeState()
        self.application_state = _FakeState()
        self.closed = False

    async def receive(self):
        import json

        await asyncio.sleep(0.05)
        return {"text": json.dumps({"type": "ping"})}

    async def send_bytes(self, data):
        pass

    async def send_text(self, data):
        pass

    async def close(self, code=None, reason=None):
        self.closed = True


def _user(ws):
    return UserInfo(
        id="loghygiene",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )


async def _make_server(store):
    from fastapi import FastAPI
    from hypha.websocket import WebsocketServer

    store._app = FastAPI()
    return WebsocketServer(store)


async def test_supersede_cancel_does_not_surface_as_error():
    """The supersede-cancel of an old generation must exit CLEANLY — the task
    completes (not cancelled, no exception propagated), so it never bubbles up to
    uvicorn as 'Exception in ASGI application' at ERROR — while cleanup still runs.
    """
    os.environ["HYPHA_WS_IDLE_TIMEOUT"] = "600"
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws_name = "ws-user-loghyg"
        user = _user(ws_name)
        await store.register_workspace(
            dict(name=ws_name, description="x", owners=["loghygiene"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        client_id = "client-reconnecting"
        ws1 = _PingingWS()
        t1 = asyncio.create_task(
            server.establish_websocket_communication(ws1, ws_name, client_id, user)
        )
        await asyncio.sleep(0.35)
        assert not t1.done()

        closed_before = RedisRPCConnection._closed_total_int

        # Supersede: same client_id reconnects → cancels t1.
        ws2 = _PingingWS()
        t2 = asyncio.create_task(
            server.establish_websocket_communication(ws2, ws_name, client_id, user)
        )
        await asyncio.sleep(0.5)

        # gen1 must have exited cleanly — NOT as a cancelled task and NOT raising.
        assert t1.done(), "gen1 must have exited"
        assert not t1.cancelled(), (
            "supersede-cancel must be swallowed (clean exit), not surface as a "
            "cancelled task / uncaught CancelledError → ASGI ERROR"
        )
        assert t1.exception() is None, (
            f"gen1 must exit cleanly, but raised {t1.exception()!r}"
        )
        # Cleanup still ran (the whole point of the supersede-cancel).
        assert RedisRPCConnection._closed_total_int - closed_before >= 1

        t2.cancel()
        await asyncio.gather(t2, return_exceptions=True)
    finally:
        await store.teardown()
        os.environ.pop("HYPHA_WS_IDLE_TIMEOUT", None)


async def test_genuine_cancel_still_propagates():
    """A cancel with NO newer generation (e.g. event-loop shutdown) must still
    propagate — we only swallow the intentional supersede-cancel."""
    os.environ["HYPHA_WS_IDLE_TIMEOUT"] = "600"
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws_name = "ws-user-genuine"
        user = _user(ws_name)
        await store.register_workspace(
            dict(name=ws_name, description="x", owners=["loghygiene"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        ws = _PingingWS()
        t = asyncio.create_task(
            server.establish_websocket_communication(ws, ws_name, "solo", user)
        )
        await asyncio.sleep(0.35)
        # No supersede → generation stays at this task's gen. A cancel here is
        # genuine and must propagate (task ends up cancelled).
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert t.cancelled()
    finally:
        await store.teardown()
        os.environ.pop("HYPHA_WS_IDLE_TIMEOUT", None)


async def test_disconnect_logs_at_info_not_error(caplog):
    """WebsocketServer.disconnect() logs the lifecycle line at INFO, not ERROR."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = _PingingWS()
        with caplog.at_level(logging.DEBUG, logger="websocket-server"):
            await server.disconnect(ws, "Client normal close", 1000)

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert not errors, (
            f"disconnect() must not log at ERROR: "
            f"{[r.getMessage() for r in errors]}"
        )
        assert any(
            "Disconnecting, reason" in r.getMessage() and r.levelno == logging.INFO
            for r in caplog.records
        ), "the disconnect lifecycle line must be logged at INFO"
    finally:
        await store.teardown()
