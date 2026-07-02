"""Reproduce the prod WebSocket memory leak: superseded connections orphaned forever.

Prod evidence (valiant-goat, 0.21.103): ~533 `establish_websocket_communication`
coroutines suspended at `await asyncio.wait_for(websocket.receive(), timeout=...)`
(websocket.py ~line 520), with `live RedisRPCConnection (513) >> registry active
(70)`. Leaked conns had `evbus_set=True/filtered_set=True` — i.e. `disconnect()`
never ran, because the coroutine never exited the try block, so its finally
(which calls `conn.disconnect()`) never fired.

Root cause: the per-receive `wait_for(timeout=idle_timeout)` is reset by ANY
inbound frame — including the hypha-rpc client's application-level keepalive
`ping` every 20s — so it is a "time since last inbound frame" gap, not a liveness
check. When a client reconnects with the SAME client_id (supersede), the server
bumps the generation and overwrites `_websockets`/`_last_seen` to point at the new
connection, but does NOT tear down the old coroutine. If the old socket still
delivers frames (keepalive / half-open behind a proxy) the old coroutine parks at
`receive()` forever, and the idle sweeper can no longer see it (its `_last_seen`
entry now belongs to the new generation). The orphaned coroutine pins a full
`RedisRPCConnection` + event-bus handlers + ~50 RPC method-proxy coroutines.

Fix: on generation bump, cancel the old generation's task so its finally runs
cleanup immediately.
"""
import asyncio
import gc
import json
import os

import pytest

from hypha.core import RedisRPCConnection, UserInfo
from hypha.core.auth import create_scope
from hypha.core.store import RedisStore

pytestmark = pytest.mark.asyncio


class _FakeState:
    name = "CONNECTED"


class _BlockingWS:
    """A socket whose receive() blocks forever (a truly dead, silent half-open peer)."""

    def __init__(self):
        self.client_state = _FakeState()
        self.application_state = _FakeState()
        self.closed = False

    async def receive(self):
        await asyncio.Future()  # never resolves

    async def send_bytes(self, data):
        pass

    async def send_text(self, data):
        pass

    async def close(self, code=None, reason=None):
        self.closed = True


class _PingingWS(_BlockingWS):
    """A socket that keeps delivering client keepalive 'ping' frames every 50ms.

    This is the dangerous case: inbound frames keep resetting the per-receive idle
    timeout, so the receive loop never idle-times-out on its own.
    """

    async def receive(self):
        await asyncio.sleep(0.05)
        return {"text": json.dumps({"type": "ping"})}


async def _make_server(store):
    from fastapi import FastAPI
    from hypha.websocket import WebsocketServer

    store._app = FastAPI()
    return WebsocketServer(store)


def _user(ws):
    return UserInfo(
        id="wsleak",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )


async def test_superseded_websocket_generation_is_reaped():
    """A reconnect (same client_id) must tear down the old, still-pinging connection.

    Without the fix, the old generation's coroutine stays parked in receive()
    forever (its socket keeps delivering pings), leaking its RedisRPCConnection.
    """
    # Long idle timeout so the ONLY thing that can reap the old generation is the
    # active supersede-cancel (not the idle timeout).
    os.environ["HYPHA_WS_IDLE_TIMEOUT"] = "600"
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws_name = "ws-user-supersede"
        user = _user(ws_name)
        await store.register_workspace(
            dict(name=ws_name, description="x", owners=["wsleak"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        client_id = "client-reconnecting"

        created_before = RedisRPCConnection._created_total_int
        closed_before = RedisRPCConnection._closed_total_int

        # Generation 1: a pinging connection (keeps its socket "active").
        ws1 = _PingingWS()
        t1 = asyncio.create_task(
            server.establish_websocket_communication(ws1, ws_name, client_id, user)
        )
        await asyncio.sleep(0.4)
        assert not t1.done(), "gen1 should be running (parked in receive loop)"

        # Generation 2: SAME client_id reconnects (supersede).
        ws2 = _PingingWS()
        t2 = asyncio.create_task(
            server.establish_websocket_communication(ws2, ws_name, client_id, user)
        )
        # Give the supersede-cancel + gen1's finally time to run.
        await asyncio.sleep(0.6)

        # The OLD generation must have been reaped (its finally ran cleanup).
        assert t1.done(), (
            "SUPERSEDE LEAK: the old websocket generation was not reaped after a "
            "reconnect — its coroutine is orphaned in receive() forever, pinning a "
            "RedisRPCConnection + event-bus handlers + ~50 method-proxy coroutines."
        )
        # gen2 must still be alive (we did not clobber the new connection).
        assert not t2.done(), "the new generation must remain connected"

        # closed_total must have advanced for the reaped generation.
        gc.collect()
        closed_delta = RedisRPCConnection._closed_total_int - closed_before
        created_delta = RedisRPCConnection._created_total_int - created_before
        assert closed_delta >= 1, (
            f"the superseded connection's disconnect() must run (closed_total++); "
            f"created={created_delta}, closed={closed_delta}"
        )

        # Cleanly stop gen2.
        t2.cancel()
        try:
            await t2
        except BaseException:
            pass
    finally:
        await store.teardown()
        os.environ.pop("HYPHA_WS_IDLE_TIMEOUT", None)


async def test_dead_silent_websocket_is_reaped_by_idle_timeout():
    """Baseline: a truly dead (silent) socket is still reaped by the idle timeout.

    Guards against a regression where the supersede change breaks the normal
    single-connection idle-timeout path.
    """
    os.environ["HYPHA_WS_IDLE_TIMEOUT"] = "1"
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws_name = "ws-user-dead"
        user = _user(ws_name)
        await store.register_workspace(
            dict(name=ws_name, description="x", owners=["wsleak"],
                 persistent=False, read_only=False),
            overwrite=True,
        )

        closed_before = RedisRPCConnection._closed_total_int
        ws = _BlockingWS()
        t = asyncio.create_task(
            server.establish_websocket_communication(ws, ws_name, "client-dead", user)
        )
        # Wait past the 1s idle timeout.
        await asyncio.sleep(2.5)
        assert t.done(), "a silent dead socket must be reaped by the idle timeout"
        assert ws.closed, "the idle-timeout path must close the websocket"
        assert RedisRPCConnection._closed_total_int - closed_before >= 1
    finally:
        await store.teardown()
        os.environ.pop("HYPHA_WS_IDLE_TIMEOUT", None)
