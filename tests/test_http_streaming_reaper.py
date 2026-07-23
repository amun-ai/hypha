"""#0011 — HTTP-streaming liveness reaper.

A svamp machine daemon on the HTTP-streaming (`/rpc`) transport that dies
HALF-OPEN (docker veth yanked, no FIN) was never reaped: the stream loop only
detects loss via `request.is_disconnected()`, which never fires without an
`http.disconnect` event, so the client's service registrations ghosted forever
(only a server restart cleared them). WebSockets are unaffected (they have a
`wait_for(receive(), idle_timeout)` reaper).

Fix (`hypha/http_rpc.py`): a background liveness sweep pings idle streaming
clients via the workspace manager's `ping_client` (RPC-layer ping to the
client's built-in service) and reaps non-responders — removing their service
registrations via `store.remove_client` → `delete_client`. Crucially, idle
alone NEVER reaps (idle-but-live is normal for streaming); only a client that
does NOT answer an active ping is reaped, so a healthy idle machine is never
false-reaped.
"""
import asyncio

import pytest

from hypha.core import UserInfo
from hypha.core.auth import create_scope
from hypha.core.store import RedisStore
from hypha.http_rpc import HTTPStreamingRPCServer

pytestmark = pytest.mark.asyncio


def _user(ws):
    return UserInfo(
        id="reaper-test",
        is_anonymous=False,
        email=None,
        parent=None,
        roles=[],
        scope=create_scope(f"{ws}#a", current_workspace=ws),
        expires_at=None,
    )


async def _make_server(store):
    server = HTTPStreamingRPCServer(store)
    # Fast, deterministic sweeps for the test.
    server._stream_ping_timeout = 0.3
    server._stream_idle_threshold = 0.0  # any registered idle stream is a candidate
    return server


async def test_dead_streaming_client_is_reaped():
    """A registered streaming client that does NOT answer an RPC ping (dead /
    half-open) is reaped: its stream is closed and its service registrations are
    removed."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = "ws-user-reaper-dead"
        client_id = "ghost-machine"
        conn_key = f"{ws}/{client_id}"

        # A genuinely-dead client: register the streaming connection but there is
        # no live RPC responder, so ping_client will time out.
        await server._create_connection(ws, client_id, _user(ws))
        assert conn_key in server._connections

        # A service registration for this client (the actual "ghost").
        redis = store.get_redis()
        service_key = f"services:public|test-svc:{ws}/{client_id}:built-in@app"
        await redis.set(service_key, b"{}")

        # Force it to be an idle candidate and sweep.
        server._last_activity[conn_key] = 0.0
        await server._do_liveness_sweep()

        # The stream + local state are gone, and the None close-sentinel was
        # queued.
        assert conn_key not in server._connections, "dead stream must be reaped"
        assert conn_key not in server._last_activity
        # The service registration (the ghost) was removed by delete_client.
        assert await redis.get(service_key) is None, (
            "the dead client's service registration must be removed"
        )
    finally:
        await store.teardown()


async def test_recently_active_streaming_client_is_not_pinged_or_reaped():
    """A stream with fresh upstream activity is NOT a sweep candidate — it is
    never pinged and never reaped (idle-gate, first line of false-reap defense)."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        # Non-zero threshold so 'fresh' activity is genuinely below it.
        server._stream_idle_threshold = 60.0
        ws = "ws-user-reaper-fresh"
        client_id = "busy-client"
        conn_key = f"{ws}/{client_id}"

        await server._create_connection(ws, client_id, _user(ws))
        # Fresh activity (now) — below the 60s threshold.
        import time as _t

        server._last_activity[conn_key] = _t.time()

        await server._do_liveness_sweep()

        assert conn_key in server._connections, (
            "an actively-communicating stream must never be reaped"
        )
        assert conn_key in server._last_activity
    finally:
        await store.teardown()


async def test_alive_candidate_is_refreshed_not_reaped():
    """When the liveness check reports the client ALIVE, the sweep must REFRESH
    its activity and NOT reap it — the active-ping gate that prevents
    false-reaping a healthy idle streaming client.

    We drive the alive path via the fail-safe return (no workspace manager ⇒
    `_ping_stream_client` returns True ⇒ 'alive'), which exercises the exact
    refresh-not-reap branch. (The real-`pong` path is the same branch and is
    validated end-to-end on a running server, where a genuinely-idle streaming
    client answers the RPC ping and survives.)"""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = "ws-user-reaper-alive"
        client_id = "idle-but-alive"
        conn_key = f"{ws}/{client_id}"

        await server._create_connection(ws, client_id, _user(ws))
        server._last_activity[conn_key] = 0.0  # stale ⇒ becomes a candidate

        saved = store._workspace_manager
        store._workspace_manager = None  # ⇒ _ping_stream_client returns True (alive)
        try:
            await server._do_liveness_sweep()
        finally:
            store._workspace_manager = saved

        assert conn_key in server._connections, (
            "an alive candidate must NOT be reaped"
        )
        # Its activity was refreshed away from the stale sentinel, so it will not
        # be re-pinged every single sweep.
        assert server._last_activity[conn_key] > 0.0, (
            "an alive candidate's activity must be refreshed"
        )
    finally:
        await store.teardown()
