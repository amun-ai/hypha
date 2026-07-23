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


async def test_dead_streaming_client_reaped_after_consecutive_ping_misses():
    """A dead/half-open client (no RPC-ping responder) is reaped ONLY after
    _stream_max_ping_misses CONSECUTIVE misses — not on the first miss — and its
    service registrations are removed."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        server._stream_max_ping_misses = 2
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

        server._last_activity[conn_key] = 0.0  # idle candidate

        # First miss: NOT reaped yet (below the consecutive-miss threshold).
        await server._do_liveness_sweep()
        assert conn_key in server._connections, "must not reap on the first miss"
        assert server._ping_misses.get(conn_key) == 1
        assert await redis.get(service_key) is not None

        # Second consecutive miss: reaped.
        await server._do_liveness_sweep()
        assert conn_key not in server._connections, (
            "must reap after max consecutive misses"
        )
        assert conn_key not in server._last_activity
        assert conn_key not in server._ping_misses
        assert await redis.get(service_key) is None, (
            "the dead client's service registration must be removed"
        )
    finally:
        await store.teardown()


async def test_single_ping_miss_does_not_reap():
    """A single ping miss must NEVER reap — a live-but-loaded streaming client
    can miss one slow POST round-trip (FLAG 2). Reaping requires
    _stream_max_ping_misses CONSECUTIVE misses."""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        server._stream_max_ping_misses = 3
        ws = "ws-user-reaper-transient"
        client_id = "loaded-machine"
        conn_key = f"{ws}/{client_id}"

        await server._create_connection(ws, client_id, _user(ws))
        server._last_activity[conn_key] = 0.0

        await server._do_liveness_sweep()  # one miss

        assert conn_key in server._connections, (
            "a single ping miss must not reap a (possibly just-loaded) client"
        )
        assert server._ping_misses.get(conn_key) == 1
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
        server._ping_misses[conn_key] = 2  # had earlier transient misses

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
        # A successful ping RESETS the consecutive-miss counter — earlier
        # transient misses must not accumulate toward a reap.
        assert conn_key not in server._ping_misses, (
            "an answered ping must reset the consecutive-miss counter"
        )
    finally:
        await store.teardown()


# ── #0012: wedged-stream (downstream queue full) force-recycle ───────────────


async def test_drop_counter_increments_when_full_and_resets_on_drain():
    """send_to_queue counts CONSECUTIVE downstream-queue drops, and a successful
    enqueue (queue draining) resets the counter — so a slow-but-draining client
    never accumulates toward a force-recycle. #0012"""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = "ws-user-wedge-count"
        client_id = "machine"
        conn_key = f"{ws}/{client_id}"

        conn, queue = await server._create_connection(ws, client_id, _user(ws))
        # send_to_queue is the connection's message handler.
        send = conn._handle_message

        # Fill the queue to capacity so further enqueues drop.
        while not queue.full():
            queue.put_nowait(b"x")

        await send(b"overflow")
        assert server._drop_counts.get(conn_key) == 1
        await send(b"overflow")
        assert server._drop_counts.get(conn_key) == 2

        # Drain one slot ⇒ next enqueue succeeds ⇒ counter RESETS.
        queue.get_nowait()
        await send(b"ok")
        assert conn_key not in server._drop_counts, (
            "a successful enqueue must reset the consecutive-drop counter"
        )
    finally:
        await store.teardown()


async def test_wedged_stream_is_force_recycled():
    """A connection whose downstream queue is persistently full (drops past the
    threshold, i.e. not draining) is force-recycled: its stuck stream task is
    cancelled so the client reconnects with a fresh queue. #0012"""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        ws = "ws-user-wedge"
        client_id = "wedged-machine"
        conn_key = f"{ws}/{client_id}"

        await server._create_connection(ws, client_id, _user(ws))
        # Simulate the response task stuck on a backpressured yield.
        stuck = asyncio.create_task(asyncio.sleep(3600))
        server._stream_tasks[conn_key] = stuck
        # Persistent drops at/above the threshold ⇒ wedged.
        server._drop_counts[conn_key] = server._stream_max_drops

        await server._do_liveness_sweep()

        # The stuck stream task was cancelled (force-recycle) and the counter
        # cleared.
        try:
            await stuck
        except asyncio.CancelledError:
            pass
        assert stuck.cancelled(), "a wedged stream's task must be cancelled"
        assert conn_key not in server._drop_counts
    finally:
        await store.teardown()


async def test_below_threshold_stream_is_not_recycled():
    """A connection with drops BELOW the threshold (transiently backed up, still
    draining) is NOT force-recycled. #0012"""
    store = RedisStore(None, redis_uri=None)
    await store.init(reset_redis=True)
    try:
        server = await _make_server(store)
        server._stream_idle_threshold = 1e9  # keep it out of the idle-ping path
        ws = "ws-user-wedge-ok"
        client_id = "busy-but-draining"
        conn_key = f"{ws}/{client_id}"

        await server._create_connection(ws, client_id, _user(ws))
        stuck = asyncio.create_task(asyncio.sleep(3600))
        server._stream_tasks[conn_key] = stuck
        server._drop_counts[conn_key] = server._stream_max_drops - 1  # below

        await server._do_liveness_sweep()
        await asyncio.sleep(0)

        assert not stuck.cancelled(), (
            "a still-draining stream (below drop threshold) must not be recycled"
        )
        assert server._drop_counts.get(conn_key) == server._stream_max_drops - 1
        stuck.cancel()
    finally:
        await store.teardown()
