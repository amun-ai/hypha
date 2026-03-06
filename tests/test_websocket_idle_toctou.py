"""Tests for TOCTOU race fix in WebsocketServer._do_idle_cleanup().

The race: between building the "to_disconnect" snapshot and actually calling
disconnect(), a connection may have received new messages (updating _last_seen).
Without the re-check guard the connection would be disconnected while active.

No mocks: uses plain async functions and a minimal server stub.
"""
import time
import asyncio
import pytest

pytestmark = pytest.mark.asyncio


class _SimpleWS:
    """Minimal WebSocket stand-in (no starlette dependency)."""
    pass


def _make_server(idle_timeout=60):
    """Create a minimal WebsocketServer instance bypassing __init__."""
    from hypha.websocket import WebsocketServer

    server = WebsocketServer.__new__(WebsocketServer)
    server._stop = False
    server._websockets = {}
    server._last_seen = {}
    server._idle_timeout = idle_timeout
    server._idle_cleanup_task = None
    server.store = None
    return server


async def test_stale_connection_is_disconnected():
    """A connection idle longer than the timeout must be disconnected."""
    server = _make_server(idle_timeout=60)
    now = time.time()

    key = "ws/client-stale"
    ws = _SimpleWS()
    server._websockets[key] = ws
    server._last_seen[key] = now - 120  # 2 min ago → stale

    disconnected = []

    async def fake_disconnect(websocket, reason, code):
        disconnected.append(websocket)

    server.disconnect = fake_disconnect

    await server._do_idle_cleanup()

    assert ws in disconnected, "Stale connection should have been disconnected"
    assert key not in server._last_seen, "_last_seen entry should be removed"


async def test_active_connection_is_not_disconnected():
    """A connection that is still within the idle window must not be touched."""
    server = _make_server(idle_timeout=60)
    now = time.time()

    key = "ws/client-active"
    ws = _SimpleWS()
    server._websockets[key] = ws
    server._last_seen[key] = now - 10  # 10 s ago → still active

    disconnected = []

    async def fake_disconnect(websocket, reason, code):
        disconnected.append(websocket)

    server.disconnect = fake_disconnect

    await server._do_idle_cleanup()

    assert ws not in disconnected, "Active connection should NOT be disconnected"


async def test_toctou_connection_becomes_active_during_cleanup():
    """
    TOCTOU fix: a connection that becomes active AFTER being added to the
    to_disconnect candidate list but BEFORE disconnect() is called must NOT
    be forcibly terminated.

    We simulate this by having the disconnect() of connection A update
    _last_seen[B] (mimicking a message arriving for B while A is being
    cleaned up).  Without the re-check guard, B would still be disconnected.
    With the fix, B is skipped.
    """
    server = _make_server(idle_timeout=60)
    now = time.time()

    key_a = "ws/client-a"
    key_b = "ws/client-b"
    ws_a = _SimpleWS()
    ws_b = _SimpleWS()

    server._websockets[key_a] = ws_a
    server._websockets[key_b] = ws_b
    # Both stale at snapshot time
    server._last_seen[key_a] = now - 120
    server._last_seen[key_b] = now - 90

    disconnected = []

    async def fake_disconnect(websocket, reason, code):
        disconnected.append(websocket)
        if websocket is ws_a:
            # While A is being disconnected, B receives a message and becomes active
            server._last_seen[key_b] = time.time()

    server.disconnect = fake_disconnect

    await server._do_idle_cleanup()

    assert ws_a in disconnected, "Stale connection A should be disconnected"
    assert ws_b not in disconnected, (
        "TOCTOU: connection B became active during cleanup and must NOT be disconnected"
    )


async def test_already_removed_connection_skipped():
    """Ghost _last_seen entry (no active WebSocket) is cleaned up without calling disconnect()."""
    server = _make_server(idle_timeout=60)
    now = time.time()

    key = "ws/client-gone"
    # Put it in _last_seen but NOT in _websockets (already cleaned up by disconnect handler)
    server._last_seen[key] = now - 120

    disconnected = []

    async def fake_disconnect(websocket, reason, code):
        disconnected.append(websocket)

    server.disconnect = fake_disconnect

    await server._do_idle_cleanup()

    assert disconnected == [], "No disconnect should be called for already-removed connection"
    assert key not in server._last_seen, "Ghost _last_seen entry must be removed by idle cleanup"


async def test_multiple_stale_connections_all_disconnected():
    """All stale connections are disconnected when none become active during the run."""
    server = _make_server(idle_timeout=60)
    now = time.time()

    keys = [f"ws/client-{i}" for i in range(5)]
    wss = [_SimpleWS() for _ in range(5)]

    for key, ws in zip(keys, wss):
        server._websockets[key] = ws
        server._last_seen[key] = now - 200  # All stale

    disconnected = []

    async def fake_disconnect(websocket, reason, code):
        disconnected.append(websocket)

    server.disconnect = fake_disconnect

    await server._do_idle_cleanup()

    assert set(disconnected) == set(wss), "All stale connections should be disconnected"
    for key in keys:
        assert key not in server._last_seen


async def test_disconnect_error_still_removes_last_seen():
    """If disconnect() raises, _last_seen is still cleaned up (no retry loop)."""
    server = _make_server(idle_timeout=60)
    now = time.time()

    key = "ws/client-erroring"
    ws = _SimpleWS()
    server._websockets[key] = ws
    server._last_seen[key] = now - 200

    async def failing_disconnect(websocket, reason, code):
        raise RuntimeError("WebSocket already closed")

    server.disconnect = failing_disconnect

    await server._do_idle_cleanup()

    # Even though disconnect raised, _last_seen must be cleaned up
    assert key not in server._last_seen, (
        "_last_seen entry should be removed even when disconnect() raises"
    )
