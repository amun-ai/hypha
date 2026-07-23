"""Tests for graceful-shutdown connection draining (F4).

On a server rollout/redeploy, abruptly cutting long-lived connections forces
clients to detect the dead pod only via their heartbeat/read timeout, producing
a thundering-herd reconnection storm. The fix proactively signals connected
clients to reconnect:

- WebSocket clients receive an explicit close frame (1001 GOING_AWAY).
- HTTP-streaming clients receive the ``None`` sentinel that ends their stream
  cleanly (clean EOF -> immediate reconnect).

No mocks: these exercise the real ``close_all_clients`` / ``close_all_streams``
methods against minimal in-process stand-ins, mirroring the pattern in
test_websocket_idle_toctou.py.
"""
import asyncio

import pytest
from fastapi import status

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- #
# WebSocket: close_all_clients
# --------------------------------------------------------------------------- #
class _ClientState:
    """Minimal stand-in for starlette WebSocketState (only .name is used)."""

    def __init__(self, name="CONNECTED"):
        self.name = name


class _SimpleWS:
    """Minimal WebSocket stand-in recording close() calls."""

    def __init__(self, state="CONNECTED", raise_on_close=False):
        self.client_state = _ClientState(state)
        self.raise_on_close = raise_on_close
        self.closed_with = None  # (code, reason) once closed

    async def close(self, code=status.WS_1000_NORMAL_CLOSURE, reason=None):
        if self.raise_on_close:
            raise RuntimeError("socket already gone")
        self.closed_with = (code, reason)
        self.client_state.name = "CLOSED"


def _make_ws_server():
    """Create a minimal WebsocketServer instance bypassing __init__."""
    from hypha.websocket import WebsocketServer

    server = WebsocketServer.__new__(WebsocketServer)
    server._stop = False
    server._websockets = {}
    return server


async def test_close_all_clients_signals_every_connection_with_going_away():
    """Every active websocket must receive a 1001 GOING_AWAY close frame."""
    server = _make_ws_server()
    sockets = {f"ws/client-{i}": _SimpleWS() for i in range(5)}
    server._websockets = dict(sockets)

    count = await server.close_all_clients()

    assert count == 5
    for ws in sockets.values():
        assert ws.closed_with is not None, "Client was not signaled to close"
        code, reason = ws.closed_with
        assert code == status.WS_1001_GOING_AWAY
        assert reason == "Server is shutting down"


async def test_close_all_clients_empty_is_noop():
    """Draining with no active connections returns 0 and does not raise."""
    server = _make_ws_server()
    assert await server.close_all_clients() == 0


async def test_close_all_clients_skips_already_closed():
    """A socket that is no longer CONNECTED must not be closed again."""
    server = _make_ws_server()
    already = _SimpleWS(state="CLOSED")
    live = _SimpleWS()
    server._websockets = {"ws/a": already, "ws/b": live}

    await server.close_all_clients()

    # already-closed socket: close() never invoked (closed_with stays None)
    assert already.closed_with is None
    assert live.closed_with is not None


async def test_close_all_clients_continues_when_one_socket_errors():
    """One unresponsive socket must not prevent draining the others."""
    server = _make_ws_server()
    bad = _SimpleWS(raise_on_close=True)
    good1 = _SimpleWS()
    good2 = _SimpleWS()
    server._websockets = {"ws/bad": bad, "ws/g1": good1, "ws/g2": good2}

    count = await server.close_all_clients()

    assert count == 3  # all were attempted
    assert good1.closed_with is not None
    assert good2.closed_with is not None


async def test_close_all_clients_custom_code_and_reason():
    """Caller-provided code/reason are honored."""
    server = _make_ws_server()
    ws = _SimpleWS()
    server._websockets = {"ws/a": ws}

    await server.close_all_clients(code=status.WS_1012_SERVICE_RESTART, reason="bye")

    assert ws.closed_with == (status.WS_1012_SERVICE_RESTART, "bye")


# --------------------------------------------------------------------------- #
# HTTP streaming: close_all_streams
# --------------------------------------------------------------------------- #
def _make_http_server():
    """Create a minimal HTTPStreamingRPCServer instance bypassing __init__."""
    from hypha.http_rpc import HTTPStreamingRPCServer

    server = HTTPStreamingRPCServer.__new__(HTTPStreamingRPCServer)
    server._connections = {}
    server._connection_info = {}
    server._lock = asyncio.Lock()
    # Liveness-sweep state that close_all_streams() touches during shutdown
    # (normally set by __init__, which this helper bypasses). #0011.
    server._sweep_task = None
    server._stop_sweep = False
    return server


async def test_close_all_streams_pushes_sentinel_to_every_queue():
    """Each active stream queue must receive the None close sentinel."""
    server = _make_http_server()
    queues = {}
    for i in range(4):
        q = asyncio.Queue(maxsize=100)
        queues[f"ws/client-{i}"] = q
        server._connections[f"ws/client-{i}"] = q

    count = await server.close_all_streams()

    assert count == 4
    for q in queues.values():
        assert q.qsize() == 1
        assert q.get_nowait() is None, "Stream was not signaled to close"


async def test_close_all_streams_empty_is_noop():
    """Draining with no active streams returns 0 and does not raise."""
    server = _make_http_server()
    assert await server.close_all_streams() == 0


async def test_close_all_streams_tolerates_full_queue():
    """A backed-up (full) queue must not abort draining of the others."""
    server = _make_http_server()
    full = asyncio.Queue(maxsize=1)
    full.put_nowait(b"pending")  # now full
    ok = asyncio.Queue(maxsize=100)
    server._connections = {"ws/full": full, "ws/ok": ok}

    # Must not raise despite the full queue.
    count = await server.close_all_streams()

    # Only the queue with capacity was signaled.
    assert count == 1
    assert ok.get_nowait() is None


async def test_close_all_streams_sentinel_matches_stream_close_contract():
    """The sentinel value must be exactly None (what stream_messages() checks)."""
    server = _make_http_server()
    q = asyncio.Queue(maxsize=100)
    server._connections = {"ws/a": q}

    await server.close_all_streams()

    # stream_messages() does `if data is None: break` — anything else would not
    # terminate the stream cleanly.
    assert q.get_nowait() is None
